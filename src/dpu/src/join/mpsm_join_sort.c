/*
 * Join with multiple tasklets
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <mutex.h>
#include <string.h>

#include "argument.h"

#define NR_TASKLETS 12

#define MUTEX_SIZE 52

#define BLOCK_LOG 11
#define BLOCK_SIZE (1 << BLOCK_LOG)
#define ELEM_PER_BLOCK (BLOCK_SIZE >> 3)

#define RADIX_DIGIT 4
#define RADIX (1 << RADIX_DIGIT)

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host sort_merge_partitioning_arg param_sort_merge_partitioning;
__host dpu_results_t dpu_results;

// The number of instructions
__host uint32_t nb_instrs;
// The number of cycles
__host uint32_t nb_cycles;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

uint32_t NR_INSTRUCTIONS = 0;
uint32_t NR_CYCLES = 0;

// Packet Value
uint32_t PACKET_BYTE = 0;
// Total number of elements in all packets
uint32_t TOTAL_ELEM = 0;

// Start Addr of a table
char *key_src_addr;
char *key_sorted_addr;
// Start Addr of each partition (when R)
char *partition_idx_addr;

// The number of packets thread will handle
uint32_t packets_per_th = 0;
// Min/Max Value from each tasklet
uint32_t max_all = 0;
uint32_t min_all = 1 << 20;

// R Table or not
bool r_table = false;

// Variables for Histogram
uint32_t hist_interval;
uint32_t histogram_bucket_num = 0;
char *write_buffer;

// Variables for Sort
void *sorted_buff1;
void *sorted_buff2;

/* Function for Histogram */
void SetHistogram(int32_t idx, uint32_t value)
{
	uint32_t *wb = (uint32_t *)write_buffer + idx;
	(*wb) = value;
}
void IncrHistogram(int32_t idx)
{
	uint32_t *wb = (uint32_t *)write_buffer + idx;
	*wb += 1;
}
uint32_t GetHistogram(int32_t idx)
{
	uint32_t *wb = (uint32_t *)write_buffer + idx;
	return *wb;
}
uint32_t GetIncrHistogram(int32_t idx)
{
	uint32_t *wb = (uint32_t *)write_buffer + idx;
	*wb += 1;
	return (*wb - 1);
}
/* Function for Histogram */

void quickSort(tuplePair_t* arr, uint32_t* stack, uint32_t size)
{
    // Initialize of the variables
    int32_t top = -1;
    uint32_t start, end, pivot;

    // Push initial values to stack
    stack[++top] = 0;
    stack[++top] = size;

    while (top >= 0)
    {
        // Pop
        end = stack[top--];
        start = stack[top--];

        if (end - start < 2) continue;

        // Between two points
        pivot = start + (end - start) / 2;

        /* Partition Starts */

        int32_t left = start;
        int32_t right = end - 1;
        tuplePair_t temp;

        // Value of the pivot
        uint32_t pivot_val = arr[pivot].lvalue;

        // Swap
        temp = arr[pivot];
        arr[pivot] = arr[end - 1];
        arr[end - 1] = temp;

        while (left < right)
        {
            if (arr[left].lvalue < pivot_val) left++;
            else if (arr[right].lvalue >= pivot_val) right--;
            else
            {
                // Swap
                temp = arr[left];
                arr[left] = arr[right];
                arr[right] = temp;
            }
        }

        uint32_t idx = right;
        if (arr[right].lvalue < pivot) idx++;

        // Swap
        temp = arr[end - 1];
        arr[end - 1] = arr[idx];
        arr[idx] = temp;

        pivot = idx;

        /* Partition Ends */

        stack[++top] = pivot + 1;
        stack[++top] = end;

        stack[++top] = start;
        stack[++top] = pivot;
    }
}

void radixSort(tuplePair_t* arr, tuplePair_t* arr_buff, uint32_t size)
{
    uint32_t digit = 0, factor = 1;
    uint32_t idx = 0;

    uint32_t max = arr[0].lvalue;
    // Calculate the max value in the array
    for (uint32_t i = 1; i < 8; i++)
        if (max < arr[i].lvalue) max = arr[i].lvalue;
    
    // Calculate the digit of the max value
    for (uint32_t i = max; i > 0; i >>= 3) digit++;

    // Radix Sort
    for (uint32_t i = 0; i < digit + 1; i++)
    {
        for (uint32_t j = 0; j < 8; j++)
        {
            // Put into the queue
            for (uint32_t k = 0; k < size; k++) 
                if (((arr[k].lvalue / factor) & 0x0111) == j) arr_buff[idx++] = arr[k];
        }

        factor <<= 3;

        for (uint32_t j = 0; j < size; j++) arr[j] = arr_buff[j];
        idx = 0;
    }
}

void mergeSort(tuplePair_t* arr, tuplePair_t* arr_buff, uint32_t elem_num)
{
    // Merge sort without recursion
    for (uint32_t arr_size = 1; arr_size < elem_num; arr_size <<= 1)
    {
	    // Index for sorted array
	    uint32_t left = 0;
	    uint32_t right = left + arr_size;

        // Index for new sorted array
        uint32_t sorted_idx = 0;

        // Where to stop
        uint32_t end = right + arr_size;

        for (uint32_t iter = 0; iter < (elem_num / (arr_size << 1)); iter++)
        {
            // Index to compare
            uint32_t i = left;
            uint32_t j = right;

            // When write all the elements to buffer
            while (sorted_idx < end)
            {
                if (arr[i].lvalue < arr[j].lvalue) arr_buff[sorted_idx++] = arr[i++];
                else arr_buff[sorted_idx++] = arr[j++];
                // When one of the array ends
                if ((i >= right) || (j >= end)) break;
            }

            // Copy the remains
            if (i >= right)
            {
                while (j < end) arr_buff[sorted_idx++] = arr[j++];
            }
            else if (j >= end)
            {
                while (i < right) arr_buff[sorted_idx++] = arr[i++];
            }

            if (iter == (elem_num / (arr_size << 1)))
            {
                if (elem_num % (arr_size << 1) != 0)
                {
                    iter--;
                    end = elem_num;
                    continue;
                }
            }

            left += (arr_size * 2);
            right += (arr_size * 2);
            
            end = right + arr_size;
        }

        // Copy the buffer to the original
        for (uint32_t idx = 0; idx < elem_num; idx++) arr[idx] = arr_buff[idx];
    }
}

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        // Reset the heap
        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        perfcounter_config(COUNT_CYCLES, true);
    }

    barrier_wait(&my_barrier);

    //////////////////////////////////////////////
	// Variable Setup
	//////////////////////////////////////////////

    if (tasklet_id == 0)
    {
        key_src_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_partitioning.r_packet_start_byte;
        key_sorted_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_partitioning.r_sorted_start_byte;
        partition_idx_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_partitioning.histogram_addr_start_byte;
    }
    else if (tasklet_id == 1)
    {
        packets_per_th = BLOCK_SIZE / sizeof(data_packet_u64_128_t);
    }
    else if (tasklet_id == 2)
    {
        PACKET_BYTE = param_sort_merge_partitioning.num_packets * sizeof(data_packet_u64_128_t);
    }
    else if (tasklet_id == 3)
    {
        write_buffer = (char *)mem_alloc(32 * 1024);
    }

    barrier_wait(&my_barrier);

    // Allocate Buffer
    void* tuples_read_buff = NULL;
    tuples_read_buff = mem_alloc(BLOCK_SIZE);


    /*
     * Phase 1.
     *    - Check the number of total elements in packets
     *    - Find the max/min value of all elements and its digit
     */

    // Max value from each tasklet
    uint32_t max_val = 0;
    // Min value from each tasklet
    uint32_t min_val = 1 << 20;

    // Buffer for each packet
    data_packet_u64_128_t *packet;
    // Buffer for each element in packet
    tuplePair_t *elem_packet;

    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            BLOCK_SIZE);
        
        // Total number of elements in packets
        uint32_t num_elem = 0;

        packet = (data_packet_u64_128_t *) tuples_read_buff;
        // Packets per each tasklet
        for (uint32_t p = 0; p < packets_per_th; p++)
        {
            elem_packet = (tuplePair_t *)(&packet[p]);
            // Tuples per packet
            for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
            {
                if (elem_packet[e].lvalue == 0) continue;
                else
                {
                    num_elem++;

                    // Update the max value
                    if (max_val < elem_packet[e].lvalue) max_val = elem_packet[e].lvalue;
                    // Update the min value
                    if (min_val > elem_packet[e].lvalue) min_val = elem_packet[e].lvalue;
                }
            }
        }

        mutex_lock(&(mutex_atomic[49]));
        // Count the total number of elements
        TOTAL_ELEM += num_elem;
        mutex_unlock(&(mutex_atomic[49]));
    }

    barrier_wait(&my_barrier);
    if (tasklet_id == 4) histogram_bucket_num = TOTAL_ELEM / (ELEM_PER_BLOCK * 0.5);
    
    // Global min/max value
    mutex_lock(&(mutex_atomic[39]));
    if (max_val > max_all) max_all = max_val;
    if (min_val < min_all) min_all = min_val;
    mutex_unlock(&(mutex_atomic[39]));

    barrier_wait(&my_barrier);
    // if (tasklet_id == 8) printf("TOTAL_ELEM\t%u\thistogram_bucket_num\t%u\t\n", TOTAL_ELEM, histogram_bucket_num);

    /*
     * Phase 2.
     *    - Build a histogram to partition each relation
     */

    if (tasklet_id == 4)
    {
        // Buffer used for histogram build
        // histogram_buff = (uint16_t *) mem_alloc((histogram_bucket_num + 1) * sizeof(uint16_t));
        hist_interval = (max_all - min_all) / histogram_bucket_num;

        // printf("hist_interval\t%u\t\n", hist_interval);

        // Memset Histogram
        for (uint32_t i = 0; i < histogram_bucket_num + 1; i++)
        {
            SetHistogram(i, 0);
        }
    }

    barrier_wait(&my_barrier);

    // Target bucket of histogram
    uint32_t dest_bucket = 0;
    // Build Histogram
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            BLOCK_SIZE);

        packet = (data_packet_u64_128_t *) tuples_read_buff;
        // Packets per each tasklet
        for (uint32_t p = 0; p < packets_per_th; p++)
        {
            elem_packet = (tuplePair_t *)(&packet[p]);
            
            // Tuples per packet
            for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
            {
                if (elem_packet[e].lvalue == 0) continue;
                else
                {   
                    dest_bucket = (elem_packet[e].lvalue - min_all) / hist_interval;

                    // Histgoram build
                    mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));
                    // histogram_buff[dest_bucket]++;
                    IncrHistogram(dest_bucket);
                    mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    // // Check the elements in histogram
    // if (tasklet_id == 10)
    // {
    //     uint32_t *wb;
    //     for (uint32_t i = 0; i < histogram_bucket_num + 1; i++)
    //     {
    //         wb = (uint32_t *)write_buffer + i;
    //         printf("histogram_bucket[%u]\t%u\t\n", i, *wb);
    //     }
    // }



    /*
     * Phase 3.
     *    - Build a cumulative histogram for address
     *    - Reallocate each tuples based on its range
     */

    // Buffer used for histogram build
    if (tasklet_id == 5)
    {
        uint32_t temp_hist = GetHistogram(0);
        uint32_t temp_hist2 = GetHistogram(0);

        SetHistogram(0, 0);

        for (uint32_t idx = 1; idx < histogram_bucket_num + 1; idx++)
        {
            temp_hist2 = GetHistogram(idx);
            SetHistogram(idx, GetHistogram(idx - 1) + temp_hist);
            temp_hist = temp_hist2;
        }

        // uint32_t *wb;
        // for (uint32_t i = 0; i < histogram_bucket_num + 1; i++)
        // {
        //     wb = (uint32_t *)write_buffer + i;
        //     printf("TOTAL_ELEM\t%u\thistogram_bucket_num\t%u\tcumulative_histogram_bucket[%u]\t%u\t\n", TOTAL_ELEM, histogram_bucket_num, i, *wb);
        // }
    }

    barrier_wait(&my_barrier);

    // Index of the histogram
    uint32_t hist = 0;

    // Store elements in random manner
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            BLOCK_SIZE);

        packet = (data_packet_u64_128_t *) tuples_read_buff;
        
        // Packets per each tasklet
        for (uint32_t p = 0; p < packets_per_th; p++)
        {
            elem_packet = (tuplePair_t *)(&packet[p]);
            
            // Tuples per packet
            for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
            {
                if (elem_packet[e].lvalue == 0) continue;
                else
                {   
                    dest_bucket = (elem_packet[e].lvalue - min_all) / hist_interval;

                    mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));

                    hist = GetIncrHistogram(dest_bucket);
                    // if (hist == 0) printf("This is the first bucket\t%u\t\n", elem_packet[e].lvalue);
                    mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));

                    mram_write(
                        &(elem_packet[e]),
                        (__mram_ptr void *)(key_sorted_addr + hist * sizeof(tuplePair_t)),
                        sizeof(tuplePair_t));
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    

    /*
     * Phase 4.
     *    - Sort elements in MRAM
     */
    
    // Allocate buffer for tuples
    if (tasklet_id == 6) sorted_buff1 = mem_alloc(BLOCK_SIZE);
    else if (tasklet_id == 7) sorted_buff2 = mem_alloc(BLOCK_SIZE);

    for (uint32_t bucket_id = tasklet_id + 1; bucket_id < histogram_bucket_num + 1; bucket_id += NR_TASKLETS)
    {
        uint32_t *wb = (uint32_t *)write_buffer + bucket_id;
        uint32_t *prev_wb = (uint32_t *)write_buffer + (bucket_id - 1);

        int32_t elem_num = 0;
        if (bucket_id == histogram_bucket_num) elem_num = TOTAL_ELEM - (*wb);
        else elem_num = (*wb) - (*prev_wb);

        if (elem_num <= 0) continue;

        // Read from MRAM
        mram_read(
            (__mram_ptr void const *)(key_sorted_addr + (*prev_wb) * sizeof(tuplePair_t)),
            tuples_read_buff,
            elem_num * sizeof(tuplePair_t));

        mutex_lock(&(mutex_atomic[tasklet_id & 0x01]));

        if ((tasklet_id & 0x01) == 0)
        {
            // quickSort(tuples_read_buff, (uint32_t *) sorted_buff1, elem_num);
            radixSort(tuples_read_buff, (tuplePair_t *) sorted_buff1, elem_num);
            // mergeSort(tuples_read_buff, sorted_buff1, elem_num);
        }
        else
        {
            // quickSort(tuples_read_buff, (uint32_t *) sorted_buff2, elem_num);
            radixSort(tuples_read_buff, (tuplePair_t *) sorted_buff2, elem_num);
            // mergeSort(tuples_read_buff, sorted_buff2, elem_num);
        }

        if (bucket_id == 23)
        {
            for (uint32_t i = 0; i < elem_num; i++)
            {
                elem_packet = (tuplePair_t *) tuples_read_buff;
                printf("tuples_read_buff[%u]\t%u\t\n", i, elem_packet[i].lvalue);
            }
        }

        mutex_unlock(&(mutex_atomic[tasklet_id &0x01]));

        // Write back to MRAM
        mram_write(tuples_read_buff,
            (__mram_ptr void *)(key_sorted_addr + (*prev_wb) * sizeof(tuplePair_t)),
            elem_num * sizeof(tuplePair_t));
    }

    barrier_wait(&my_barrier);
    return 0;
}
