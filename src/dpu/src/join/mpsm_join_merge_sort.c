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

// The number of tuples thread will handle
uint32_t tuples_per_th = 0;
// Min/Max Value from each tasklet
uint32_t max_all = 0;
uint32_t min_all = 1 << 20;

// R Table or not
bool r_table = false;

// Variables for Histogram
uint32_t hist_interval;
uint32_t histogram_bucket_num = 0;
uint32_t *histogram_buff;

// Variables for Sort
void *sorted_buff1;
void *sorted_buff2;

/* Function for Histogram */
void SetHistogram(int32_t idx, uint32_t value)
{
	histogram_buff[idx] = value;
}
void IncrHistogram(int32_t idx)
{
	histogram_buff[idx] += 1;
}
uint32_t GetHistogram(int32_t idx)
{
	return histogram_buff[idx];
}
uint32_t GetIncrHistogram(int32_t idx)
{
	histogram_buff[idx] += 1;
    return (histogram_buff[idx] - 1);
}
/* Function for Histogram */

void mergeSort(tuplePair_t* arr, tuplePair_t* arr_buff, uint32_t elem_num)
{
    if (elem_num > ELEM_PER_BLOCK) elem_num = ELEM_PER_BLOCK;

    // Merge sort without recursion
    for (uint32_t arr_size = 1; arr_size < elem_num; arr_size <<= 1)
    {
        // Index for sorted array
        uint32_t left = 0;
        uint32_t right = left + arr_size;

        // Where to stop
        uint32_t end = right + arr_size;
        if (end > elem_num) end = elem_num;

        // Index for new sorted array
        uint32_t sorted_idx = 0;

        uint32_t iter_loop = elem_num / (arr_size << 1);
        if (elem_num % (arr_size << 1) > arr_size) iter_loop++;

        for (uint32_t iter = 0; iter < iter_loop; iter++)
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

            left += (arr_size << 1);
            right += (arr_size << 1);

            if (right >= elem_num) 
            {
                arr_buff[sorted_idx] = arr[left];
                break;
            }
            
            end = right + arr_size;
            if (end > elem_num) end = elem_num;
        }

        // Copy the buffer to the original
        for (uint32_t idx = 0; idx < elem_num; idx++) arr[idx] = arr_buff[idx];
    }
}

void compareMerge(tuplePair_t* arr, tuplePair_t* cmp_arr, tuplePair_t* result, uint32_t addr)
{
    uint32_t arr_idx = 0;
    uint32_t cmp_idx = 0;
    uint32_t idx = 0;

    while (arr_idx < ELEM_PER_BLOCK)
    {
        if (arr[arr_idx].lvalue >= cmp_arr[cmp_idx].lvalue) result[idx] = cmp_arr[cmp_idx++];
        else result[idx] = arr[arr_idx++];

        if (idx == ELEM_PER_BLOCK)
        {
            mram_write(result,
                (__mram_ptr void *)(key_sorted_addr + (addr * sizeof(tuplePair_t))),
                BLOCK_SIZE);
            
            idx = 0;
        }

        if (cmp_idx == ELEM_PER_BLOCK) 
        {
            while (arr_idx < ELEM_PER_BLOCK) result[idx++] = arr[arr_idx++];
        }
    }

    while (cmp_idx < ELEM_PER_BLOCK) result[idx++] = cmp_arr[cmp_idx++];

    mram_write(result,
        (__mram_ptr void *)(key_sorted_addr + (addr * sizeof(tuplePair_t)) + BLOCK_SIZE),
        BLOCK_SIZE);
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
        tuples_per_th = BLOCK_SIZE / sizeof(tuplePair_t);
        PACKET_BYTE = param_sort_merge_partitioning.num_packets * param_sort_merge_partitioning.packet_size;
        histogram_buff = (uint32_t*)mem_alloc(32 * 1024);
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

    // Buffer for each element in packet
    tuplePair_t *elem_packet;

    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        uint32_t elems_to_validate = ((PACKET_BYTE) - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >>3);
    
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));
        
        // Total number of elements in packets
        uint32_t num_elem = 0;

        elem_packet = (tuplePair_t *) tuples_read_buff;
        // Tuples per each tasklet
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            elem_packet = (tuplePair_t *) tuples_read_buff;
            
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

        mutex_lock(&(mutex_atomic[49]));
        // Count the total number of elements
        TOTAL_ELEM += num_elem;
        mutex_unlock(&(mutex_atomic[49]));
    }

    barrier_wait(&my_barrier);
    if (tasklet_id == 4) histogram_bucket_num = TOTAL_ELEM / (ELEM_PER_BLOCK * 0.4);
    
    // Global min/max value
    mutex_lock(&(mutex_atomic[39]));
    if (max_val > max_all) max_all = max_val;
    if (min_val < min_all) min_all = min_val;
    mutex_unlock(&(mutex_atomic[39]));

    barrier_wait(&my_barrier);



    /*
     * Phase 2.
     *    - Build a histogram to partition each relation
     */

    if (tasklet_id == 4)
    {
        // Buffer used for histogram build
        hist_interval = (max_all - min_all) / histogram_bucket_num;

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
        uint32_t elems_to_validate = ((PACKET_BYTE) - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >>3);

        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Packets per each tasklet
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            elem_packet = (tuplePair_t *) tuples_read_buff;
            
            if (elem_packet[e].lvalue == 0) continue;
            else
            {
                dest_bucket = (elem_packet[e].lvalue - min_all) / hist_interval;

                if (dest_bucket >= histogram_bucket_num)
                {
                    dest_bucket = (histogram_bucket_num -1);
                }

                // Histgoram build
                mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));
                IncrHistogram(dest_bucket);
                mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));
            }
        }
    }

    barrier_wait(&my_barrier);



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

        if (TOTAL_ELEM != GetHistogram(histogram_bucket_num))
        {
            printf("ERROR! TOTAL_ELEM != GetHistogram(histogram_bucket_num)\n");
        }
    }

    barrier_wait(&my_barrier);

    // Index of the histogram
    uint32_t hist = 0;

    // Store elements in random manner
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        uint32_t elems_to_validate = ((PACKET_BYTE) - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >>3);

        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Tuples per each tasklet
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            elem_packet = (tuplePair_t *)tuples_read_buff;
            
            if (elem_packet[e].lvalue == 0) continue;
            else
            {
                dest_bucket = (elem_packet[e].lvalue - min_all) / hist_interval;

                if (dest_bucket >= histogram_bucket_num)
                {
                    dest_bucket = (histogram_bucket_num -1);
                }

                mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));
                hist = GetIncrHistogram(dest_bucket);
                mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));

                mram_write(
                    &(elem_packet[e]),
                    (__mram_ptr void *)(key_sorted_addr + hist * sizeof(tuplePair_t)),
                    sizeof(tuplePair_t));
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

    barrier_wait(&my_barrier);

    for (uint32_t bucket_id = tasklet_id; bucket_id < histogram_bucket_num + 1; bucket_id += NR_TASKLETS)
    {
        int32_t elem_num = 0;
        int32_t offset_elem = 0;

        if (bucket_id == 0)
        {
            elem_num = histogram_buff[bucket_id] - 0;
            offset_elem = 0;
        }
        else
        {
            elem_num = histogram_buff[bucket_id] - histogram_buff[bucket_id-1];
            offset_elem = histogram_buff[bucket_id-1];
        }

        if (elem_num <= 0) continue;
        else if (elem_num <= ELEM_PER_BLOCK)
        {
            // Read from MRAM
            mram_read(
                (__mram_ptr void const *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t)),
                tuples_read_buff,
                elem_num * sizeof(tuplePair_t));

            mutex_lock(&(mutex_atomic[23]));
            mergeSort(tuples_read_buff, sorted_buff1, elem_num);
            mutex_unlock(&(mutex_atomic[23]));

            // if (bucket_id == 0)
            // {
            //     for (uint32_t i = 0; i < elem_num; i++)
            //     {
            //         elem_packet = (tuplePair_t *) tuples_read_buff;
            //         printf("tuples_read_buff_after[%u]\t%u\t\n", i, elem_packet[i].lvalue);
            //     }
            // }

            // Write back to MRAM
            mram_write(tuples_read_buff,
                (__mram_ptr void *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t)),
                elem_num * sizeof(tuplePair_t));
        }
        else
        {
            // Number of elements to sort in an iteration
            uint32_t sort_num = 0;

            // An iteration for all the elements
            for (uint32_t iter = 0; iter < elem_num / ELEM_PER_BLOCK; iter++)
            {
                if (iter == elem_num / ELEM_PER_BLOCK) sort_num = elem_num - (ELEM_PER_BLOCK * (iter + 1));
                else sort_num = ELEM_PER_BLOCK;

                // Read from MRAM
                mram_read(
                    (__mram_ptr void const *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t) + iter * BLOCK_SIZE),
                    tuples_read_buff,
                    sort_num * sizeof(tuplePair_t));

                mutex_lock(&(mutex_atomic[23]));
                mergeSort(tuples_read_buff, (tuplePair_t *) sorted_buff2, sort_num);
                mutex_unlock(&(mutex_atomic[23]));

                // Write back to MRAM
                mram_write(tuples_read_buff,
                    (__mram_ptr void *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t) + iter * BLOCK_SIZE),
                    sort_num * sizeof(tuplePair_t));
            }

            for (uint32_t iter = 0; iter < (elem_num / ELEM_PER_BLOCK) / 2; iter++)
            {
                // Read from MRAM
                mram_read(
                    (__mram_ptr void const *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t) + iter * BLOCK_SIZE),
                    tuples_read_buff,
                    BLOCK_SIZE);

                if ((offset_elem + iter + 1) > elem_num) break;

                mutex_lock(&(mutex_atomic[23]));

                mram_read(
                    (__mram_ptr void const *)(key_sorted_addr + offset_elem * sizeof(tuplePair_t) + (iter + 1) * BLOCK_SIZE),
                    sorted_buff1,
                    BLOCK_SIZE);

                compareMerge(tuples_read_buff, sorted_buff1, sorted_buff2, (offset_elem + iter));

                mutex_unlock(&(mutex_atomic[23]));
            }
        }
    }

    barrier_wait(&my_barrier);

    for (uint32_t byte_offset = (TOTAL_ELEM << 3); byte_offset < PACKET_BYTE; byte_offset += BLOCK_SIZE)
    {
        uint32_t elems_to_validate = (PACKET_BYTE - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >>3);

        tuplePair_t* buff = (tuplePair_t *) tuples_read_buff;
        // Read table
        mram_read(
            (__mram_ptr void const *)(key_sorted_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Set partition to R
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            buff[e].lvalue = 0;
            buff[e].rvalue = 0;
        }

        // Write table
        mram_write(
            tuples_read_buff,
            (__mram_ptr void *)(key_sorted_addr + byte_offset),
            (elems_to_validate << 3));
    }

    barrier_wait(&my_barrier);
    return 0;
}
