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

#define RADIX 0xFF
#define RADIX_DIGIT 8

// Lock
//uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Mutex
MUTEX_INIT(my_mutex);
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host sort_merge_partitioning_arg param_sort_merge_partitioning;
__host sort_merge_partitioning_return_arg param_sort_merge_partitioning_return;
__host dpu_results_t dpu_results;

// The number of instructions
__host uint32_t nb_instrs;
// The number of cycles
__host uint32_t nb_cycles;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

uint32_t NR_INSTRUCTIONS = 0; 
uint32_t NR_CYCLES = 0;

// Page Value
uint32_t PAGE_NUM = 0;
// Total number of elements in all packets
uint32_t TOTAL_ELEM = 0;

// Relation R start addr
char *key_src_addr;
char *key_sorted_addr;
// Histogram start addr
char *histogram_addrs;

// Number of indexes to sort
uint32_t compressed_elem = 0;

// Buffer for Histogram
uint32_t *histogram_buff;
uint32_t *histogram_buff_2;
uint32_t *cum_histogram_buff;

uint32_t *histogram_addrs_buff;
// Number for Radix Partitioning
uint32_t max_digit = 0;
uint32_t min_digit = 16;
// Number of buckets for Histogram
uint32_t histogram_bucket_num = 0;
uint32_t histogram_bucket_elem = 0;

void radixSort(tuplePair_t* arr, uint32_t size, uint32_t* queue)
{
    int digit = 0, factor = 1;
    int front = 0, rear = 0;

    uint32_t max = arr[0].lvalue;
    // Calculate the max value in the array
    for (uint32_t i = 1; i < size; i++)
        if (max < arr[i].lvalue) max = arr[i].lvalue;

    // Calculate the digit of the max value
    for (uint32_t i = max; i > 0; i >>= 8) digit++;

    // Radix Sort
    for (int i = 0; i < digit; i++)
    {
        for (int j = 0; j < (RADIX + 1); j++)
        {
            for (uint32_t k = 0; k < size; k++)
            {
                if ((arr[k].lvalue / factor) % (RADIX + 1) == j)
                {
                    // Put into the queue
                    queue[rear++] = arr[k].lvalue;
                } 
            }
        }
        factor <<= 8;

        for (int j = front; j != rear; j++)
        {
            if (front == rear) break;
            else
            {
                uint32_t k = queue[front++];
                arr[j].lvalue = k;
            }
        }
        
        front = 0;
        rear = 0;
    }
}

void quickSort(tuplePair_t* arr, uint32_t* stack, int start, int end)
{
    uint32_t size = end -start + 1;

    // Initialize of the variables
    int pivot, top = -1;
    uint32_t temp = 0;

    // Push initial values of l and h to stack
    stack[++top] = start;
    stack[++top] = end;

    while (top >= 0)
    {
        // Pop
        end = stack[top--];
        start = stack[top--];

        uint32_t x = arr[end].lvalue;
        int i = start - 1;

        for (int j = start; j <= end - 1; j++)
        {
            if (arr[j].lvalue <= x)
            {
                i++;
                temp = arr[i].lvalue;
                arr[i].lvalue = arr[j].lvalue;
                arr[j].lvalue = temp;
            }
        }
        
        temp = arr[i + 1].lvalue;
        arr[i + 1].lvalue = arr[end].lvalue;
        arr[end].lvalue = temp;
        // Set pivot
        pivot = i + 1;

        if (pivot - 1 > start)
        {
            stack[++top] = start;
            stack[++top] = pivot - 1;
        }

        if (pivot + 1 < end)
        {
            stack[++top] = pivot + 1;
            stack[++top] = end;
        }
    }

    bool find = false;
    int sub = 0;
    for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(tuplePair_t); i++)
    {
        if (arr[i].lvalue == 0) continue;
        else
        {
            if (find) arr[i - sub] = arr[i];
            else 
            {
                find = true;
                sub = i;
                arr[0] = arr[i];
            }
        }
    }
}

uint32_t buildSubHistogram(uint32_t* histogram_src, uint32_t* compressed_hist, uint32_t* cumulative_hist, uint32_t bucket_num)
{
    /*
     * Build cumulative histogram
     * Build compressed histogram
     */

    // Compressed histogram to fit nearly WRAM size
    uint32_t compressed_bytes = 0;
    uint32_t compressed_elem = 0;

    // Initialize cumulative histogram
    cumulative_hist[0] = 0;
    compressed_hist[compressed_elem++] = 0;

    for (uint32_t idx = 0; idx < histogram_bucket_num; idx++)
    {
        // Build cumulative histogram
        cumulative_hist[idx + 1] = cumulative_hist[idx] + histogram_src[idx];

        // Compressed histogram to fit in WRAM
        if (histogram_src[idx] != 0)
        {
            if (compressed_bytes + (histogram_src[idx] * sizeof(tuplePair_t)) <= BLOCK_SIZE)
            {
                compressed_bytes += histogram_src[idx] * sizeof(tuplePair_t);
                compressed_hist[compressed_elem] = idx + 1;
            }
            else
            {
                compressed_bytes = histogram_src[idx] * sizeof(tuplePair_t);
                compressed_hist[compressed_elem++] = idx;
            }
        }
    }

    return compressed_elem;
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
        histogram_addrs = (char *)MRAM_BASE_ADDR + param_sort_merge_partitioning.histogram_addr_start_byte;
        PAGE_NUM = param_sort_merge_partitioning.num_packets;
    }

    barrier_wait(&my_barrier);

    // The number of packets thread will handle
    uint32_t packets_per_th = (BLOCK_SIZE / 2) / sizeof(data_packet_u64_128_t); 

    // Allocate Buffer for packets
    data_packet_u64_128_t *read_packets_payload = NULL;
    read_packets_payload = (data_packet_u64_128_t *) mem_alloc(sizeof(data_packet_u64_128_t) * packets_per_th);


    /*
     * Phase 1.
     *    - Check the number of total elements in packets
     *    - Find the max/min value of all elements and its digit
     *    - Ready for Radix Range Partitioning
     */

    // Max value from each tasklet
    uint32_t max_val = 0;
    // Min value from each tasklet
    uint32_t min_val = 1 << 20;

    for (uint32_t my_packet_id = tasklet_id; my_packet_id < (PAGE_NUM / packets_per_th); my_packet_id += NR_TASKLETS)
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + my_packet_id * packets_per_th * sizeof(data_packet_u64_128_t)),
            read_packets_payload,
            sizeof(data_packet_u64_128_t) * packets_per_th);

        // Total number of elements in packets
        int num_elem = 0;

        data_packet_u64_128_t *packet = (data_packet_u64_128_t *) read_packets_payload;
        // Packets per each tasklet
        for (uint32_t p = 0; p < packets_per_th; p++)
        {
            tuplePair_t *elem_packet = (tuplePair_t *)(&packet[p]);
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
        // Count the total number of elements
        mutex_lock(my_mutex);
        TOTAL_ELEM += num_elem;
        mutex_unlock(my_mutex);
    }

    // Calculate the number of digit for the max value
    uint32_t max_digit_th = 0;
    while (max_val > RADIX)
    {
        max_digit_th++;
        max_val >>= RADIX_DIGIT;
    }
    // Calculate the number of digit for the min value
    uint32_t min_digit_th = 0;
    while (min_val > RADIX)
    {
        min_digit_th++;
        min_val >>= RADIX_DIGIT;
    }

    // Max/Min digit for all tasklets
    mutex_lock(my_mutex);
    if (max_digit_th > max_digit) max_digit = max_digit_th;
    if (min_digit_th < min_digit) min_digit = min_digit_th;
    mutex_unlock(my_mutex);

    barrier_wait(&my_barrier);

    //if (tasklet_id == 9) printf("max_val\t%u\tmin_val\t%u\t\n", max_val, min_val);

    /*
     * Phase 2.
     *    - Build a histogram to partition the relation R
     */

    // Total Bytes to handle
    uint32_t total_bytes = TOTAL_ELEM * sizeof(tuplePair_t);

    uint32_t digit_diff = max_digit - min_digit + 1;
    // Number of total bars of the histogram
    uint32_t bucket_num = digit_diff << RADIX_DIGIT;

    mutex_lock(my_mutex);
    if (histogram_bucket_num < bucket_num) histogram_bucket_num = bucket_num;
    mutex_unlock(my_mutex);

    barrier_wait(&my_barrier);

    if (tasklet_id == 1)
    {
        // Buffer used for histogram build
        histogram_buff = (uint32_t *) mem_alloc(histogram_bucket_num * sizeof(uint32_t));

        // Initialize the histogram
        for (uint32_t idx = 0; idx < histogram_bucket_num; idx++) histogram_buff[idx] = 0;
    }

    barrier_wait(&my_barrier);

    for (uint32_t my_packet_id = tasklet_id; my_packet_id < (PAGE_NUM / packets_per_th); my_packet_id += NR_TASKLETS)
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + my_packet_id * packets_per_th * sizeof(data_packet_u64_128_t)),
            read_packets_payload,
            sizeof(data_packet_u64_128_t) * packets_per_th);

        data_packet_u64_128_t *packet = (data_packet_u64_128_t *) read_packets_payload;
        // Packets per each tasklet
        for (uint16_t p = 0; p < packets_per_th; p++)
        {
            tuplePair_t *elem_packet = (tuplePair_t *)(&packet[p]);
            // Tuples per packet
            uint32_t elem_digit, elem_val;
            for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
            {
                elem_digit = 0;
                elem_val = elem_packet[e].lvalue;

                if (elem_val == 0) continue;
                else
                {
                    // Digit for each element
                    while (elem_val > 0xFF)
                    {
                        elem_digit++;
                        elem_val >>= 8;
                    }

                    // Histogram for radix partitioning
                    uint32_t histogram_idx = (elem_digit - min_digit) << RADIX_DIGIT;
                    uint32_t modify_radix = elem_digit * RADIX_DIGIT;
                    histogram_idx += ((elem_packet[e].lvalue & (RADIX << modify_radix)) >> modify_radix);
                    
                    // Build a histogram
                    mutex_lock(my_mutex);
                    histogram_buff[histogram_idx]++;
                    mutex_unlock(my_mutex);
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    /*
     * Phase 3.
     *    - Build a cumulative histogram
     *    - Find the maximum elements in the histogram
     *    - Calculate the number of elements to sort at once
     */
    
    if (tasklet_id == 5)
    {
        // Buffer used for store elements in MRAM
        histogram_buff_2 = (uint32_t *) mem_alloc(histogram_bucket_num * sizeof(uint32_t));
        cum_histogram_buff = (uint32_t *) mem_alloc((histogram_bucket_num + 1) * sizeof(uint32_t));
        
        // Compressed and Cumulative Histogram
        compressed_elem = buildSubHistogram(histogram_buff, histogram_buff_2, cum_histogram_buff, histogram_bucket_num);
    }

    barrier_wait(&my_barrier);

    /*
     * Phase 4.
     *    - Store elements to MRAM in random manner
     */

    // Initialize the histogram buffer
    if (tasklet_id == 6)
    {
        for (uint32_t idx = 0; idx < histogram_bucket_num; idx++) histogram_buff[idx] = 0;
    }

    barrier_wait(&my_barrier);

    bool w_not_fit = false;
    for (uint32_t my_packet_id = tasklet_id; my_packet_id < (PAGE_NUM / packets_per_th); my_packet_id += NR_TASKLETS)
    {
        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + my_packet_id * packets_per_th * sizeof(data_packet_u64_128_t)),
            read_packets_payload,
            sizeof(data_packet_u64_128_t) * packets_per_th);

        data_packet_u64_128_t *packet = (data_packet_u64_128_t *) read_packets_payload;
        // Packets per each tasklet
        for (uint16_t p = 0; p < packets_per_th; p++)
        {
            tuplePair_t *elem_packet = (tuplePair_t *)(&packet[p]);
            // Tuples per packet
            for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
            {
                uint32_t elem_digit = 0;
                uint32_t elem_val = elem_packet[e].lvalue;

                if (elem_val == 0) continue;
                else
                {
                    // Digit for each element
                    while (elem_val > RADIX)
                    {
                        elem_digit++;
                        elem_val >>= RADIX_DIGIT;
                    }

                    // Histogram for radix partitioning
                    uint32_t histogram_idx = (elem_digit - min_digit) << RADIX_DIGIT;
                    uint32_t modify_radix = elem_digit * RADIX_DIGIT;
                    histogram_idx += ((elem_packet[e].lvalue & (RADIX << modify_radix)) >> modify_radix);

                    mutex_lock(my_mutex);

                    // Store the element
                    uint32_t hist = cum_histogram_buff[histogram_idx] + histogram_buff[histogram_idx];
                    histogram_buff[histogram_idx]++;
                    mram_write(
                        &(elem_packet[e]),
                        (__mram_ptr void *)(key_sorted_addr + hist * sizeof(tuplePair_t)),
                        sizeof(tuplePair_t));

                    mutex_unlock(my_mutex);
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    /*
     * Phase 5.
     *    - Sort elements in MRAM
     *    - Build a list of histogram_bar_addr
     */

    // Buffer for elements to sort for each tasklet
    tuplePair_t* to_sort_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);

    // Initialize used buffer to use as a queue
    for (uint32_t idx = 0; idx < histogram_bucket_num; idx++)
    {
        histogram_buff[idx] = 0;
    }
    for (uint32_t my_packet_id = tasklet_id; my_packet_id < compressed_elem; my_packet_id += NR_TASKLETS)
    {
        mutex_lock(my_mutex);

        // Read from MRAM
        mram_read(
            (__mram_ptr void const *)(key_sorted_addr + cum_histogram_buff[histogram_buff_2[my_packet_id]] * sizeof(tuplePair_t)),
            to_sort_buff,
            BLOCK_SIZE);

        mutex_unlock(my_mutex);

        quickSort(to_sort_buff, histogram_buff, 0, BLOCK_SIZE / sizeof(tuplePair_t) - 1);

        mutex_lock(my_mutex); 

        // Write back to MRAM
        mram_write(
            to_sort_buff,
            (__mram_ptr void *)(key_sorted_addr + cum_histogram_buff[histogram_buff_2[my_packet_id]] * sizeof(tuplePair_t)),
            BLOCK_SIZE);

        mutex_unlock(my_mutex);
    }
    
    barrier_wait(&my_barrier);
    return 0;
}