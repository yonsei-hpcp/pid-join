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

#define BLOCK_LOG 10
#define BLOCK_SIZE (1 << BLOCK_LOG)
#define ELEM_PER_BLOCK (BLOCK_SIZE >> 3)

// Lock
// uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Mutex
MUTEX_INIT(my_mutex);
MUTEX_INIT(done_mutex);
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
BARRIER_INIT(rem_barrier, NR_TASKLETS)

// Variables from Host
__host sort_merge_probe_arg param_sort_merge_probe;
__host sort_merge_probe_return_arg param_sort_merge_probe_return;
__host dpu_results_t dpu_results;

// The number of instructions
__host uint32_t nb_instrs;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

uint32_t NR_INSTR = 0;
uint32_t NR_CYCLES = 0;

// Page Value
uint32_t PAGE_NUM = 0;
// Total number of elements in all packets
uint32_t TOTAL_ELEM = 0;

// Relation S start addr
char *key_src_addr;
char *key_sorted_addr;

// Relation R start addr
char *probe_src_addr;

// Probe result addr
char *result_addr;

// Buffer for data movement
tuplePair_t* to_sort_buff;
tuplePair_t* queue_buff;
tuplePair_t* result_buff;

// Variables used for Probe
uint32_t start_probe = 0;
uint32_t end_probe = 0;
uint32_t byte_index_r = 0;
uint32_t num_elem_r = 0;
uint32_t tasklets_done = NR_TASKLETS;

// Prove hit count
uint32_t hit_count = 0;

// Remained tasklets
uint32_t rem_tasklet = NR_TASKLETS;

void radixSort(tuplePair_t* arr, uint32_t size, tuplePair_t* queue)
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
        for (int j = 0; j < 256; j++)
        {
            for (uint32_t k = 0; k < size; k++)
            {
                if ((arr[k].lvalue / factor) % 256 == j)
                {
                    // Put into the queue
                    queue[rear++].lvalue = arr[k].lvalue;
                } 
            }
        }
        factor <<= 8;

        for (int j = front; j != rear; j++)
        {
            if (front == rear) break;
            else
            {
                uint32_t k = queue[front++].lvalue;
                arr[j].lvalue = k;
            }
        }
        
        front = 0;
        rear = 0;
    }
}

void quickSort(tuplePair_t* arr, tuplePair_t* stack, int start, int end)
{
    // Initialize of the variables
    int pivot, top = -1;
    uint32_t temp;

    // Push initial values of l and h to stack
    stack[++top].lvalue = start;
    stack[++top].lvalue = end;

    while (top >= 0)
    {
        // Pop
        end = stack[top--].lvalue;
        start = stack[top--].lvalue;

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
            stack[++top].lvalue = start;
            stack[++top].lvalue = pivot - 1;
        }

        if (pivot + 1 < end)
        {
            stack[++top].lvalue = pivot + 1;
            stack[++top].lvalue = end;
        }
    }

    bool find = false;
    int sub = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE / sizeof(tuplePair_t); i++)
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
        key_src_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_probe.s_packet_start_byte;
        key_sorted_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_probe.s_sorted_start_byte;
        probe_src_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_probe.r_partitioned_start_byte;
        result_addr = (char *)MRAM_BASE_ADDR + param_sort_merge_probe.result_probe_start_byte;

        PAGE_NUM = param_sort_merge_probe.num_packets;
    }

    // Allocate Buffer
    data_packet_u64_128_t *read_packets_payload = NULL;
    read_packets_payload = (data_packet_u64_128_t *) mem_alloc(sizeof(data_packet_u64_128_t));

    barrier_wait(&my_barrier);


    /*
     * Phase 1.
     *    - Convert elements in packet into linear alignment
     */

    // Relation buffer to sort
    to_sort_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);

    // Total number of elements in packets
    int num_elem = 0;
    for (uint32_t my_packet_id = tasklet_id; my_packet_id < PAGE_NUM; my_packet_id += NR_TASKLETS)
    {
        // Read Page
        mram_read(
            (__mram_ptr void const *)(key_src_addr + my_packet_id * sizeof(data_packet_u64_128_t)),
            read_packets_payload,
            sizeof(data_packet_u64_128_t));
        
        tuplePair_t *elem_packet = (tuplePair_t *) read_packets_payload;
        // Tuples per packet
        for (uint32_t e = 0; e < (sizeof(data_packet_u64_128_t) / sizeof(tuplePair_t)); e++)
        {
            if (elem_packet[e].lvalue == 0) continue;
            else 
            {
                to_sort_buff[num_elem++] = elem_packet[e];
            }
        }

        // Count the total number of elements
        mutex_lock(my_mutex);
            
        TOTAL_ELEM += num_elem;
            
        if (num_elem != 0)
        {   
            // Write elements
            mram_write(to_sort_buff,
                (__mram_ptr void *)(key_sorted_addr + TOTAL_ELEM * sizeof(tuplePair_t)),
                num_elem * sizeof(tuplePair_t));

            // Update TOTAL_ELEM
            TOTAL_ELEM += num_elem;
        }

        mutex_unlock(my_mutex);
        num_elem = 0;
    }

    barrier_wait(&my_barrier);


    /*
     * Phase 2.
     *    - Sort the relation S
     *    - Probe with the relation R
     */

    // Reset WRAM
    if (tasklet_id == 3) 
    {
        mem_reset();
        // Queue Buffer for radix sort
        queue_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);
    }

    barrier_wait(&my_barrier);

    uint32_t r_total_bytes = param_sort_merge_probe.r_total_bytes;
    uint32_t r_total_elem = r_total_bytes / sizeof(tuplePair_t);

    // Range of relation R to probe
    uint32_t start_probe = 0, end_probe = 0;

    // Relation buffer to sort
    to_sort_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);
    // Buffer to store probe result
    result_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);
    
    // Number of iterations based on a tasklet
    uint32_t iter_s = 0;
    if ((TOTAL_ELEM * sizeof(tuplePair_t)) % (BLOCK_SIZE * NR_TASKLETS) == 0) 
        iter_s = (TOTAL_ELEM * sizeof(tuplePair_t)) / (BLOCK_SIZE * NR_TASKLETS);
    else iter_s = (TOTAL_ELEM * sizeof(tuplePair_t)) / (BLOCK_SIZE * NR_TASKLETS) + 1;

    // Sort and Probe
    uint32_t num_elem_s = 0;
    for (uint32_t index_s = 0; index_s < iter_s; index_s++)
    {
        uint32_t byte_index_s = (NR_TASKLETS * index_s + tasklet_id) * BLOCK_SIZE;

        // Read only the partition of S
        if (byte_index_s < TOTAL_ELEM * sizeof(tuplePair_t))
        {
            // Read the relation S
            mram_read(
                (__mram_ptr void const *)(key_sorted_addr + byte_index_s),
                to_sort_buff,
                BLOCK_SIZE);
            
            if (byte_index_s + BLOCK_SIZE > TOTAL_ELEM * sizeof(tuplePair_t))
            {
                num_elem_s = TOTAL_ELEM - (byte_index_s / sizeof(tuplePair_t));
            }
            else
                num_elem_s = BLOCK_SIZE / sizeof(tuplePair_t);

            // // Radix Sort
            // radixSort(to_sort_buff, num_elem_s, result_buff);
            // Quick Sort
            quickSort(to_sort_buff, result_buff, 0, num_elem_s - 1);
        }
        else
        {
            mutex_lock(done_mutex);
            tasklets_done--;
            mutex_unlock(done_mutex);
        }

        // Index of chunk S
        uint32_t s_idx = 0;

        // Read the full relation R
        for (uint32_t byte_index_r = 0; byte_index_r < r_total_bytes; byte_index_r += BLOCK_SIZE)
        {
            // Read the partition R
            if (tasklet_id == 0)
            {
                mram_read((__mram_ptr void const *)(probe_src_addr + byte_index_r), queue_buff, BLOCK_SIZE);
                start_probe = queue_buff[0].lvalue;
                
                // Calculate the total elements in relation R
                if (byte_index_r + BLOCK_SIZE > r_total_bytes)
                    num_elem_r = r_total_elem - byte_index_r / sizeof(tuplePair_t);
                else
                    num_elem_r = BLOCK_SIZE / sizeof(tuplePair_t);

                end_probe = queue_buff[num_elem_r - 1].lvalue; 
            }

            barrier_wait(&my_barrier);
            if (tasklets_done == 0) break;

            // Index of chunk R
            uint32_t r_idx = 0;

            if (byte_index_s < TOTAL_ELEM * sizeof(tuplePair_t))
            {
                // Circulate the chunk S
                for (uint32_t s = s_idx; s < num_elem_s; s++)
                {
                    if ((to_sort_buff[s].lvalue >= start_probe) && (to_sort_buff[s].lvalue <= end_probe))
                    {
                        // Circulate the chunk R
                        for (uint32_t r = r_idx; r < num_elem_r; r++)
                        {
                            // Found the match
                            if (to_sort_buff[s].lvalue == queue_buff[r].lvalue)
                            {
                                result_buff[s] = queue_buff[r];

                                // Update the r_idx
                                r_idx = r--;
                                // Break the for loop
                                break;
                            }
                            else continue;
                        }
                    }
                    // Nothing match in this chunk R
                    else {
                        s_idx = s;
                        break;
                    }

                    if (s == num_elem_s - 1) 
                    {
                        mutex_lock(done_mutex);
                        tasklets_done--;
                        mutex_unlock(done_mutex);
                    }
                }
            }
        }

        mram_write(
            result_buff,
            (__mram_ptr void const *)(result_addr + byte_index_s),
            BLOCK_SIZE);

        barrier_wait(&my_barrier);
        // Finish this chunk R
        if (tasklets_done == 0) break;
    }

    barrier_wait(&my_barrier);
    return 0;
}
