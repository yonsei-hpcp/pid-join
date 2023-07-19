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

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host sort_merge_probe_all_arg param_sort_merge_probe_all;
__host dpu_results_t dpu_results;

// THe number of instructions
__host uint32_t nb_instrs;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

uint32_t NR_INSTR = 0;
uint32_t NR_CYCLES = 0;

// Each relation start address
uint32_t R_key_sorted_addr;
uint32_t S_key_sorted_addr;

// Join result address
uint32_t result_addr;

// Buffer for data movement

// Information about relation R, S
uint32_t r_total_bytes;
uint32_t s_total_bytes;
uint32_t r_total_elem;
uint32_t s_total_elem;

// Information of index R
uint32_t r_info_elem;
uint32_t *r_info_index;

int main()
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
        R_key_sorted_addr = MRAM_BASE_ADDR + param_sort_merge_probe_all.r_sorted_start_byte;
        // printf("R_key_sorted_addr: %u\n", R_key_sorted_addr);
        S_key_sorted_addr = MRAM_BASE_ADDR + param_sort_merge_probe_all.s_sorted_start_byte;
        // printf("S_key_sorted_addr: %u\n", S_key_sorted_addr);
        result_addr = MRAM_BASE_ADDR + param_sort_merge_probe_all.result_probe_start_byte;
    }
    else if (tasklet_id == 1)
    {
        r_total_bytes = param_sort_merge_probe_all.r_total_bytes;
        s_total_bytes = param_sort_merge_probe_all.s_total_bytes;
        r_total_elem = r_total_bytes / sizeof(tuplePair_t);
        s_total_elem = s_total_bytes / sizeof(tuplePair_t);
    }

    barrier_wait(&my_barrier);

    /*
     * Phase 1.
     *    - Scan table R and get the start index
     */

    if (tasklet_id == 0)
    {
        printf("R_total_elem\t%u\t\n", r_total_elem);
        printf("S_total_elem\t%u\t\n", s_total_elem);
        r_info_elem = r_total_elem / ELEM_PER_BLOCK;
        r_info_index = (uint32_t *) mem_alloc(r_info_elem * sizeof(uint32_t));
    }

    // Relation R buffer to read index
    tuplePair_t* r_tuples_buff = (tuplePair_t *)mem_alloc(BLOCK_SIZE);

    barrier_wait(&my_barrier);

    ///////////////////////////////////
    // Read table R
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < r_total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
    {
        uint32_t elems_to_validate = ((r_total_bytes) - byte_offset) >> 3;
            if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >>3);

        // Read partition R to merge
        mram_read(
            (__mram_ptr void const *)(S_key_sorted_addr + byte_offset),
            r_tuples_buff,
            (elems_to_validate<<3)); // ?????


        // for (int e = 1; e < (elems_to_validate); e++)
        // {
        //     if (r_tuples_buff[e-1].lvalue > r_tuples_buff[e].lvalue)
        //     {
        //         if (r_tuples_buff[e].lvalue != 0)
        //         {
        //             printf("ERROR:e:%d, %d > %d/ elem_idx = %u r_total_bytes = %u\n", 
        //                 e, 
        //                 r_tuples_buff[e-1].lvalue, r_tuples_buff[e].lvalue, (byte_offset>>3), r_total_bytes);
        //         }
        //     }
        // }

        r_info_index[byte_offset / BLOCK_SIZE] = r_tuples_buff[0].lvalue;
    }

    barrier_wait(&my_barrier);



    /*
     * Phase 2.
     *    - Merge two relations
     */

    // Relation S buffer to merge
    tuplePair_t * s_tuples_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE);

    // Outer loop based on table S
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < s_total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
    {
        // Read partition S to merge
        mram_read(
            (__mram_ptr void const *)(S_key_sorted_addr + byte_offset),
            s_tuples_buff,
            BLOCK_SIZE);

        // Number of elements in each buffer
        uint32_t elem_buff = ELEM_PER_BLOCK;
        if (byte_offset + BLOCK_SIZE > s_total_bytes)
            elem_buff = (s_total_bytes - byte_offset) / sizeof(tuplePair_t);

        // if (byte_offset == 0)
        // {
        //     for (uint32_t i = 0; i < elem_buff; i++)
        //         printf("s_tuples_buff[%u]\t%u\t%u\n", i, s_tuples_buff[i].lvalue, s_tuples_buff[i].rvalue);
        // }

        uint32_t start = 0;
        if (s_tuples_buff[0].lvalue < s_tuples_buff[elem_buff - 1].lvalue)
        {
            for (uint32_t idx = 1; idx < r_info_elem; idx++)
            {
                // Where to start merge
                if (s_tuples_buff[0].lvalue < r_info_index[idx])
                {
                    start = idx - 1;
                    break;
                }
                start = r_info_elem - 1;
            }
        }
        // if (tasklet_id == 3) printf("r_info_elem[%u] %u\ts_tuples_buff[0] %u\t\n", start, r_info_index[start], s_tuples_buff[0].lvalue);

        uint32_t s_idx = 0;
        while (s_idx < elem_buff)
        {
            // Load table R block
            for (uint32_t r_byte_index = start; r_byte_index < r_info_elem; r_byte_index++)
            {
                mram_read(
                    (__mram_ptr void const *)(R_key_sorted_addr + (r_byte_index * BLOCK_SIZE)),
                    r_tuples_buff,
                    BLOCK_SIZE);

                // MUST BE AN ERROR
                if (r_tuples_buff[0].lvalue == 0)
                {
                    if (r_byte_index == start) continue;
                    else
                    {
                        s_idx = elem_buff;
                        break;
                    }
                }

                // if (r_tuples_buff[0].lvalue < s_tuples_buff[0].lvalue) continue;
                // if (tasklet_id == 1) printf("r_tuples_buff\t%u\ts_tuples_buff\t%u\n", r_tuples_buff[0].lvalue, s_tuples_buff[0].lvalue);

                uint32_t r_idx = 0;
                // Merge
                while (r_idx < ELEM_PER_BLOCK)
                {
                    if (s_tuples_buff[s_idx].lvalue == r_tuples_buff[r_idx].lvalue)
                    {
                        s_tuples_buff[s_idx].lvalue = r_tuples_buff[r_idx].rvalue;
                        
                        // if (tasklet_id == 0)
                        // {
                        //     printf("s_tuples_buff[s_idx]: %d %d | r_tuples_buff[r_idx]: %d %d\n", 
                        //         s_tuples_buff[s_idx].lvalue, s_tuples_buff[s_idx].rvalue, 
                        //         r_tuples_buff[r_idx].lvalue, r_tuples_buff[r_idx].rvalue);
                        // }

                        s_idx++;
                    }
                    else r_idx++;

                    // End of merging
                    if (s_idx >= elem_buff) break;
                }

                // End of merging
                if (s_idx >= elem_buff) break;
            }
            break;
        }

        mram_write(
            s_tuples_buff,
            (__mram_ptr void *)(result_addr + byte_offset),
            BLOCK_SIZE);
    }

    barrier_wait(&my_barrier);

    return 0;
}