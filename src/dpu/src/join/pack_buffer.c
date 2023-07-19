/*
 * Select with multiple tasklets
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <handshake.h>
#include <barrier.h>
#include <mutex.h>
#include <string.h>

#include "argument.h"
#include "hash.h"

#define MUTEX_SIZE 52
#define PARTITION_BL_SIZE 2048

#define NR_TASKLETS 12

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Variables from Host
__host glb_partition_count_arg param_glb_partition_count;
__host finish_join_arg param_finish_join;
__host dpu_results_t dpu_results;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t NR_CYCLE = 0;

// Radix Value
uint32_t RADIX;

uint64_t *zero_buffer = NULL;
int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        perfcounter_config(COUNT_CYCLES, 1);
        zero_buffer = (uint64_t *)mem_alloc(BLOCK_SIZE2);
    }
    
    barrier_wait(&my_barrier);

    if (tasklet_id < 8)
    {
        int offset = (32 * tasklet_id);
        for (int i = 0; i < 32; i++)
        {
            zero_buffer[offset + i] = 0x0;
        }
    }


    uint32_t join_result_start_addr = MRAM_BASE_ADDR + param_finish_join.join_result_start_byte;
    int32_t max_elems = (param_finish_join.max_bytes >> 3);
    int32_t effective_elems = (param_finish_join.effective_bytes >> 3);

    int32_t iterations = param_finish_join.max_bytes / BLOCK_SIZE2;
    int32_t leftover_bytes = param_finish_join.max_bytes % BLOCK_SIZE2;

    if (leftover_bytes > 0)
        iterations += 1;
    else
        leftover_bytes = BLOCK_SIZE2;

    int32_t effective_iterations = param_finish_join.effective_bytes / BLOCK_SIZE2;
    int32_t effective_leftover_bytes = param_finish_join.effective_bytes % BLOCK_SIZE2;

    if (effective_leftover_bytes > 0)
        effective_iterations += 1;
    else 
        effective_leftover_bytes = BLOCK_SIZE2;

    tuplePair_t *buffer = (tuplePair_t *)mem_alloc(BLOCK_SIZE2);

    barrier_wait(&my_barrier);

    for (int32_t it = tasklet_id; it < iterations; it += NR_TASKLETS)
    {
        int32_t process_bytes = BLOCK_SIZE2;

        if (it == (iterations - 1))
            process_bytes = leftover_bytes;

        if (it < (effective_iterations-1))
        {
            continue;
        }
        else if (it == (effective_iterations-1))
        {
            int32_t effective_elem = (effective_leftover_bytes >> 3);
            int32_t process_elem = (process_bytes >> 3);

            mram_read(join_result_start_addr + (it * BLOCK_SIZE2), buffer, process_bytes);

            for (int i = effective_elem; i < process_elem; i++)
            {
                buffer[i].lvalue = 0;
                buffer[i].rvalue = 0;
            }
            mram_write(buffer, join_result_start_addr + (it * BLOCK_SIZE2), process_bytes);
        }
        else if (it >= effective_iterations)
        {
            mram_write(zero_buffer, join_result_start_addr + (it * BLOCK_SIZE2), process_bytes);
        }
    }

    /* Partitioning Relation S */
    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        printf("INSTR: %d\n", NR_CYCLE);
        dpu_results.cycle_count = NR_CYCLE;
    }

    barrier_wait(&my_barrier);
    return 0;
}
