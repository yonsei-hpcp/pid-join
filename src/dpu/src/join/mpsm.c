/*
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

#include "common.h"
#include "argument.h"
#include "hash.h"

#define MUTEX_SIZE 32
#define NR_TASKLETS 12

#define BLOCK_SIZE 2048

uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// __host hash_nphj_build_arg param_hash_nphj_build;
__host dpu_results_t dpu_results;
__host uint64_t NB_INSTR;

uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;
uintptr_t hash_table_start;

BARRIER_INIT(my_barrier, NR_TASKLETS);

int miss_count = 0;
int SHARED_COUNT = 0;



int main(void)
{
    int tasklet_id = me();

    if (tasklet_id == 0)
    {
        // Reset the heap
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
    }

    tuplePair_t* buffs = (tuplePair_t*)mem_alloc(BLOCK_SIZE);

    barrier_wait(&my_barrier);

    // Sort Phase
    

    barrier_wait(&my_barrier);

    // Merge Join Phase
    

    // Done

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        NB_INSTR = perfcounter_get();
        printf("Build NB_INSTR: %lu\n", NB_INSTR);
    }
    return 0;
}
