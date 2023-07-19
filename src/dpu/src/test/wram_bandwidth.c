/*
 * MRAM-WRAM R/W Latency with multiple tasklets
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <mutex.h>
#include <barrier.h>
#include <handshake.h>
#include <perfcounter.h>

#include "common.h"
#include "argument.h"


// Variables from Host
__host microbenchmark_arg param_microbenchmark;
__host microbenchmark_return_arg param_microbenchmark_return;

// Total Count from Selection
__host uint32_t SHARED_COUNT;

// The number of instructions
__host uint32_t nb_instrs;

// Mutex
MUTEX_INIT(my_mutex);

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;
// Global Addrs
uint32_t TARGET_ADDR1 = (uint32_t)DPU_MRAM_HEAP_POINTER;
uint32_t TARGET_ADDR2 = (uint32_t)DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE 2048

int32_t* buffer;

// Copy
static void random_access(int64_t *bufferA, int64_t *bufferB, int elem_num) 
{
        for (int j = 0; j < 1024; j++)
        {
                //#pragma unroll
                for (int i = 0; i < 32; i++)
                {
                        bufferB[i + 0] = bufferA[i];
                        bufferB[i + 512] = bufferA[i];
                        bufferB[i + 5] = bufferA[i];
                        bufferB[i + 135] = bufferA[i];
                        bufferB[i + 3] = bufferA[i];
                        bufferB[i + 61] = bufferA[i];
                        bufferB[i + 477] = bufferA[i];
                        bufferB[i + 177] = bufferA[i];
                        bufferB[i + 3] = bufferA[i];
                        bufferB[i + 234] = bufferA[i];
                        bufferB[i + 67] = bufferA[i];
                        bufferB[i + 635] = bufferA[i];
                        bufferB[i + 61] = bufferA[i];
                        bufferB[i + 645] = bufferA[i];
                        bufferB[i + 4] = bufferA[i];
                        bufferB[i + 317] = bufferA[i];
                }
        }
}

static void sequential_access(uint64_t *bufferA, uint64_t *bufferB, uint32_t elem_num) 
{
        for (int j = 0; j < 1024; j++)
        {
                #pragma unroll
                for (int i = 0; i < 512; i++)
                {
                        bufferB[i] = bufferA[i];
                        // bufferA[0] = bufferB[j];
                        // bufferA[512] = bufferB[j];
                        // bufferA[1] = bufferB[j];
                        // bufferA[511 - i] = bufferB[j];
                        // bufferA[2 + i] = bufferB[j];
                        // bufferA[510 - i] = bufferB[j];
                        // bufferA[3 + i] = bufferB[j];
                        // bufferA[509 - i] = bufferB[j];
                        // bufferA[4 + i] = bufferB[j];
                        // bufferA[508 - i] = bufferB[j];
                        // bufferA[5 + i] = bufferB[j];
                        // bufferA[507 - i] = bufferB[j];
                        // bufferA[6 + i] = bufferB[j];
                        // bufferA[506 - i] = bufferB[j];
                        // bufferA[7 + i] = bufferB[j];
                        // bufferA[505 - i] = bufferB[j];
                        // bufferA[8 + i] = bufferB[j];
                        // bufferA[504 - i] = bufferB[j];
                        // bufferA[9 + i] = bufferB[j];
                        // bufferA[503 - i] = bufferB[j];
                }
        }
}


int main(void)
{
        int tasklet_id = me();

        if (tasklet_id == 0)
        {                    
                // Initialize once the cycle counter
                mem_reset(); // Reset the heap
                perfcounter_config(COUNT_CYCLES, true);
        }

        // Barrier
        barrier_wait(&my_barrier);
        
        // Initialize a local cache to store the MRAM block
        
        int wram_working_set = 60*1024;
        
        // switch (param_microbenchmark.benchmark_type)
        // {
        // case 0:
        // case 1:
        //         wram_working_set = 2048;
        //         break;
        // case 2:
        // case 3: 
        //         wram_working_set = 2048 << 1;
        //         break;
        // case 4:
        // case 5:
        //         wram_working_set = 2048 << 2;
        //         break;
        // case 6:
        // case 7:
        //         wram_working_set = 2048 << 3;
        //         break;
        // case 8:
        // case 9:
        //         wram_working_set = 2048 << 4;
        //         break;
        // default:
        //         break;
        // }

        if (tasklet_id == 0)
        {
                buffer = (int32_t*)mem_alloc(wram_working_set);

                for (int i = 0; i < wram_working_set / sizeof(int32_t); i++)
                {
                        buffer[i] = i;
                }
        }
        
        barrier_wait(&my_barrier);

        // random
        if ((param_microbenchmark.benchmark_type & 1) == 0)
        {
                uint32_t cycle_start;
                if (tasklet_id==0)
                        cycle_start = perfcounter_get();
                barrier_wait(&my_barrier);
                random_access(buffer, buffer + (32*1024/sizeof(int32_t)), wram_working_set >> 4);
                barrier_wait(&my_barrier);
                uint32_t cycle_end;
                if (tasklet_id==0)
                        cycle_end = perfcounter_get();
                
                if (tasklet_id==0)
                {
                        param_microbenchmark_return.cycle_count = cycle_end - cycle_start;
                        param_microbenchmark_return.xfer_byte_size = 1024 * (32 * 16) * sizeof(int64_t)  * NR_TASKLETS;
                }
        }
        // sequential
        else
        {
                uint32_t cycle_start;
                if (tasklet_id==0)
                        cycle_start = perfcounter_get();
                barrier_wait(&my_barrier);
                sequential_access(buffer , buffer + (32*1024/sizeof(int32_t)), wram_working_set >> 4);
                barrier_wait(&my_barrier);
                uint32_t cycle_end;
                if (tasklet_id==0)
                        cycle_end = perfcounter_get();
                
                
                if (tasklet_id==0)
                {
                        param_microbenchmark_return.cycle_count = cycle_end - cycle_start;
                        param_microbenchmark_return.xfer_byte_size = 1024 * (512) * sizeof(uint64_t) * 2 * NR_TASKLETS;
                }
        }
        
        barrier_wait(&my_barrier);

        return 0;
}