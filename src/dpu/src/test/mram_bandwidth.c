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

#define NR_TASKLETS 12

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

int main(void)
{
        int tasklet_id = me();

        if (tasklet_id == 0)
        {                    
                // Initialize once the cycle counter
                mem_reset(); // Reset the heap
        }

        // Barrier
        barrier_wait(&my_barrier);
        
        // Initialize a local cache to store the MRAM block
        int64_t* buffer = (int64_t*)mem_alloc(BLOCK_SIZE);


        int mram_granularity;
        
        switch (param_microbenchmark.benchmark_type)
        {
        case 0:
        case 1:
                mram_granularity = 8;
                break;
        case 2:
        case 3: 
                mram_granularity = 16;
                break;
        case 4:
        case 5:
                mram_granularity = 32;
                break;
        case 6:
        case 7:
                mram_granularity = 64;
                break;
        case 8:
        case 9:
                mram_granularity = 128;
                break;
        case 10:
        case 11:
                mram_granularity = 256;
                break;
        case 12:
        case 13:
                mram_granularity = 512;
                break;
        case 14:
        case 15:
                mram_granularity = 1024;
                break;
        case 16:
        case 17:
                mram_granularity = 2048;
                break;
        default:
                break;
        }

        barrier_wait(&my_barrier);

        if(tasklet_id == 0)
        {
                perfcounter_config(COUNT_CYCLES, true);
        }

        // Read
        if ((param_microbenchmark.benchmark_type & 1) == 0)
        {
                for (unsigned int byte_index = (tasklet_id * mram_granularity); byte_index < (1*1024*1024); byte_index += mram_granularity * NR_TASKLETS)
                {
                        __mram_ptr void const *address_A = (__mram_ptr void const *)(MRAM_BASE_ADDR + byte_index);
                        // Load cache with current MRAM block
                        mram_read(address_A, buffer, mram_granularity);
                }
                barrier_wait(&my_barrier);
                
                if (tasklet_id == 0)
                {
                        param_microbenchmark_return.cycle_count = perfcounter_get();
                        param_microbenchmark_return.mram_read =  1 * 1024 * 1024;
                        param_microbenchmark_return.mram_granularity = mram_granularity;
                        param_microbenchmark_return.xfer_byte_size = param_microbenchmark_return.mram_read;
                }
        }
        // Write
        else
        {
                for (unsigned int byte_index = (tasklet_id * mram_granularity); byte_index < (1*1024*1024); byte_index += mram_granularity * NR_TASKLETS)
                {
                        __mram_ptr void *address_A = (__mram_ptr void *)(MRAM_BASE_ADDR + byte_index);
                        // Load cache with current MRAM block
                        mram_write(buffer, address_A, mram_granularity);
                }
                
                barrier_wait(&my_barrier);
                
                if (tasklet_id == 0)
                {
                        param_microbenchmark_return.cycle_count = perfcounter_get();
                        param_microbenchmark_return.mram_write = 1 * 1024 * 1024;
                        param_microbenchmark_return.mram_granularity = mram_granularity;
                        param_microbenchmark_return.xfer_byte_size = param_microbenchmark_return.mram_write;
                }
        }
        
        
        return 0;
}