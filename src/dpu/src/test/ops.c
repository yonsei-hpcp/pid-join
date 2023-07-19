/*
 *https://github.com/CMU-SAFARI/prim-benchmarks/blob/main/Microbenchmarks/Arithmetic-Throughput/dpu/task.c
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
#include "hash_test.h"

#define NR_TASKLETS 16

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

#define BLOCK_SIZE 1856
#define OPS 10000

// Arithmetic operation
#define ADD_TEST(bufferA, bufferB, scalar, T)                         \
    {                                                                 \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)    \
        {                                                             \
            for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) \
            {                                                         \
                T temp = bufferA[i];                                  \
                temp += scalar;                                       \
                ((T *)bufferB)[i] = temp;                             \
            }                                                         \
        }                                                             \
    }

#define MUL_TEST(bufferA, bufferB, scalar, T)                         \
    {                                                                 \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)    \
        {                                                             \
            for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) \
            {                                                         \
                T temp = bufferA[i];                                  \
                temp *= scalar;                                       \
                ((T *)bufferB)[i] = temp;                             \
            }                                                         \
        }                                                             \
    }

#define DIV_TEST(bufferA, bufferB, scalar, T)                         \
    {                                                                 \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)    \
        {                                                             \
            for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) \
            {                                                         \
                T temp = bufferA[i];                                  \
                temp /= scalar;                                       \
                ((T *)bufferB)[i] = temp;                             \
            }                                                         \
        }                                                             \
    }

#define SHIFT_TEST(bufferA, bufferB, scalar, T)                       \
    {                                                                 \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)    \
        {                                                             \
            for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) \
            {                                                         \
                T temp = bufferA[i];                                  \
                temp <<= scalar;                                      \
                ((T *)bufferB)[i] = temp;                             \
            }                                                         \
        }                                                             \
    }

#define WRAM_SEQUENTIAL_TEST(bufferA, bufferB, scalar, T)               \
    {                                                                   \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)      \
        {                                                               \
            for (unsigned int i = 0; i < (BLOCK_SIZE / sizeof(T)); i++) \
            {                                                           \
                T temp = ((T *)bufferA)[i];                             \
                ((T *)bufferB)[i] = temp;                               \
            }                                                           \
        }                                                               \
    }

#define WRAM_RANDOM_TEST(bufferA, bufferB, scalar, T)                 \
    {                                                                 \
        for (unsigned int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(T))); l++)    \
        {                                                             \
            for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) \
            {                                                         \
                T temp = bufferA[i];                                  \
                ((T *)bufferB)[i] = temp;                             \
            }                                                         \
        }                                                             \
    }

int64_t *shared_buffer;

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
    int64_t *buffer;
    int64_t *buffer_dest;
    
    if (param_microbenchmark.benchmark_type != 15)
    {
        buffer = (int64_t *)mem_alloc(BLOCK_SIZE);
        
        buffer_dest = (int64_t *)mem_alloc(BLOCK_SIZE);
        memset(buffer, 0xf, BLOCK_SIZE);
    }
    else
    {
        if (tasklet_id == 0)
        {
            shared_buffer = mem_alloc(58*1024);
            for (int i = 0; i < 58*1024/sizeof(int64_t); i++)
            {
                //shared_buffer[i] = rand(i) % (58*1024/sizeof(int64_t));
                shared_buffer[i] = (i) % (58*1024/sizeof(int64_t));
            }
        }
    }


    // Barrier
    barrier_wait(&my_barrier);

    if (param_microbenchmark.benchmark_type == 0)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {
            ADD_TEST(buffer, buffer_dest, 0x1111ff, int32_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 1)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {
            ADD_TEST(buffer, buffer_dest, 0xfff1111ff, int64_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 2)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {
            MUL_TEST(buffer, buffer_dest, 0xfff1111ff, int32_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 3)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            MUL_TEST(buffer, buffer_dest, 0xfff1111ff, int64_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 4)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            DIV_TEST(buffer, buffer_dest, 3, int32_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 5)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            DIV_TEST(buffer, buffer_dest, 3, int64_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 6)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            ADD_TEST(buffer, buffer_dest, 2.0, float);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 7)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            ADD_TEST(buffer, buffer_dest, 2.0, double);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 8)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            MUL_TEST(buffer, buffer_dest, 0.001, float);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 9)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            MUL_TEST(buffer, buffer_dest, 0.001, double);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 10)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            DIV_TEST(buffer, buffer_dest, 1.001, float);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 11)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            DIV_TEST(buffer, buffer_dest, 1.001, double);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 12)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            SHIFT_TEST(buffer, buffer_dest, 1, int32_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 13)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            SHIFT_TEST(buffer, buffer_dest, 1, int64_t);
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num;
        }
    }
    else if (param_microbenchmark.benchmark_type == 14)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();
        }

        barrier_wait(&my_barrier);

        // WRAM_SEQUENTIAL_TEST(buffer, buffer_dest, 1, int64_t);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            for (int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(int64_t))); l++)
            {
//#pragma unroll
                for (unsigned int i = 0; i < (BLOCK_SIZE / sizeof(int64_t)); i++)
                {
                    int64_t temp = ((int64_t *)buffer)[i];
                    ((int64_t *)buffer_dest)[i] = temp;
                }
            }
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num * 8 * 2;
        }
    }
    else if (param_microbenchmark.benchmark_type == 15)
    {
        int64_t start_clock;
        int64_t end_clock;

        if (tasklet_id == 0)
        {
            start_clock = perfcounter_get();

        }

        barrier_wait(&my_barrier);

        if (tasklet_id < param_microbenchmark.tasklet_num)
        {

            int addr = 0;
            int64_t* loc_buff = shared_buffer;
            for (int l = 0; l < (OPS / (BLOCK_SIZE / sizeof(int64_t))); l++)
            {
                // #pragma unroll
                for (unsigned int i = 0; i < (BLOCK_SIZE / sizeof(int64_t)); i++)
                {
                    int64_t temp = ((int64_t *)loc_buff)[addr];
                    ((int64_t *)loc_buff)[temp] = temp;
                    
                }
            }
        }

        barrier_wait(&my_barrier);

        if (tasklet_id == 0)
        {
            end_clock = perfcounter_get();
            param_microbenchmark_return.cycle_count = end_clock - start_clock;
            param_microbenchmark_return.ops_count = OPS * param_microbenchmark.tasklet_num * (8 * 2);
        }
    }

    return 0;
}