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
#define ELEM_PER_BLK (2048 / sizeof(tuplePair_t))

uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

__host nested_loop_join_arg param_nested_loop_join;
__host dpu_results_t dpu_results;
__host uint64_t NB_INSTR;

uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;
uintptr_t hash_table_start;

BARRIER_INIT(my_barrier, NR_TASKLETS);

tuplePair_t *R_buff;

#define R_BUFF_ELEM (NR_TASKLETS * BLOCK_SIZE / sizeof(tuplePair_t))

int miss_count = 0;
int hit_count = 0;
int valid_count = 0;
int invalid_count = 0;

int ticket = 0;
int R_bytes;
int R_blk_count;
int R_leftover_bytes;
int R_loop_count;
int R_loop_count_leftover;
int S_bytes;
int S_blk_count;
int S_leftover_bytes;
int S_loop_count;
int S_loop_count_leftover;

tuplePair_t *shared_r_buffs;

int64_t read_count = 0;

int main(void)
{
    // #ifdef S_BUFF_24KB
    int tasklet_id = me();

    if (tasklet_id == 0)
    {
        // Reset the heap
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
    }

    barrier_wait(&my_barrier);

    // relation R start addr
    tuplePair_t *wram_S_read_buff = (tuplePair_t *)mem_alloc(BLOCK_SIZE);

    // Bug Handlings Needed
    // 1. buffer가 12개 보다 작을 경우.
    // 2. buffer 크기보다 작을 경우.
    // 3. Element가 1개도 없을 경우.

    // Var Setup
    if (tasklet_id == 0)
    {
        R_buff = (tuplePair_t *)mem_alloc(NR_TASKLETS * BLOCK_SIZE);
    }
    else if (tasklet_id == 1)
    {
        R_bytes = param_nested_loop_join.packet_size * param_nested_loop_join.R_num_packets;
        R_blk_count = R_bytes / BLOCK_SIZE;
        R_leftover_bytes = R_bytes % BLOCK_SIZE;
        if (R_leftover_bytes > 0) R_blk_count++;
        R_loop_count = R_blk_count / 3;
        R_loop_count_leftover = R_blk_count % 3;
        if (R_loop_count_leftover > 0) R_loop_count++;

        printf("R_bytes: %d\n", R_bytes);
        printf("R_blk_count: %d\n", R_blk_count);
        printf("R_leftover_bytes: %d\n", R_leftover_bytes);
        printf("R_loop_count: %d\n", R_loop_count);
        printf("R_loop_count_leftover: %d\n", R_loop_count_leftover);
    }
    else if (tasklet_id == 2)
    {
        S_bytes = param_nested_loop_join.packet_size * param_nested_loop_join.S_num_packets;
        S_blk_count = S_bytes / (BLOCK_SIZE);
        S_leftover_bytes = S_bytes % (BLOCK_SIZE);
        if (S_leftover_bytes > 0) S_blk_count++;
        S_loop_count = S_blk_count / NR_TASKLETS;
        S_loop_count_leftover = S_blk_count % NR_TASKLETS;
        if (S_loop_count_leftover > 0) S_loop_count++;

        printf("S_bytes: %d\n", S_bytes);
        printf("S_blk_count: %d\n", S_blk_count);
        printf("S_leftover_bytes: %d\n", S_leftover_bytes);
        printf("S_loop_count: %d\n", S_loop_count);
        printf("S_loop_count_leftover: %d\n", S_loop_count_leftover);
    }

    barrier_wait(&my_barrier);

    uint32_t R_addr = MRAM_BASE_ADDR + param_nested_loop_join.R_packet_start_byte;
    uint32_t S_addr = MRAM_BASE_ADDR + param_nested_loop_join.S_packet_start_byte;
    uint32_t result_addr = MRAM_BASE_ADDR + param_nested_loop_join.result_start_byte;

    tuplePair_t *R_buff_threadwise = (tuplePair_t *)(((uint32_t)R_buff) + tasklet_id * BLOCK_SIZE);
    ////////////////////////////////////////////////////////////////////

    for (int outer_loop = 0; outer_loop < S_loop_count; outer_loop++)
    {
        int outer_blks_to_check = NR_TASKLETS;
        
        // if Last loop
        if (outer_loop == (S_loop_count - 1))
        {
            if (S_loop_count_leftover != 0)
            {
                outer_blks_to_check = S_loop_count_leftover;
            }
        }

        // Read 4KB blks from S
        if (tasklet_id < outer_blks_to_check)
        {
            // from to size
            mram_read(
                (__mram_ptr const void *)(S_addr + ((outer_loop * NR_TASKLETS + tasklet_id) * (BLOCK_SIZE))),
                wram_S_read_buff,
                BLOCK_SIZE);
        }

        // Check if there is a match
        // if (tasklet_id < outer_blks_to_check)
        {
            // Inner Loop
            for (int inner_loop = 0; inner_loop < R_loop_count; inner_loop++)
            {
                int inner_blks_to_check = 3;

                if (inner_loop == (R_loop_count - 1))
                {
                    if (R_loop_count_leftover != 0)
                    {
                        inner_blks_to_check = R_loop_count_leftover;
                    }
                }

                barrier_wait(&my_barrier);

                // Read 2KB blks from R
                if (tasklet_id < inner_blks_to_check)
                {
                    // printf("Read R table: %d\n", inner_loop * NR_TASKLETS + tasklet_id);
                    mram_read(
                        (__mram_ptr const void *)(R_addr + ((inner_loop * 3 + tasklet_id) * BLOCK_SIZE)),
                        R_buff_threadwise,
                        BLOCK_SIZE);
                }

                barrier_wait(&my_barrier);

                // Compare here,
                if (tasklet_id < outer_blks_to_check)
                {
                    for (int e_s = 0; e_s < 256; e_s++)
                    {

                        int loops = inner_blks_to_check * 256;

                        for (int e_r = 0; e_r < loops;)
                        {

                            mutex_lock(&(mutex_atomic[0]));
                            valid_count++;
                            mutex_unlock(&(mutex_atomic[0]));
                            if (wram_S_read_buff[e_s].lvalue == R_buff[e_r].lvalue)
                            {
                                wram_S_read_buff[e_s].lvalue = R_buff[e_r].rvalue;
                                break;
                            }
                            e_r++;

                            mutex_lock(&(mutex_atomic[0]));
                            valid_count++;
                            mutex_unlock(&(mutex_atomic[0]));
                            if (wram_S_read_buff[e_s].lvalue == R_buff[e_r].lvalue)
                            {
                                wram_S_read_buff[e_s].lvalue = R_buff[e_r].rvalue;
                                break;
                            }
                            e_r++;

                            mutex_lock(&(mutex_atomic[0]));
                            valid_count++;
                            mutex_unlock(&(mutex_atomic[0]));
                            if (wram_S_read_buff[e_s].lvalue == R_buff[e_r].lvalue)
                            {
                                wram_S_read_buff[e_s].lvalue = R_buff[e_r].rvalue;
                                break;
                            }
                            e_r++;

                            mutex_lock(&(mutex_atomic[0]));
                            valid_count++;
                            mutex_unlock(&(mutex_atomic[0]));
                            if (wram_S_read_buff[e_s].lvalue == R_buff[e_r].lvalue)
                            {
                                wram_S_read_buff[e_s].lvalue = R_buff[e_r].rvalue;
                                break;
                            }
                            e_r++;
                        }
                    }
                }
            }
        }

        // barrier_wait(&my_barrier);
        //  Write Back if neeeded,
        if (tasklet_id < outer_blks_to_check)
        {
            mram_write(
                wram_S_read_buff,
                (__mram_ptr void *)(result_addr + ((outer_loop * NR_TASKLETS + tasklet_id) * (BLOCK_SIZE))),
                BLOCK_SIZE);
        }
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        NB_INSTR = perfcounter_get();
        printf("Build NB_INSTR: %lu\n", NB_INSTR);
        // printf("misscount %d / %d / %d\n",
        printf("valid_count: %d invalid_count: %d\n", valid_count, invalid_count);
    }
    return 0;
}
