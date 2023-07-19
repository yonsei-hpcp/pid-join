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
#include "common.h"
#include "hash.h"

#define MUTEX_SIZE 53
#define RADIX_BL_SIZE 96
#define PARTITION_BL_SIZE 1024

uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

__host hash_phj_probe_arg param_phj_probe_hash_table_inner;
__host hash_phj_probe_return_arg param_hash_phj_probe_return;
__host dpu_results_t dpu_results;
__host uint64_t NR_CYCLE;
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;
char *hash_table_start;

#define HASH_TABLE_BUFF_SIZE 32768
#define NR_TASKLETS 12
#define SIZE_BLOCK2 2048
#define SIZE_BLOCK1 1024

#define ELEM_PER_BLOCK2 (SIZE_BLOCK2 >> 3)
#define ELEM_PER_BLOCK1 (SIZE_BLOCK1 >> 3)

uint32_t message[NR_TASKLETS];
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t global_counter = 0;
uint32_t miss_count = 0;
uint32_t zero_count = 0;
uint32_t hit_count = 0;
int SHARED_COUNT = 0;
int64_t TID_PAD = 0;

uint32_t S_part_addr;
uint32_t s_histogram;
uint32_t Hash_Table_addr;
uint32_t joined_result_addr;
int total_partition;


uint32_t Hash_Table_buff = 0;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Join JK TID JK TID
void Probe_2(
    uint32_t tasklet_id,
    uint32_t elem_byte, 
    tuplePair_t* S_key_buff, 
    tuplePair_t* Out_buff, 
    uint32_t S_key_addr_curr_partition,
    uint32_t Result_addr,
    tuplePair_t * Hash_Table_buff)
{
    uint32_t input_rel_offset = 0;
    int Counter = 0;
    int last_blk_elems;
    int total_blks = elem_byte / SIZE_BLOCK1;
    if (total_blks*SIZE_BLOCK1 < (elem_byte))
    {
        last_blk_elems = (elem_byte - (total_blks*SIZE_BLOCK1)) >> 3;
        total_blks++;
    }
    else
    {
        last_blk_elems = SIZE_BLOCK1 >> 3;
    }

    for (int blk = tasklet_id; blk < total_blks; blk+=NR_TASKLETS)
    {
        int elem_per_block = (SIZE_BLOCK1 / sizeof(tuplePair_t));
        if (blk == (total_blks -1))
        {
            elem_per_block = last_blk_elems;
        }
     
        mram_read(
            (__mram_ptr void const *)(S_key_addr_curr_partition + blk*SIZE_BLOCK1), 
            S_key_buff, 
            SIZE_BLOCK1);

        
        for (int j = 0; j < elem_per_block; j++)
        {
            uint32_t KEY, TID;
            KEY = S_key_buff[j].lvalue;
            TID = S_key_buff[j].rvalue;

            // linear or quad
            uint32_t hashed1 = join_hash(KEY);
            uint32_t hashed = hashed1;

            // double hashing
            // uint32_t hashed1 = double_hash1(KEY);
            // uint32_t hashed = hashed1;
            // uint32_t hashed2 = double_hash2(KEY);

            hashed &= 0x0FFF;

            // linear probe
            int runs = 0;
            int quadratic_counter = 1;
            int distance = 1;

            while (1)
            {
                #ifdef VALIDATION
                if (KEY == 0)
                {
                    printf("%sERROR!: Relation_S_JK_buff[j] == 0\n", KYEL);
                    dpu_results.ERROR_TYPE_1 = 1;
                    return;
                }
                #endif

                if (Hash_Table_buff[hashed].lvalue == 0)
                {
                    mutex_lock(&(mutex_atomic[2]));
                    printf("Zero: key: %u tid: %u\n", KEY, TID);
                    zero_count++;
                    mutex_unlock(&(mutex_atomic[2]));
                    break;
                }
                else if (Hash_Table_buff[hashed].lvalue == (int32_t)KEY)
                {
                    mutex_lock(&(mutex_atomic[1]));
                    hit_count++;
                    mutex_unlock(&(mutex_atomic[1]));

                    Out_buff[Counter].lvalue = Hash_Table_buff[hashed].rvalue;
                    Out_buff[Counter].rvalue = TID;
                    
                    Counter++;

                    break;
                }
                else
                {
                    mutex_lock(&(mutex_atomic[3]));
                    miss_count++;
                    mutex_unlock(&(mutex_atomic[3]));
                    
                    // Linear Probing
                    hashed++;
                    if (hashed >= (4096))
                        hashed = 0;
                    
                    // Quadratic Probe
                    // hashed += (quadratic_counter * quadratic_counter);
                    //     quadratic_counter++;
                    //     if (hashed >= (0x0FFf+1))
                    //         hashed = (hashed % (0x0FFf+1));

                    // Double Hashing
                    // hashed = hashed1 + (distance * double_hash2(KEY)); distance++;
                    //     if (hashed >= (4096))
                    //         hashed %= 4096;
                      
                }

                runs++;

                if (runs > (0x0FFf+1))
                {
                    // Join Not Matched
                    break;
                }
            }

            // Buffer Filled.
            if (Counter == (SIZE_BLOCK1 / sizeof(tuplePair_t)))
            {
                mutex_lock(mutex_atomic + 4);
                int origin_shared_count = SHARED_COUNT;
                SHARED_COUNT += Counter;
                mutex_unlock(mutex_atomic + 4);

                mram_write(Out_buff, 
                (__mram_ptr void *)(Result_addr + (origin_shared_count << 3)), 
                SIZE_BLOCK1);
                
                Counter = 0;  
            }
        }
    }

    // Buffer Filled.
    if (Counter > 0)
    {
        mutex_lock(mutex_atomic + 4);
        int origin_shared_count = SHARED_COUNT;
        SHARED_COUNT += Counter;
        mutex_unlock(mutex_atomic + 4);

        mram_write(Out_buff, 
        (__mram_ptr void *)(Result_addr + (origin_shared_count << 3)), 
        Counter << 3);
        
        Counter = 0;  
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(void)
{
    int tasklet_id = me();

    if (tasklet_id == 0)
    {
        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        perfcounter_config(COUNT_CYCLES, true);

        Hash_Table_buff = (uint32_t)mem_alloc(HASH_TABLE_BUFF_SIZE);
    }

    // pointers
    S_part_addr = (uint32_t)MRAM_BASE_ADDR + param_phj_probe_hash_table_inner.parted_S_offset;
    s_histogram = (uint32_t)MRAM_BASE_ADDR + param_phj_probe_hash_table_inner.parted_S_info_offset;
    Hash_Table_addr = (uint32_t)MRAM_BASE_ADDR + param_phj_probe_hash_table_inner.HT_offset;
    joined_result_addr = (uint32_t)MRAM_BASE_ADDR + param_phj_probe_hash_table_inner.Result_offset;
    total_partition = param_phj_probe_hash_table_inner.partition_num;
    
    barrier_wait(&my_barrier);



    ////////////////////////////////////////////////////////////////////////
    // Probe hash table
    ////////////////////////////////////////////////////////////////////////

    Key64_t *Relation_S_JK_buff = mem_alloc(SIZE_BLOCK1);
    Key64_t *Output_buff = mem_alloc(SIZE_BLOCK1);

    for (int current_partition = 0; current_partition < total_partition; current_partition++)
    {
        // Var Setup
        uint32_t offset__[4];
        uint32_t offset;
        uint32_t offset_temp;

        mram_read((__mram_ptr void const *)(s_histogram + current_partition * sizeof(uint32_t)), offset__, 16);

        barrier_wait(&my_barrier);
        
        if ((current_partition & 0x1) == 0x0)
        {
            offset = offset__[0];
            offset_temp = offset__[1];
        }
        else
        {
            offset = offset__[1];
            offset_temp = offset__[2];
        }
        
        uint32_t elem_size = offset_temp - offset;
        uint32_t S_key_addr_curr_partition = (uint32_t)S_part_addr + offset * sizeof(tuplePair_t);

        // printf("S_key_addr_curr_partition: %u\n", S_key_addr_curr_partition);
        
        if (current_partition == (total_partition - 1))
        {
            elem_size = param_phj_probe_hash_table_inner.S_num - offset;
        }
        uint32_t elem_byte = elem_size * sizeof(tuplePair_t);

        // Read hash table
        uint32_t curr_Hash_Table_addr = Hash_Table_addr + current_partition * HASH_TABLE_BUFF_SIZE;
        
        for (int blk = tasklet_id; blk < 16; blk += NR_TASKLETS)
        {
            mram_read(
                (__mram_ptr void const *)(curr_Hash_Table_addr + blk*SIZE_BLOCK2),
                Hash_Table_buff + blk*SIZE_BLOCK2,
                SIZE_BLOCK2);
        }
    
        barrier_wait(&my_barrier);

        if (param_phj_probe_hash_table_inner.key_table_type == JOIN_TYPE_EQUI)
            Probe_2(
                tasklet_id,
                elem_byte, 
                (tuplePair_t*)Relation_S_JK_buff, 
                (tuplePair_t*)Output_buff, 
                S_key_addr_curr_partition,
                joined_result_addr,
                Hash_Table_buff);
        else
        {
            dpu_results.ERROR_TYPE_0 = 2;
        }

        barrier_wait(&my_barrier);
    }


    ////////////////////////////////////////////////////////////////////////
    // Validation
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // Ends
    ////////////////////////////////////////////////////////////////////////
    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        dpu_results.ERROR_TYPE_2 = zero_count;
        {
            printf("miss_count: %u hit_count: (%u+%u=%u)/%u, total_partition:%d\n", miss_count, hit_count, zero_count, (hit_count+zero_count), param_phj_probe_hash_table_inner.S_num, total_partition);
        }

        NR_CYCLE = perfcounter_get();
        printf("Probe NR_CYCLE: %lu\n", NR_CYCLE);
        dpu_results.cycle_count = NR_CYCLE;

        if ((hit_count + zero_count) != param_phj_probe_hash_table_inner.S_num)
        {
            dpu_results.ERROR_TYPE_1 = 1;
        }
        param_hash_phj_probe_return.result_size = SHARED_COUNT;
        param_hash_phj_probe_return.miss_count = miss_count;
    }
    barrier_wait(&my_barrier);

    return 0;
}
