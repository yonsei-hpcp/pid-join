///////////////////////////////////////////

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
// #include "common.h"
#include "hash.h"

#define MUTEX_SIZE 53
#define RADIX_BL_SIZE 96
#define PARTITION_BL_SIZE 1024

uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

__host hash_phj_build_arg param_phj_build_hash_table;
__host dpu_results_t dpu_results;

__host uint64_t NB_CYCLE;
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;
uint32_t hash_table_start;

#define HASH_TABLE_BUFF_SIZE 32768
#define SIZE_BLOCK 1024
#define NR_TASKLETS 12
#define HASH_TABLE_ELEM (HASH_TABLE_BUFF_SIZE >> 3)
#define ELEM_PER_BLOCK (SIZE_BLOCK >> 3)

uint32_t message[NR_TASKLETS];

BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t global_counter = 0;
uint32_t miss_count = 0;
uint32_t hit_count = 0;
// uint32_t zero_count = 0;
uint32_t total_elem_size = 0;
int elem_size;
uint32_t start_offset_byte = 0;
tuplePair_t *Hash_Table_buff;


bool DO_BREAK = false;
int ReadPartitionElemNum(uint32_t r_histogram, uint32_t my_ticket, uint32_t R_part_addr, uint32_t total_partition, uint32_t* start_offset_byte)
{
    // Var Setup
    uint32_t offset__[4] = {0,0,0,0};
    uint32_t offset;
    uint32_t offset_temp;
    mram_read((__mram_ptr void const *)(r_histogram + my_ticket * sizeof(uint32_t)), offset__, 16);

    if ((my_ticket & 0x1) == 0x0)
    {
        offset = offset__[0];
        offset_temp = offset__[1];
    }
    else
    {
        offset = offset__[1];
        offset_temp = offset__[2];
    }

    uint32_t work_addr_pair = R_part_addr + offset * sizeof(Key64_t);

    int32_t elem_size = offset_temp - offset;

    start_offset_byte[0] = (offset << 3);

    if (my_ticket == (total_partition - 1))
    {
        // buggy here. offset is not ,,,,
        elem_size = param_phj_build_hash_table.R_num - offset;
    }

    return elem_size;
}

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
    }

    barrier_wait(&my_barrier);

    if (sizeof(tuplePair_t) != 8)
    {
        dpu_results.ERROR_TYPE_3 = sizeof(tuplePair_t);
        return 0;
    }
    
    ////////////////////////////////////////////////////////////////////////
    // Variables Setup
    ////////////////////////////////////////////////////////////////////////

    // pointers
    uint32_t R_part_addr = MRAM_BASE_ADDR + param_phj_build_hash_table.parted_R_offset;
    uint32_t Hash_Table_addr = MRAM_BASE_ADDR + param_phj_build_hash_table.HT_offset;
    uint32_t r_histogram = MRAM_BASE_ADDR + param_phj_build_hash_table.parted_R_info_offset;
    uint32_t total_partition = param_phj_build_hash_table.partition_num;

    if (tasklet_id == 0)
    {
        Hash_Table_buff = (tuplePair_t *)mem_alloc(HASH_TABLE_BUFF_SIZE);
    }
    tuplePair_t *Relation_R_buff = (tuplePair_t *)mem_alloc(SIZE_BLOCK);

    ////////////////////////////////////////////////////////////////////////
    // build hash table
    ////////////////////////////////////////////////////////////////////////
    
    for (int curr_partition = 0; curr_partition < total_partition; curr_partition++)
    {
        
        ////////////////////////////////////////////////////////////////////////
        // Read all partition blocks
        ////////////////////////////////////////////////////////////////////////
        
        if(tasklet_id == 1)
        {
            elem_size = ReadPartitionElemNum(r_histogram, curr_partition, R_part_addr, total_partition, &start_offset_byte);
        }
        
        barrier_wait(&my_barrier);
        
        #ifdef VALIDATION
        if (elem_size > (HASH_TABLE_BUFF_SIZE / sizeof(tuplePair_t)))
        {
            dpu_results.ERROR_TYPE_3 = elem_size;
            break;
        }
        #endif

        int32_t elem_byte = elem_size * sizeof(tuplePair_t);

        if (tasklet_id == 0)
        {
            total_elem_size += elem_size;
        }
        
        int total_blk_num = elem_byte / SIZE_BLOCK;
        if (elem_byte > (total_blk_num * SIZE_BLOCK))
        {
            total_blk_num++;
        }

        // Hash Table Clearing
        for (int blk = tasklet_id; blk < 32; blk+=NR_TASKLETS)
        {
            uint64_t* hash_table_temp_addr = (uint64_t*)((uint32_t)Hash_Table_buff + blk * SIZE_BLOCK);
            
            for (int e = 0; e < (128);)
            {
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
                hash_table_temp_addr[e++] = 0;
            }
        }

        barrier_wait(&my_barrier);


        // Build Hash Table Here
        for (int blk = tasklet_id; blk < total_blk_num; blk+=NR_TASKLETS)
        {
            int32_t elem_for_this_blk = ((elem_byte - (blk * SIZE_BLOCK))) >> 3;
            
            if (elem_for_this_blk > (SIZE_BLOCK>>3))
            {
                elem_for_this_blk = (SIZE_BLOCK>>3);
            }
            // Firstly Read Data Block
            //from to size
            mram_read(R_part_addr + (start_offset_byte + blk * SIZE_BLOCK), Relation_R_buff, SIZE_BLOCK);

            // Hash table building
            for (int32_t e = 0; e < elem_for_this_blk; e++)
            {
                uint32_t KEY = Relation_R_buff[e].lvalue;
                uint32_t TID = Relation_R_buff[e].rvalue;

                // linear or quad
                uint32_t hashed1 = join_hash(KEY);
                uint32_t hashed = hashed1;
                // double hashing
                // uint32_t hashed1 = double_hash1(KEY);
                // uint32_t hashed = hashed1;
                // uint32_t hashed2 = double_hash2(KEY);

                hashed &= 0xFFF;
            
                #ifdef VALIDATION
                if (KEY == 0)
                {
                    printf("%sERROR!: Relation_S_buff[j] == 0 \
                    elem_for_this_blk: %d e: %d \
                    start_offset_byte: %d \
                    blk: %d \
                    param_phj_build_hash_table.R_num: %d \
                    curr_partition: %d\
                    elem_byte: %d\n", 
                        KYEL, 
                        elem_for_this_blk, 
                        e, 
                        start_offset_byte, 
                        blk, 
                        param_phj_build_hash_table.R_num, curr_partition,
                        elem_byte);

                    dpu_results.ERROR_TYPE_3 = TID;
                    return 0;
                }
                #endif
                
                // linear probe
                int counter = 0;
                int quadratic_counter = 1;
                int distance = 1;
                
                while (1)
                {
                    uint32_t hashed_mutex = hashed & 0x1f;
                    mutex_lock(mutex_atomic + hashed_mutex);

                    if (Hash_Table_buff[hashed].lvalue == 0)
                    {
                        Hash_Table_buff[hashed].lvalue = KEY;
                        mutex_unlock(mutex_atomic + hashed_mutex);
                        
                        Hash_Table_buff[hashed].rvalue = TID;
                        
                        #ifdef VALIDATION
                        mutex_lock(&(mutex_atomic[49]));
                        hit_count++;
                        mutex_unlock(&(mutex_atomic[49]));
                        #endif

                        break;
                    }
                    else
                    {
                        mutex_unlock(mutex_atomic + hashed_mutex);
                        #ifdef VALIDATION
                        if ((Hash_Table_buff[hashed].lvalue) == (KEY))
                        {
                            printf("%sError!: Key Duplicated. [%d/%d] A:%d %u B:%d %u\n", 
                                KYEL, e, elem_for_this_blk, Hash_Table_buff[hashed].lvalue, Hash_Table_buff[hashed].rvalue, KEY, TID);
                            dpu_results.ERROR_TYPE_2 = 1;
                            return 0;
                        }
                        #endif
                    
                        counter++;
                        #ifdef VALIDATION
                        mutex_lock(&(mutex_atomic[50]));
                        miss_count++;
                        mutex_unlock(&(mutex_atomic[50]));
                        #endif

                        // Linear Probing
                        hashed++;
                        if (hashed >= (HASH_TABLE_ELEM))
                            hashed = 0;

                        // Quadratic Probing
                        // hashed += (quadratic_counter * quadratic_counter);
                        // quadratic_counter++;
                        // if (hashed >= (HASH_TABLE_ELEM))
                        //     hashed = (hashed % HASH_TABLE_ELEM);

                        // Double Hashing
                        // hashed = hashed1 + (distance * double_hash2(KEY)); distance++;
                        // if (hashed >= (HASH_TABLE_ELEM))
                        //     hashed %= 4096;
                           
                            
                        #ifdef VALIDATION
                        // if (counter == (2 * HASH_TABLE_ELEM))
                        // {
                        //     printf("Error\n");
                        //     dpu_results.ERROR_TYPE_3 = 1;
                        //     break;
                        // }
                        #endif
                    }
                }
            }
        }

        barrier_wait(&my_barrier);

        for (int blk = tasklet_id; blk < 32; blk+=NR_TASKLETS)
        {
            uint32_t ht_write_addr = Hash_Table_addr + curr_partition * (HASH_TABLE_BUFF_SIZE) + blk*SIZE_BLOCK;
            mram_write((uint32_t)Hash_Table_buff + blk*SIZE_BLOCK, (__mram_ptr void *)(ht_write_addr), SIZE_BLOCK);
        }
        
        barrier_wait(&my_barrier);
    }
    
    ////////////////////////////////////////////////////////////////////////
    // Ends
    ////////////////////////////////////////////////////////////////////////

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        printf("Miss count: %u/%u\n", miss_count, param_phj_build_hash_table.R_num);
        printf("hit_count: %u/%u\n", hit_count, param_phj_build_hash_table.R_num);
        printf("total_partition: %d\n", total_partition);
        
        if (hit_count != param_phj_build_hash_table.R_num)
        {
            dpu_results.ERROR_TYPE_0 = hit_count;
            dpu_results.ERROR_TYPE_1 = param_phj_build_hash_table.R_num;
            dpu_results.ERROR_TYPE_2 = 10;
            dpu_results.ERROR_TYPE_3 = total_elem_size;
        }
        if (total_elem_size != param_phj_build_hash_table.R_num)
        {
            dpu_results.ERROR_TYPE_3 = total_elem_size;
            dpu_results.ERROR_TYPE_2 = 11;
        }
        // dpu_results.ERROR_TYPE_0 = (zero_count+1);
        
        NB_CYCLE = perfcounter_get();
        printf("Build NB_CYCLE: %lu\n", NB_CYCLE);
        dpu_results.cycle_count = NB_CYCLE;
    }
    barrier_wait(&my_barrier);
    return 0;
}