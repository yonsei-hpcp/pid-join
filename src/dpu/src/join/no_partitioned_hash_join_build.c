// /*
//  */
// #include <stdint.h>
// #include <stdio.h>
// #include <defs.h>
// #include <mram.h>
// #include <alloc.h>
// #include <perfcounter.h>
// #include <handshake.h>
// #include <barrier.h>
// #include <mutex.h>

// #include "common.h"
// #include "argument.h"
// #include "hash.h"

// #define MUTEX_SIZE 32
// #define NR_TASKLETS 12

// #define BLOCK_SIZE 2048

// uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// __host hash_nphj_build_arg param_hash_nphj_build;
// __host dpu_results_t dpu_results;
// __host uint64_t NB_INSTR;

// uintptr_t MRAM_BASE_ADDR = DPU_MRAM_HEAP_POINTER;
// uintptr_t hash_table_start;

// BARRIER_INIT(my_barrier, NR_TASKLETS);

// int miss_count = 0;
// int SHARED_COUNT = 0;

// int main(void)
// {
//     int tasklet_id = me();

//     if (tasklet_id == 0)
//     {
//         // Reset the heap
//         mem_reset();
//         perfcounter_config(COUNT_CYCLES, true);
//     }

//     barrier_wait(&my_barrier);

//     // relation R start addr
//     tuplePair_t *wram_buff = (tuplePair_t *)mem_alloc(BLOCK_SIZE);

//     if (tasklet_id == 0)
//     {
//         memset(wram_buff, 0, BLOCK_SIZE);
//     }

//     barrier_wait(&my_barrier);

//     uint32_t hash_table_addr = MRAM_BASE_ADDR + param_hash_nphj_build.hash_table_start_byte;
//     int total_bytes = param_hash_nphj_build.packet_size * param_hash_nphj_build.num_packets;
//     int leftover_bytes = total_bytes % BLOCK_SIZE;
//     if (leftover_bytes == 0)
//         leftover_bytes = BLOCK_SIZE;

//     //////////////////////////////////////////////////////////////////////////
//     // memset the hash table    
//     //////////////////////////////////////////////////////////////////////////

//     for (int byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
//     {
//         int elem_num = (BLOCK_SIZE >> 3);

//         if ((byte_offset + BLOCK_SIZE) > total_bytes)
//         {
//             elem_num = (leftover_bytes >> 3);
//         }
//     }

//     //////////////////////////////////////////////////////////////////////////
//     // build hash table    
//     //////////////////////////////////////////////////////////////////////////


//     int hash_table_bucket_size = param_hash_nphj_build.num_packets * param_hash_nphj_build.packet_size / sizeof(tuplePair_t);

//     for (int byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
//     {
//         int elem_num = (BLOCK_SIZE >> 3);

//         if ((byte_offset + BLOCK_SIZE) > total_bytes)
//         {
//             elem_num = (leftover_bytes >> 3);
//         }

//         for (int e = 0; e < elem_num; e++)
//         {
//             if (wram_buff[e].lvalue != 0)
//             {
//                 uint32_t hashed = join_hash(wram_buff[e].lvalue);

//                 int local_count = 0;

//                 tuplePair_t hash_table_entry;
                
//                 while (1)
//                 {
//                     uint32_t radix = hashed & 0x1f;
//                     uint32_t bucket_index = hashed & hash_table_bucket_size;
                    
//                     mram_read(
//                         (__mram_ptr const void*)(hash_table_addr + sizeof(tuplePair_t) * bucket_index), 
//                         &hash_table_entry, 
//                         sizeof(tuplePair_t));

//                     // hash table hit
//                     if (hash_table_entry.lvalue == 0)
//                     {
//                         mram_write(
//                             &hash_table_entry, 
//                             (__mram_ptr void *)(hash_table_addr + sizeof(tuplePair_t) * bucket_index), 
//                             sizeof(tuplePair_t));
//                         mutex_unlock(&(mutex_atomic[radix]));
//                         break;
//                     }
//                     else
//                     {
//                         mutex_lock(&(mutex_atomic[50]));
//                         miss_count++;
//                         mutex_unlock(&(mutex_atomic[50]));
                        
//                         hashed += 1;

//                         if (hashed >= (hash_table_bucket_size))
//                         {
//                             hashed = 0;
//                         }
//                         continue;
//                     }
//                 }
//             }
//         }
//     }

//     barrier_wait(&my_barrier);

//     if (tasklet_id == 0)
//     {
//         NB_INSTR = perfcounter_get();
//         printf("Build NB_INSTR: %lu\n", NB_INSTR);
//     }
//     return 0;
// }
