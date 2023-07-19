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

// __host hash_nphj_probe_arg param_hash_nphj_probe;
// __host dpu_results_t dpu_results;
// __host uint64_t NB_INSTR;

// uintptr_t MRAM_BASE_ADDR = DPU_MRAM_HEAP_POINTER;

// BARRIER_INIT(my_barrier, NR_TASKLETS);

// int count = 0;
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
//     tuplePair_t *result_buff = (tuplePair_t *)mem_alloc(BLOCK_SIZE);

//     uint32_t hash_table_addr = MRAM_BASE_ADDR + param_hash_nphj_probe.hash_table_start_byte;
//     uint32_t result_addr = MRAM_BASE_ADDR + param_hash_nphj_probe.result_start_byte;

//     int total_bytes = param_hash_nphj_probe.packet_size * param_hash_nphj_probe.num_packets;
//     int leftover_bytes = total_bytes % BLOCK_SIZE;
//     if (leftover_bytes == 0)
//         leftover_bytes = BLOCK_SIZE;

//     int hash_table_bucket_size = param_hash_nphj_probe.hash_table_bucket_num;

//     for (int byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
//     {
//         int elem_num = (BLOCK_SIZE >> 3);

//         if ((byte_offset + BLOCK_SIZE) > total_bytes)
//         {
//             elem_num = (leftover_bytes >> 3);
//         }

//         int local_hit_count = 0;

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
//                     mutex_lock(&(mutex_atomic[radix]));
                    
//                     mram_read(
//                         (__mram_ptr const void*)(hash_table_addr + sizeof(tuplePair_t) * bucket_index), 
//                         &hash_table_entry, 
//                         sizeof(tuplePair_t));

//                     if (hash_table_entry.lvalue == 0)
//                     {
//                         break;
//                     }
//                     else if (hash_table_entry.lvalue == wram_buff[e].lvalue)
//                     {
//                         result_buff[local_hit_count].lvalue = hash_table_entry.rvalue;
//                         result_buff[local_hit_count].rvalue = wram_buff[e].rvalue;
//                     }
//                     else
//                     {
//                         mutex_unlock(&(mutex_atomic[radix]));

//                         mutex_lock(&(mutex_atomic[50]));
//                         count++;
//                         mutex_unlock(&(mutex_atomic[50]));
                        
//                         hashed += 1;
//                         local_count++;

//                         if (hashed >= (hash_table_bucket_size))
//                         {
//                             hashed = 0;
//                         }
//                         continue;
//                     }
//                 }
//                 // write back
//                 if (local_hit_count == (BLOCK_SIZE >> 3))
//                 {
//                     mutex_lock(&(mutex_atomic[49]));
//                     int off = SHARED_COUNT;
//                     SHARED_COUNT += local_hit_count;
//                     mutex_unlock(&(mutex_atomic[49]));

//                     mram_write(result_buff,
//                         (__mram_ptr void*)(result_addr + off * sizeof(tuplePair_t)),
//                         sizeof(tuplePair_t) * local_hit_count);

//                     local_hit_count = 0;
//                 }
//             }
//         }
//         // write back
//         if (local_hit_count > 0)
//         {
//             mutex_lock(&(mutex_atomic[49]));
//             int off = SHARED_COUNT;
//             SHARED_COUNT += local_hit_count;
//             mutex_unlock(&(mutex_atomic[49]));

//             mram_write(result_buff,
//                 (__mram_ptr void*)(result_addr + off * sizeof(tuplePair_t)),
//                 sizeof(tuplePair_t) * local_hit_count);

//             local_hit_count = 0;
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
