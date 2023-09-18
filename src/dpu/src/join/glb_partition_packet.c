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
#define RADIX_BL_SIZE 96

#define PARTITION_BL_LOG 10
#define PARTITION_BL_SIZE (1 << PARTITION_BL_LOG)
#define PARTITION_BL_ELEM (PARTITION_BL_SIZE >> 3)

#define NUM_DPU_LOG 6
#define NUM_DPU_RANK (1 << NUM_DPU_LOG)

#define CACHE_SIZE (1 << 13)

#define NR_TASKLETS 12

#define BLOCK_SIZE1 1800
#define BLOCK_SIZE1_ELEM (BLOCK_SIZE1 >> 3)

#define BLOCK_SIZE2 2048
#define BLOCK_SIZE2_ELEM (BLOCK_SIZE2 >> 3)
// Variables from Host
__host glb_partition_packet_arg param_glb_partition_packet;
__host dpu_results_t dpu_results;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];
// Mutex
MUTEX_INIT(my_mutex);

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t NR_CYCLE = 0;

int32_t TOTAL_PAGE = 0;
int32_t LAST_PAGE_ELEM = 0;

// TIDs Addr
uint32_t Tids_addr;
uint32_t Pair_addr;
uint32_t Tids_dest_addr;

// Payload Addr
uint32_t Payload_addr;
uint32_t Payload_dest_addr;

// Partition Infos
uint32_t packet_histogram_addr;
uint32_t local_histogram_addr;

// Initialize a local buffer to store the MRAM block
int32_t *histogram_buff = NULL;
uint32_t *addr_buff = NULL;

int64_t *hist_accmulate = NULL;

int32_t num_ranks = 0;

int32_t PACKET_SIZE = 0;

void SetHistogram(int32_t idx, uint32_t value)
{
    int32_t *wb = histogram_buff + idx;
    (*wb) = value;
}

void IncrHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    *wb += 1;
}

int32_t GetIncrHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    *wb += 1;
    return (*wb - 1);
}

#define GetIncrHistogram__(ret, radix)          \
    do                                          \
    {                                           \
        int32_t *wb = (histogram_buff + radix); \
        ret = *wb;                              \
        *wb += 1;                               \
    } while (0)

int32_t GetHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    return *wb;
}

void AddHistogram(int32_t idx, uint32_t val)
{
    int32_t *wb = histogram_buff + idx;
    *wb += val;
}

void SubHistogram(int32_t idx, uint32_t val)
{
    int32_t *wb = histogram_buff + idx;
    *wb -= val;
}

inline uint32_t tid_partition(uint32_t tid)
{
    int32_t dpu_id;
    GET_RANK_DPU_FROM_TUPLE_ID(dpu_id, tid);
    return dpu_id;
}

#define SRC_NTH_OF_BANK_GROUP 8
#define DST_NTH_OF_BANK_GROUP 8

int8_t LUT[SRC_NTH_OF_BANK_GROUP][DST_NTH_OF_BANK_GROUP] =
    {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {7, 0, 1, 2, 3, 4, 5, 6},
        {6, 7, 0, 1, 2, 3, 4, 5},
        {5, 6, 7, 0, 1, 2, 3, 4},
        {4, 5, 6, 7, 0, 1, 2, 3},
        {3, 4, 5, 6, 7, 0, 1, 2},
        {2, 3, 4, 5, 6, 7, 0, 1},
        {1, 2, 3, 4, 5, 6, 7, 0},
};

// Rank 0
//  packet 0 | packet 8 | packet 16 | packet 24 | packet 32 | packet 40 | packet 48 | packet 56 | //
//  packet 1 | packet 9 | packet 17 | packet 25 | packet 33 | packet 41 | packet 49 | packet 57 | //
// .............................................................................  //
//  packet 7 | packet 15 | packet 23 | packet 31 | packet 39 | packet 47 | packet 55 | packet 63 |//
// Rank 0
//  packet 0 | packet 8 | packet 16 | s 24 | packet 32 | packet 40 | packet 48 | packet 56 | //
//  packet 1 | packet 9 | packet 17 | packet 25 | packet 33 | packet 41 | packet 49 | packet 57 | //
// .............................................................................  //
//  packet 7 | packet 15 | packet 23 | packet 31 | packet 39 | packet 47 | packet 55 | packet 63 |//

// Rank 1
//  packet 0 | packet 8 | packet 16 | packet 24 | packet 32 | packet 40 | packet 48 | packet 56 | //
//  packet 1 | packet 9 | packet 17 | packet 25 | packet 33 | packet 41 | packet 49 | packet 57 | //
// .............................................................................  //
//  packet 7 | packet 15 | packet 23 | packet 31 | packet 39 | packet 47 | packet 55 | packet 63 |//
// Rank 2
//  packet 0 | packet 8 | packet 16 | packet 24 | packet 32 | packet 40 | packet 48 | packet 56 | //
//  packet 1 | packet 9 | packet 17 | packet 25 | packet 33 | packet 41 | packet 49 | packet 57 | //
// .............................................................................  //
//  packet 7 | packet 15 | packet 23 | packet 31 | packet 39 | packet 47 | packet 55 | packet 63 |//

/***
 * my_dpu_id = 0/
 * packet 0 packet 8 packet 16 ... packet 56
 * packet 1 packet 9 packet 17 ... packet 57
 *
 * * my_dpu_id = 8/
 * packet 8 packet 16 ... packet 56 packet 0
 * packet 9 packet 17 ... packet 57 packet 1
 * */

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (sizeof(tuplePair_t) != 8)
    {
        dpu_results.ERROR_TYPE_0 = 99;
        return 0;
    }

    if (tasklet_id == 0)
    {
        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        PACKET_SIZE = param_glb_partition_packet.packet_size;
        perfcounter_config(COUNT_CYCLES, 1);

        // printf("param_glb_partition_packet.partition_num:%d\n", param_glb_partition_packet.partition_num);
        num_ranks = (param_glb_partition_packet.partition_num / NUM_DPU_RANK);

        // Setup Histogram (packet num) & Partiton info(of global partitioning phase)
        int64_t *temp_histogram_buff = (int64_t *)mem_alloc(2048 * sizeof(int32_t));
        addr_buff = (uint32_t *)mem_alloc(2048 * sizeof(uint32_t));
        hist_accmulate = (int64_t *)mem_alloc((num_ranks + 1) * sizeof(int64_t)); // Last elem for the total amount of data

        // make accumed histogram
        packet_histogram_addr = MRAM_BASE_ADDR + param_glb_partition_packet.packet_histogram_start_byte;
        local_histogram_addr = MRAM_BASE_ADDR + param_glb_partition_packet.histogram_start_byte;
        mram_read((__mram_ptr void const *)(packet_histogram_addr), temp_histogram_buff, num_ranks * sizeof(int64_t));

        hist_accmulate[0] = 0;
        for (int p = 0; p < num_ranks; p++)
        {
            hist_accmulate[p + 1] = temp_histogram_buff[p] + hist_accmulate[p];
            // printf("hist_accmulate[%d]: %ld\n", p+1, hist_accmulate[p + 1]);
        }
        histogram_buff = (int32_t *)temp_histogram_buff;

        ////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < 2048; i++)
        {
            histogram_buff[i] = 0;
        }

        int num_blocks = ((hist_accmulate[num_ranks] * PACKET_SIZE * NUM_DPU_RANK) / BLOCK_SIZE2);
        int last_block_size = ((hist_accmulate[num_ranks] * PACKET_SIZE * NUM_DPU_RANK) % BLOCK_SIZE2);

        for (int i = 0; i < (num_blocks); i++)
        {
            mram_write(
                histogram_buff,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_glb_partition_packet.result_offset + BLOCK_SIZE2 * i),
                2048);
        }

        if (last_block_size > 0)
            mram_write(
                histogram_buff,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_glb_partition_packet.result_offset + BLOCK_SIZE2 * (num_blocks)),
                last_block_size);
        ///////////////////////////////////////////////////////////////////////
    }

    if (tasklet_id == 1)
    {
        // Address pointers of the current processing block in MRAM
        Tids_addr = MRAM_BASE_ADDR + param_glb_partition_packet.input_offset1;
        Payload_addr = MRAM_BASE_ADDR + param_glb_partition_packet.input_offset2;
        Pair_addr = MRAM_BASE_ADDR + param_glb_partition_packet.input_offset1;
    }

    // Arguments setting
    int32_t my_dpu_id = param_glb_partition_packet.dpu_id;
    int32_t partition_num = param_glb_partition_packet.partition_num;
    int32_t n_th_of_bg = my_dpu_id / 8;

    barrier_wait(&my_barrier);

    // memset
    if (tasklet_id == 0)
    {
        memset(histogram_buff, 0, partition_num * sizeof(int32_t));
    }

    // Addresss Calculation

    if (tasklet_id < 8)
    {
        for (int r = 0; r < num_ranks; r++)
        {
            uint32_t base_addr = MRAM_BASE_ADDR + param_glb_partition_packet.result_offset + ((int)hist_accmulate[r] * NUM_DPU_RANK * PACKET_SIZE);

            for (int dst_dpu = tasklet_id; dst_dpu < NUM_DPU_RANK; dst_dpu += 8)
            {
                int nth_bg_dst_dpu = (dst_dpu & 7);
                int nth_of_bg_dst_dpu = (dst_dpu / 8);

                int idx = r * NUM_DPU_RANK + dst_dpu;
                addr_buff[idx] = base_addr;
                addr_buff[idx] = addr_buff[idx] + (LUT[n_th_of_bg][nth_of_bg_dst_dpu] + (nth_bg_dst_dpu << 3)) * PACKET_SIZE;
            }
        }
    }

    barrier_wait(&my_barrier);

    if (param_glb_partition_packet.partition_type == GLB_PART_ARR_L0)
    {
        tuplePair_t *key_packet = (tuplePair_t *)mem_alloc(BLOCK_SIZE2);

        RadixPartitionArrayPacked(
            tasklet_id,
            Tids_addr,
            param_glb_partition_packet.elem_num,
            key_packet,
            param_glb_partition_packet.partition_num,
            param_glb_partition_packet.packet_size);
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        int total_ = 0;
        NR_CYCLE = perfcounter_get();
        printf("?PACKET PARTITIONING INSTR: %d\n", NR_CYCLE);

        // #ifdef VALIDATION
        int32_t *temp_buff = (int32_t *)addr_buff;

        int read_blocks = (param_glb_partition_packet.partition_num * sizeof(int32_t)) / BLOCK_SIZE2;
        int leftovers = (param_glb_partition_packet.partition_num * sizeof(int32_t)) % BLOCK_SIZE2;

        if (leftovers == 0)
        {
            leftovers = BLOCK_SIZE2;
        }
        else
        {
            read_blocks += 1;
        }

        for (int i = 0; i < (read_blocks - 1); i++)
        {
            mram_read(
                (__mram_ptr void const *)(local_histogram_addr + i * BLOCK_SIZE2),
                temp_buff,
                BLOCK_SIZE2);

            for (uint32_t j = 0; j < (BLOCK_SIZE2 / sizeof(int32_t)); j++)
            {
                if (temp_buff[j] != histogram_buff[j + i * (BLOCK_SIZE2 / sizeof(int32_t))])
                {
                    dpu_results.ERROR_TYPE_3 = 5;
                }
            }
        }

        mram_read(
            (__mram_ptr void const *)(local_histogram_addr + (read_blocks - 1) * BLOCK_SIZE2),
            temp_buff, leftovers);

        for (uint32_t j = 0; j < leftovers / sizeof(int32_t); j++)
        {
            if (temp_buff[j] != histogram_buff[j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t))])
            {
                printf("[%2d/%2d]: %3d vs %3d |",
                       j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t)),
                       (leftovers / sizeof(int32_t)) + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t)),
                       temp_buff[j],
                       histogram_buff[j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t))]);
                dpu_results.ERROR_TYPE_0 = 6;
            }
        }
        printf("\n");

        // printf("histogram..\n");
        for (int i = 0; i < partition_num; i++)
        {
            total_ += histogram_buff[i];
        }

        printf("Total. %d/%d\n", total_, param_glb_partition_packet.elem_num);
        if (total_ > param_glb_partition_packet.elem_num)
        {
            dpu_results.ERROR_TYPE_0 = total_;
            dpu_results.ERROR_TYPE_1 = param_glb_partition_packet.elem_num;
            dpu_results.ERROR_TYPE_2 = param_glb_partition_packet.partition_type;
        }

        // if (param_glb_partition_packet.partition_type == 1)
        // {
        //     if (param_glb_partition_packet.elem_num - total_ >= 2)
        //     {
        //         dpu_results.ERROR_TYPE_1 = (param_glb_partition_packet.elem_num - total_);
        //     }
        // }
        // #endif
    }

    barrier_wait(&my_barrier);

    return 0;
}

int RadixPartitionArrayPacked(
    uint32_t tasklet_id,
    uint32_t mram_source_addr,
    uint32_t num_elem,
    tuplePair_t *wram_buff,
    int partition_num,
    int packet_size)
{
    int RADIX = (partition_num - 1);
    int tot_bytes = num_elem * sizeof(tuplePair_t);
    int last_num = num_elem % BLOCK_SIZE2_ELEM;

    // packet granularity
    int elem_per_packet = packet_size >> 3; // could be 1 2 4 8 16
    int elem_per_packet_1 = elem_per_packet - 1;
    int elem_per_packet_shift = 0;
    int one_rnc_packet_per_rank_shift = 0;

    if (elem_per_packet == 1)
    {
        elem_per_packet_shift = 0;
        one_rnc_packet_per_rank_shift = 9;
    }
    else if (elem_per_packet == 2)
    {
        elem_per_packet_shift = 1;
        one_rnc_packet_per_rank_shift = 10;
    }
    else if (elem_per_packet == 4)
    {
        elem_per_packet_shift = 2;
        one_rnc_packet_per_rank_shift = 11;
    }
    else if (elem_per_packet == 8)
    {
        elem_per_packet_shift = 3;
        one_rnc_packet_per_rank_shift = 12;
    }
    else if (elem_per_packet == 16)
    {
        elem_per_packet_shift = 4;
        one_rnc_packet_per_rank_shift = 13;
    }

    if (last_num == 0)
        last_num = BLOCK_SIZE2_ELEM;

    for (int byte_addr = tasklet_id * BLOCK_SIZE2; byte_addr < tot_bytes; byte_addr += (NR_TASKLETS * BLOCK_SIZE2))
    {
        int elem_n = BLOCK_SIZE2_ELEM;
        if ((byte_addr + BLOCK_SIZE2) >= tot_bytes)
        {
            elem_n = last_num;
        }

        // Read Data
        mram_read(
            (__mram_ptr void const *)(mram_source_addr + byte_addr),
            wram_buff,
            BLOCK_SIZE2);

        for (int e = 0; e < elem_n; e++)
        {
            if (((int32_t)(wram_buff[e].lvalue)) < 0)
            {
                dpu_results.ERROR_TYPE_3 = 1;
            }

            if (wram_buff[e].lvalue != 0)
            {
                // Hash
                tuplePair_t pair;
                pair.lvalue = (uint32_t)(wram_buff[e].lvalue);
                pair.rvalue = (uint32_t)(wram_buff[e].rvalue);
                uint32_t hash_val = (RADIX & glb_partition_hash(wram_buff[e].lvalue));

                // Get Lock
                mutex_lock(&(mutex_atomic[(hash_val & 31)]));
                int32_t offset_num = GetIncrHistogram(hash_val);
                mutex_unlock(&(mutex_atomic[(hash_val & 31)]));

                // calculate data destination
                uint32_t target_addr = addr_buff[hash_val];

                int num_packets = (offset_num >> elem_per_packet_shift);
                int loc_offset = (offset_num & (elem_per_packet_1));
                target_addr += (num_packets << one_rnc_packet_per_rank_shift) + (loc_offset << 3);

#ifdef VALIDATION
                if (target_addr > (64 * 1024 * 1024))
                {
                    printf("num_packets: %d target_addr: %d offset_num: %d hash_val: %d\n", num_packets, target_addr, offset_num, hash_val);
                    dpu_results.ERROR_TYPE_1 = 5;
                }
                else
#endif
                {
                    mram_write(
                        (&pair),
                        (__mram_ptr void *)(target_addr),
                        sizeof(tuplePair_t));
                }
            }
            else
            {
                printf("wram_buff[%d/%d].lvalue: %u %u\n", e, elem_n, wram_buff[e].lvalue, wram_buff[e].rvalue);
            }
        }
    }

    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        dpu_results.cycle_count = NR_CYCLE;
    }

    barrier_wait(&my_barrier);
    return 0;
}
