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
#define PARTITION_BL_SIZE 1024
#define HASH_TABLE_BUFF_SIZE 32768

#define NR_TASKLETS 12

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Variables from Host
__host hash_local_partitioning_arg param_local_hash_partitioning;
__host hash_local_partitioning_return_arg param_local_hash_partitioning_return;
__host dpu_results_t dpu_results;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

// #define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
// #define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
BARRIER_INIT(my_barrier2, 13);

uint32_t NB_CYCLE = 0;
// Radix Value
uint32_t RADIX;
uint32_t shift_len;

// TIDs start Addr
char *Tids_src_addr;
char *Tids_dest_addr;

// relation R start addr
char *R_addr;
char *S_addr;

// Partition destination
char *R_part_addr1;
char *S_part_addr1;

// Partition Infos
char *R_partition_info_addr;
char *S_partition_info_addr;

// Target Partition Size
uint32_t target_partition_size;
uint32_t* write_buffer;

uint16_t last_block_idx = 1;
uint16_t Working_partition_idx = 0;
uint16_t End_partition_idx = 0;

uint32_t buffer_size_per_bucket;
bool Random_Access = false;

void SetHistogram(int32_t idx, uint32_t value)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    (*wb) = value;
}

void IncrHistogram(int32_t idx)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    *wb += 1;
}

uint32_t GetIncrHistogram(int32_t idx)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    *wb += 1;
    return (*wb - 1);
}

uint32_t GetHistogram(int32_t idx)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    return *wb;
}

void AddHistogram(int32_t idx, uint32_t val)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    *wb += val;
}

void SubHistogram(int32_t idx, uint32_t val)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    *wb -= val;
}

// Global Vars for Do_Partition
uint32_t Source_Partiton_ID = 0;
uint32_t Max_Source_Partition_ID = 0;

int PARTITION_NUM = 0;
int TOTAL_ELEM_NUM = 0;

int Do_Partitioning(
    uint32_t tasklet_id,
    char *src_addr,
    char *dst_addr,                 // need to fill this as array
    char *partition_info_base_addr, // need to fill this
    tuplePair_t *read_buffer,
    int tuple_size)
{
    uint32_t num_partition = RADIX + 1;

    int total_bytes = TOTAL_ELEM_NUM * tuple_size;
    int elem_num_in_buff = BLOCK_SIZE2 / tuple_size;
    // int elem_in_buff = BLOCK_SIZE2 / param_packetwise_local_hash_partitioning.tuple_size

    for (int byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < total_bytes; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
    {
        // Read Page
        mram_read(
            (__mram_ptr void const *)(src_addr + byte_offset),
            read_buffer,
            BLOCK_SIZE2);

        if (byte_offset + BLOCK_SIZE2 > total_bytes)
        {
            elem_num_in_buff = (total_bytes - byte_offset) / tuple_size;
        }

        if (tuple_size == 8)
        {
            tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;
            // Build local histogram
            uint32_t key_;
            for (int i = 0; i < elem_num_in_buff; i++)
            {
                key_ = packet_read_buff[i].lvalue;

#ifdef VALIDATION
                if (key_ < 0)
                {
                    dpu_results.ERROR_TYPE_0 = 2;
                    break;
                }
                if (key_ == 0)
                {
                    dpu_results.ERROR_TYPE_0 = 4;
                    break;
                }
#endif

                uint32_t val = (RADIX & local_partition_hash(key_));

                mutex_lock(&(mutex_atomic[val & 0x1F]));
                IncrHistogram(val);
                mutex_unlock(&(mutex_atomic[val & 0x1F]));
            }
        }
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
	{
		int loops = num_partition * sizeof(uint32_t) / BLOCK_SIZE2;
		int remains = num_partition * sizeof(uint32_t) % BLOCK_SIZE2;

		// Modify Histogram as start index
		uint32_t temp_before = write_buffer[0];
		uint32_t temp_before_2 = write_buffer[0];

		write_buffer[0] = 0;

		for (uint32_t i = 1; i < num_partition; i++)
		{
			temp_before_2 = write_buffer[i];
			write_buffer[i] = write_buffer[i - 1] + temp_before;
			temp_before = temp_before_2;
		}

		for (int i = 0; i < loops; i++)
			mram_write(
				(void *)(write_buffer + i * BLOCK_SIZE2),
				(__mram_ptr void *)((uint32_t)partition_info_base_addr + i * BLOCK_SIZE2),
				BLOCK_SIZE2);
		if (remains > 0)
			mram_write(
				(void *)(write_buffer + loops * BLOCK_SIZE2),
				(__mram_ptr void *)((uint32_t)partition_info_base_addr + loops * BLOCK_SIZE2),
				remains);
	}

    barrier_wait(&my_barrier);

    elem_num_in_buff = BLOCK_SIZE2 / tuple_size;
    //////////////////////////////////////////////////////////////
    // Do Partitioning
    for (int byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < total_bytes; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
    {
        // Read Page
        mram_read(
            (__mram_ptr void const *)(src_addr + byte_offset),
            read_buffer,
            BLOCK_SIZE2);

        if (byte_offset + BLOCK_SIZE2 > total_bytes)
        {
            elem_num_in_buff = (total_bytes - byte_offset) / tuple_size;
        }

        if (tuple_size == 8)
        {
            tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;
            // Build local histogram
            for (int i = 0; i < elem_num_in_buff; i++)
            {
                if (packet_read_buff[i].lvalue == 0)
                    continue;
                // Validation
                uint32_t key_;
                key_ = packet_read_buff[i].lvalue;

                uint32_t val = (RADIX & local_partition_hash(key_));

                mutex_lock(&(mutex_atomic[val & 0x1F]));
				uint32_t hist = write_buffer[val];
				write_buffer[val]++;
				mutex_unlock(&(mutex_atomic[val & 0x1F]));

                mram_write(
                    &(packet_read_buff[i]),
                    (__mram_ptr void *)(dst_addr + hist * sizeof(tuplePair_t)),
                    sizeof(tuplePair_t));
            }
        }
    }
    barrier_wait(&my_barrier);

    return 0;
}

void CalculatePartitionNumber(
    uint32_t tasklet_id,
    char *packet_src_addr,
    tuplePair_t *read_buffer,
    int tuple_size)
{
    int total_bytes = param_local_hash_partitioning.input_data_bytes;
    int elem_num_in_buff = BLOCK_SIZE2 / tuple_size;

    for (int32_t byte_offset = (tasklet_id * BLOCK_SIZE2); byte_offset < total_bytes; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
    {
        // Read Page
        mram_read(
            (__mram_ptr void const *)(packet_src_addr + byte_offset),
            read_buffer,
            BLOCK_SIZE2);

        if (byte_offset + BLOCK_SIZE2 > total_bytes)
        {
            elem_num_in_buff = (total_bytes - byte_offset) / tuple_size;
        }

        int num_elem = 0;

        if (tuple_size == 8)
        {
            tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;
            for (int e = 0; e < elem_num_in_buff; e++)
            {
                if (packet_read_buff[e].lvalue == 0)
                {
                    continue;
                }
                else
                {
                    num_elem++;
                }
            }
        }

        // Build local histogram
        mutex_lock(&(mutex_atomic[49]));
        TOTAL_ELEM_NUM += num_elem;
        mutex_unlock(&(mutex_atomic[49]));
        // Calculating local histogram Done
    }
}

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
        R_addr = (char *)MRAM_BASE_ADDR + param_local_hash_partitioning.input_arr_start_byte;
        R_part_addr1 = (char *)MRAM_BASE_ADDR + param_local_hash_partitioning.partitioned_input_arr_start_byte;
        R_partition_info_addr = (char *)MRAM_BASE_ADDR + param_local_hash_partitioning.result_partition_info_start_byte;
        write_buffer = (uint32_t *)mem_alloc(32 * 1024);
    }

    barrier_wait(&my_barrier);

    tuplePair_t *read_buffer = (tuplePair_t *)mem_alloc(BLOCK_SIZE2);

    /* Setup Done */
    barrier_wait(&my_barrier);

    CalculatePartitionNumber(
        tasklet_id,
        R_addr,
        read_buffer,
        param_local_hash_partitioning.tuple_size);

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        if (param_local_hash_partitioning.input_data_bytes / sizeof(tuplePair_t) != TOTAL_ELEM_NUM)
        {
            printf("ERROR: %d vs %d\n", param_local_hash_partitioning.input_data_bytes, TOTAL_ELEM_NUM);
            return 0;
        }
    }

    barrier_wait(&my_barrier);

    if (param_local_hash_partitioning.do_calculate_partition_num == 0)
    {

        if (tasklet_id == 0)
        {
            printf("TOTAL_ELEM_NUM: %d\n", TOTAL_ELEM_NUM);
            int temp_part = (TOTAL_ELEM_NUM / 2000);
            PARTITION_NUM = 1;
            while (true)
            {
                if (temp_part > 0)
                {
                    temp_part = temp_part >> 1;
                    PARTITION_NUM = PARTITION_NUM << 1;
                }
                else
                {
                    break;
                }
            }

            RADIX = PARTITION_NUM - 1;

            // Memset Histogram
			for (int32_t i = 0; i < PARTITION_NUM; i++)
			{
				write_buffer[i] = 0;
			}
        }
    }
    else
    {
        if (tasklet_id == 0)
        {
            PARTITION_NUM = param_local_hash_partitioning.do_calculate_partition_num;
            RADIX = PARTITION_NUM - 1;

            // Memset Histogram
			for (int32_t i = 0; i < PARTITION_NUM; i++)
			{
				// SetHistogram(i, 0);
				write_buffer[i] = 0; 
			}
        }
    }

    /* Setup Done */
    barrier_wait(&my_barrier);

    if (RADIX == 0)
    {
        uint32_t num_partition = RADIX + 1;

        int total_bytes = TOTAL_ELEM_NUM * param_local_hash_partitioning.tuple_size;
        int elem_num_in_buff = BLOCK_SIZE2 / param_local_hash_partitioning.tuple_size;

        for (int byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < total_bytes; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
        {
            // Read Page Data
            mram_read(
                (__mram_ptr void const *)(R_addr + byte_offset),
                read_buffer,
                BLOCK_SIZE2);

            if (byte_offset + BLOCK_SIZE2 > total_bytes)
            {
                elem_num_in_buff = (total_bytes - byte_offset) / param_local_hash_partitioning.tuple_size;
            }

            if (param_local_hash_partitioning.tuple_size == 8)
            {
                tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;
                // Build local histogram
                for (int i = 0; i < elem_num_in_buff; i++)
                {
                    // Validation
                    uint32_t key_;
                    key_ = packet_read_buff[i].lvalue;

                    #ifdef VALIDATION
                    if (key_ < 0)
                    {
                        dpu_results.ERROR_TYPE_0 = 2;
                        break;
                    }
                    if (key_ == 0)
                    {
                        dpu_results.ERROR_TYPE_0 = 4;
                        break;
                    }
                    #endif

                    mutex_lock(&(mutex_atomic[49]));
                    uint32_t hist = write_buffer[0];
                    mutex_unlock(&(mutex_atomic[49]));

                    mram_write(
                        &(packet_read_buff[i]),
                        (__mram_ptr void *)(R_part_addr1 + hist * sizeof(tuplePair_t)),
                        sizeof(tuplePair_t));
                }
            }

            barrier_wait(&my_barrier);

            if (tasklet_id == 0)
            {
                uint32_t *histogram = (uint32_t *)write_buffer;
                int temp = histogram[0];
                histogram[0] = 0;
                histogram[1] = 0;
                mram_write(histogram, (__mram_ptr void *)(R_partition_info_addr), sizeof(TupleID64_t));
                histogram[0] = temp;
            }
        }
    }
    else
    {
        // Build histogram & Local Partitioning
        Do_Partitioning(
            tasklet_id,
            R_addr,
            R_part_addr1,
            R_partition_info_addr,
            read_buffer,
            param_local_hash_partitioning.tuple_size);
    }

    barrier_wait(&my_barrier);

    /* Partitioning Relation S */
    if (tasklet_id == 0)
    {

        param_local_hash_partitioning_return.elem_num = ((int32_t *)write_buffer)[PARTITION_NUM - 1];
        param_local_hash_partitioning_return.partition_num = PARTITION_NUM;
        NB_CYCLE = perfcounter_get();
        printf("Probe NB_CYCLE: %lu\n", NB_CYCLE);
        dpu_results.cycle_count = NB_CYCLE;
        
        printf("PARTITION_NUM:%d\n", PARTITION_NUM);

        if (param_local_hash_partitioning_return.elem_num != TOTAL_ELEM_NUM)
        {
            dpu_results.ERROR_TYPE_0 = 100;
        }

        for (int i = 1; i < PARTITION_NUM; i++)
        {
            if (((int32_t *)write_buffer)[i] < ((int32_t *)write_buffer)[i - 1])
            {
                dpu_results.ERROR_TYPE_0 = 1;
                break;
            }
        }

        printf("partiton num:%d: ", PARTITION_NUM);
        for (int i = 0; i < PARTITION_NUM; i++)
        {
            printf("%d, ", ((int32_t *)write_buffer)[i]);
            if (param_local_hash_partitioning.do_calculate_partition_num == 0)
            {
                uint32_t val = (((int32_t *)write_buffer)[i + 1] - ((int32_t *)write_buffer)[i]);
                if (val >= (HASH_TABLE_BUFF_SIZE / sizeof(tuplePair_t)))
                {
                    dpu_results.ERROR_TYPE_0 = val;
                    printf("%s %d, ", KRED, val);
                    printf("%s", KWHT);
                }
            }
        }
        printf("\n");
    }

    if (tasklet_id == 0)
    {
        NB_CYCLE = perfcounter_get();
        printf("Probe NB_CYCLE: %lu\n", NB_CYCLE);
        dpu_results.cycle_count = NB_CYCLE;
    }
    barrier_wait(&my_barrier);
    return 0;
}
