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
#define FILL_BYTE 1935
#define PARTITION_BL_LOG 10
#define PARTITION_BL_SIZE (1 << PARTITION_BL_LOG)
#define PARTITION_BL_ELEM (PARTITION_BL_SIZE >> 3)

#define NUM_DPU_LOG 6
#define NUM_DPU_RANK (1 << NUM_DPU_LOG)

#define NR_TASKLETS 12

#define HASH_TABLE_BUFF_SIZE 32768

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Variables from Host
__host packetwise_hash_local_partitioning_arg param_packetwise_local_hash_partitioning;
__host packetwise_hash_local_partitioning_return_arg param_packetwise_local_hash_partitioning_return;
__host dpu_results_t dpu_results;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

MUTEX_INIT(my_mutex);
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
uint32_t NR_CYCLE = 0;
// Radix Value
uint32_t RADIX;
uint32_t shift_len;

// Page Value
uint32_t PACKET_BYTES = 0;
// Total number of elements in all packets
uint32_t TOTAL_ELEM = 0;

// TIDs start Addr
char *Tids_src_addr;
char *Tids_dest_addr;

// relation R start addr
char *key_src_addr;
char *key_dest_addr;
// Partition destination
char *tid_addr;

// Page# Infos
char *packet_num_hist_addr;
// Partition Infos
char *partition_info_addr;

// Target Partition Size
uint32_t target_partition_size;
int32_t *write_buffer;

uint16_t last_block_idx = 1;
uint16_t Working_partition_idx = 0;
uint16_t End_partition_idx = 0;

uint32_t buffer_size_per_bucket;
bool Random_Access = false;

int PARTITION_NUM = 0;
int TOTAL_ELEM_NUM = 0;


// Global Vars for Do_Partition
uint32_t Source_Partition_ID = 0;
uint32_t Max_Source_Partition_ID = 0;

int Do_Partitioning_Packetwise(
	uint32_t tasklet_id,
	int shift_value,
	char *packet_src_addr,
	char *data_dest_addr,			// need to fill this as array
	char *tid_dest_addr,			// need to fill this as array
	char *partition_info_base_addr, // need to fill this
	void* packet_read_buffer,
	int tuple_size)
{
	uint32_t num_partition = RADIX + 1;

	int elem_in_buff = BLOCK_SIZE2 / param_packetwise_local_hash_partitioning.tuple_size;

	for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < PACKET_BYTES; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
	{
		// Read Page
		mram_read(
			(__mram_ptr void const *)(key_src_addr + byte_offset),
			packet_read_buffer,
			BLOCK_SIZE2);

		if (byte_offset + BLOCK_SIZE2 > PACKET_BYTES)
		{
			elem_in_buff = (PACKET_BYTES - byte_offset) / param_packetwise_local_hash_partitioning.tuple_size;
		}

			tuplePair_t* packet_read_buff = (tuplePair_t*)packet_read_buffer;
			// Build local histogram
			for (int i = 0; i < (elem_in_buff); i++)
			{
				if (packet_read_buff[i].lvalue == 0) continue;
				// Validation
				uint32_t tid_, key_;
				key_ = packet_read_buff[i].lvalue;

				// if (key_ < 0)
				// {
				// 	dpu_results.ERROR_TYPE_0 = 2;
				// 	break;
				// }
				// if (key_ == 0)
				// {
				// 	dpu_results.ERROR_TYPE_0 = 4;
				// 	break;
				// }

				uint32_t val = (RADIX & local_partition_hash(key_));

				mutex_lock(&(mutex_atomic[val & 0x1F]));
				write_buffer[val]++;
				mutex_unlock(&(mutex_atomic[val & 0x1F]));
			}
	}

	barrier_wait(&my_barrier);

	// Write Partition Info in MRAM
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

	//////////////////////////////////////////////////////////////
	// Do Partitioning
	elem_in_buff = BLOCK_SIZE2 / param_packetwise_local_hash_partitioning.tuple_size;

	for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < PACKET_BYTES; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
	{
		// Read Page
		mram_read(
			(__mram_ptr void const *)(key_src_addr + byte_offset),
			packet_read_buffer,
			BLOCK_SIZE2);

		if (byte_offset + BLOCK_SIZE2 > PACKET_BYTES)
		{
			elem_in_buff = (PACKET_BYTES - byte_offset) / param_packetwise_local_hash_partitioning.tuple_size;
		}

			tuplePair_t* packet_read_buff = (tuplePair_t*)packet_read_buffer;
			
			// Build local histogram
			for (int i = 0; i < (elem_in_buff); i++)
			{
				if (packet_read_buff[i].lvalue == 0) continue;
				uint32_t key_ = packet_read_buff[i].lvalue;

				// Validation
				// if (key_ < 0)
				// {
				// 	dpu_results.ERROR_TYPE_0 = 2;
				// 	break;
				// }

				uint32_t val = (RADIX & local_partition_hash(key_));

				mutex_lock(&(mutex_atomic[val & 0x1F]));
				uint32_t hist = write_buffer[val];
				write_buffer[val]++;
				mutex_unlock(&(mutex_atomic[val & 0x1F]));

				mram_write(
					&(packet_read_buff[i]),
					(__mram_ptr void *)(data_dest_addr + hist * sizeof(tuplePair_t)),
					sizeof(tuplePair_t));
			}
	}
	barrier_wait(&my_barrier);

	return 0;
}

void CalculatePartitionNumber(
	uint32_t tasklet_id,
	char *packet_src_addr,
	void *read_buffer,
	int tuple_size)
{
	int elem_in_buff = BLOCK_SIZE2 / tuple_size;

	for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < PACKET_BYTES; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
	{
		// Read Page
		mram_read(
			(__mram_ptr void const *)(packet_src_addr + byte_offset),
			read_buffer,
			BLOCK_SIZE2);

		if (byte_offset + BLOCK_SIZE2 > PACKET_BYTES)
		{
			elem_in_buff = (PACKET_BYTES - byte_offset) / tuple_size;
		}

		int counted_elem = 0;

		// if (tuple_size == 8)
		// {
			tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;

			for (int e = 0; e < elem_in_buff; e++)
			{
				if (packet_read_buff[e].lvalue == 0)
				{
					continue;
				}
				else
				{
					counted_elem++;
				}
			}

		// Build local histogram
		mutex_lock(my_mutex);
		TOTAL_ELEM_NUM += counted_elem;
		mutex_unlock(my_mutex);
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
		dpu_results.ERROR_TYPE_0 = 0;
		dpu_results.ERROR_TYPE_1 = 0;
		dpu_results.ERROR_TYPE_2 = 0;
		dpu_results.ERROR_TYPE_3 = 0;
		perfcounter_config(COUNT_CYCLES, 1);
	}

	barrier_wait(&my_barrier);

	// Allocate Buffer
	void *read_buffer = NULL;
	read_buffer = mem_alloc(BLOCK_SIZE2);

	int shift_val = param_packetwise_local_hash_partitioning.shift_val;

	if (tasklet_id == 0)
	{
		key_src_addr = (char *)MRAM_BASE_ADDR + param_packetwise_local_hash_partitioning.packet_start_byte;
		key_dest_addr = (char *)MRAM_BASE_ADDR + param_packetwise_local_hash_partitioning.partitioned_result_start_byte;
		partition_info_addr = (char *)MRAM_BASE_ADDR + param_packetwise_local_hash_partitioning.result_partition_info_start_byte;
		PACKET_BYTES = param_packetwise_local_hash_partitioning.num_packets * param_packetwise_local_hash_partitioning.packet_size;
		printf("PACKET_BYTES: %u num_packets: %d packet_size: %d\n",
			   PACKET_BYTES,
			   param_packetwise_local_hash_partitioning.num_packets,
			   param_packetwise_local_hash_partitioning.packet_size);
		write_buffer = (int32_t*)mem_alloc(32 * 1024);
	}

	// Calculate Page Num
	barrier_wait(&my_barrier);

	CalculatePartitionNumber(
		tasklet_id,
		key_src_addr,
		read_buffer,
		param_packetwise_local_hash_partitioning.tuple_size);

	barrier_wait(&my_barrier);

	if (param_packetwise_local_hash_partitioning.do_calculate_partition_num == 0)
	{

		if (tasklet_id == 0)
		{
			int temp_part = (TOTAL_ELEM_NUM / FILL_BYTE);
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
			PARTITION_NUM = param_packetwise_local_hash_partitioning.do_calculate_partition_num;
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
		int elem_in_buff = BLOCK_SIZE2 / param_packetwise_local_hash_partitioning.tuple_size;

		for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE2; byte_offset < PACKET_BYTES; byte_offset += (NR_TASKLETS * BLOCK_SIZE2))
		{
			// Read Page
			mram_read(
				(__mram_ptr void const *)(key_src_addr + byte_offset),
				read_buffer,
				BLOCK_SIZE2);

			if (byte_offset + BLOCK_SIZE2 > PACKET_BYTES)
			{
				elem_in_buff = (PACKET_BYTES - byte_offset) / param_packetwise_local_hash_partitioning.tuple_size;
			}

			if (param_packetwise_local_hash_partitioning.tuple_size == 8)
			{
				tuplePair_t *packet_read_buff = (tuplePair_t *)read_buffer;
				// Build local histogram
				for (int i = 0; i < elem_in_buff; i++)
				{
					if (packet_read_buff[i].lvalue == 0)
						continue;
					// Validation
					int64_t key_;
					key_ = packet_read_buff[i].lvalue;

					if (key_ < 0)
					{
						dpu_results.ERROR_TYPE_1 = 2;
						break;
					}
					if (key_ == 0)
					{
						dpu_results.ERROR_TYPE_2 = 4;
						break;
					}

					// uint32_t val = (RADIX & local_partition_hash(key_ >> shift_val, 0x01234567));
					mutex_lock(my_mutex);
					uint32_t hist = write_buffer[0];
					mutex_unlock(my_mutex);

					mram_write(
						&(packet_read_buff[i]),
						(__mram_ptr void *)(key_dest_addr + hist * sizeof(tuplePair_t)),
						sizeof(tuplePair_t));
				}
			}
		}

		barrier_wait(&my_barrier);

		if (tasklet_id == 0)
		{
			int32_t *histogram = (int32_t *)write_buffer;
			int temp = histogram[0];
			histogram[0] = 0;
			histogram[1] = 0;
			mram_write(histogram, (__mram_ptr void *)(partition_info_addr), sizeof(TupleID64_t));
			histogram[0] = temp;
		}
	}
	else
	{
		// Build histogram & Local Partitioning
		Do_Partitioning_Packetwise(
			tasklet_id,
			shift_val,
			key_src_addr,
			key_dest_addr,
			tid_addr,
			partition_info_addr,
			read_buffer,
			param_packetwise_local_hash_partitioning.tuple_size);
	}

	barrier_wait(&my_barrier);

	/* Partitioning Relation S */
	if (tasklet_id == 0)
	{

		param_packetwise_local_hash_partitioning_return.elem_num = ((int32_t *)write_buffer)[PARTITION_NUM - 1];
		param_packetwise_local_hash_partitioning_return.partition_num = PARTITION_NUM;
		NR_CYCLE = perfcounter_get();
		// printf("INSTR: %d\n", NR_CYCLE);
		// printf("\
		// param_packetwise_local_hash_partitioning.num_packets: %d\n\
		// PARTITION_NUM:%d\n\
		// packetwise_hash_local_partitioning_return_arg.elem_num: %lld %d\n",
		// 	   param_packetwise_local_hash_partitioning.num_packets,
		// 	   PARTITION_NUM,
		// 	   param_packetwise_local_hash_partitioning_return.elem_num, TOTAL_ELEM_NUM);
		dpu_results.cycle_count = NR_CYCLE;

		for (int i = 1; i < PARTITION_NUM; i++)
		{
			if (((int32_t *)write_buffer)[i] < ((int32_t *)write_buffer)[i - 1])
			{
				dpu_results.ERROR_TYPE_3 = 1;
				break;
			}
		}

		printf("partiton: %d | ", param_packetwise_local_hash_partitioning.do_calculate_partition_num);
		for (int i = 0; i < PARTITION_NUM -1; i++)
		{
			printf("%d, ", (write_buffer)[i]);
			if (param_packetwise_local_hash_partitioning.do_calculate_partition_num == 0)
			{
				int32_t val = ((write_buffer)[i + 1] - (write_buffer)[i]);
				if (val >= (HASH_TABLE_BUFF_SIZE / sizeof(tuplePair_t)))
				{
			
					dpu_results.ERROR_TYPE_1 = ((int32_t *)write_buffer)[i+1];
					dpu_results.ERROR_TYPE_2 = ((int32_t *)write_buffer)[i];
					dpu_results.ERROR_TYPE_3 = dpu_results.ERROR_TYPE_3 = param_packetwise_local_hash_partitioning.packet_size;
				}
			}
		}
		printf("\n");
	}

	return 0;
}