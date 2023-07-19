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

#define NR_TASKLETS 12

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Variables from Host
__host hash_global_partitioning_arg param_global_hash_partitioning;
__host dpu_results_t dpu_results;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint64_t NB_CYCLE = 0;
// Radix Value
uint32_t RADIX;
uint32_t shift_len;

// TIDs start Addr
char* Tids_src_addr;
char* Tids_dest_addr;

// relation R start addr
char* R_addr;
char* S_addr;

// Partition destination
char* Partitioned_R_addr;

// Partition Infos
char* R_partition_info_addr;
char* R_histogram_addr;
// Target Partition Size
uint32_t target_partition_size;
char* write_buffer;

uint16_t last_block_idx = 1;
uint16_t Working_partition_idx = 0;
uint16_t End_partition_idx = 0;

uint32_t buffer_size_per_bucket;
bool Random_Access = false;

inline char* GetWriteBufferAddr(int32_t partition_idx)
{
    return (write_buffer + partition_idx);
}

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

int Do_Partition(
    uint32_t tasklet_id,
    char* mram_source_addr, 
    char* mram_dest_addr,
    char* mram_partition_info_base_addr, 
    char* mram_histogram_base_addr, 
    uint32_t num_elem,
    tuplePair_t* wram_read_buffer)
{
    uint32_t num_partition = RADIX + 1;
    uint32_t remained_elem = (num_elem % (ELEM_PER_BLOCK2));

    // Build Histogram
    if (tasklet_id == 0)
    {
        Source_Partiton_ID = 0;
        Max_Source_Partition_ID = ((num_elem << 3) / BLOCK_SIZE2);

        if (remained_elem > 0)
        {
            Max_Source_Partition_ID++;
        }
    }

    if (remained_elem == 0)
    {
        remained_elem = ELEM_PER_BLOCK2;
    }

    if (tasklet_id == 1)
    {
        // Memset Histogram
        for (uint32_t i = 0; i < num_partition; i++)
            SetHistogram(i, 0);
    }

    barrier_wait(&my_barrier);

    // make histogram
    while (1)
    {
        // Get ID
        mutex_lock(&(mutex_atomic[50]));
        // Get Source partition id
        uint32_t my_id = Source_Partiton_ID++;
        // End Condition
        if (my_id >= Max_Source_Partition_ID)
        {
            mutex_unlock(&(mutex_atomic[50]));
            break;
        }
        mutex_unlock(&(mutex_atomic[50]));

        int Read_size = BLOCK_SIZE2;
        int elem_size = ELEM_PER_BLOCK2;

        if (my_id == (Max_Source_Partition_ID - 1))
        {
            Read_size = remained_elem << 3;
            elem_size = remained_elem;
        }
        // Read Data
        mram_read((__mram_ptr void const *)(mram_source_addr + my_id * BLOCK_SIZE2), wram_read_buffer, Read_size);

        for (int i = 0; i < elem_size; i++)
        {
            uint32_t val = (RADIX & glb_partition_hash(wram_read_buffer[i].lvalue));

            if (wram_read_buffer[i].lvalue == 0)
            {
                dpu_results.ERROR_TYPE_0 = 1;
            }

            mutex_lock(&(mutex_atomic[val & 0x1F]));
            IncrHistogram(val);
            mutex_unlock(&(mutex_atomic[val & 0x1F]));
        }
        // calculating local histogram Done
    }
    barrier_wait(&my_barrier);

    // Make global histogram
    
    if (tasklet_id == 0)
    {
        int loops = (num_partition) * sizeof(uint32_t) / BLOCK_SIZE2;
        int remains = (num_partition) * sizeof(uint32_t) % BLOCK_SIZE2;
        for (int i = 0; i < loops; i++)
            mram_write((void*)(write_buffer + i * BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_histogram_base_addr + i * BLOCK_SIZE2), BLOCK_SIZE2);
        if (remains > 0)
            mram_write((void*)(write_buffer + (loops)*BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_histogram_base_addr + (loops)*BLOCK_SIZE2), remains);
    
        // Modify Histogram as start index
        uint32_t temp_before = GetHistogram(0);
        uint32_t temp_before_2 = GetHistogram(0);

        SetHistogram(0, 0);
        
        for (uint32_t i = 1; i < num_partition; i++)
        {
            temp_before_2 = GetHistogram(i);
            SetHistogram(i, GetHistogram(i - 1) + temp_before);
            temp_before = temp_before_2;
        }

        
        for (int i = 0; i < loops; i++)
            mram_write((void*)(write_buffer + i * BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_partition_info_base_addr + i * BLOCK_SIZE2), BLOCK_SIZE2);
        if (remains > 0)
            mram_write((void*)(write_buffer + (loops)*BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_partition_info_base_addr + (loops)*BLOCK_SIZE2), remains);
    }

    //////////////////////////////////////////////////////////////
    // Do Partitioning
    if (tasklet_id == 1)
    {
        Source_Partiton_ID = 0;
    }

    barrier_wait(&my_barrier);

    // make histogram
    while (1)
    {
        // Get ID
        mutex_lock(&(mutex_atomic[50]));
        // Get Source partition id
        uint32_t my_id = Source_Partiton_ID++;
        // End Condition
        if (my_id >= Max_Source_Partition_ID)
        {
            mutex_unlock(&(mutex_atomic[50]));
            break;
        }
        mutex_unlock(&(mutex_atomic[50]));

        int Read_size = BLOCK_SIZE2;
        int elem_size = ELEM_PER_BLOCK2;

        if (my_id == (Max_Source_Partition_ID - 1))
        {
            Read_size = remained_elem << 3;
            elem_size = remained_elem;
        }

        // Read Data
        mram_read((__mram_ptr void const *)(mram_source_addr + my_id * BLOCK_SIZE2), wram_read_buffer, Read_size);

        for (int i = 0; i < elem_size; i++)
        {
            #ifdef VALIDATION
            if (wram_read_buffer[i].lvalue == 0)
            {
                dpu_results.ERROR_TYPE_0 = 9;
            }
            #endif
            // uint32_t val = (RADIX & wram_read_buffer[i]);
            uint32_t val = (RADIX & glb_partition_hash(wram_read_buffer[i].lvalue));

            mutex_lock(&(mutex_atomic[val & 0x1F]));
            uint32_t hist = GetIncrHistogram(val);
            mutex_unlock(&(mutex_atomic[val & 0x1F]));
            mram_write(&(wram_read_buffer[i]), (__mram_ptr void *)(mram_dest_addr + (hist * sizeof(tuplePair_t))), sizeof(tuplePair_t));

        }
        // calculating local histogram Done
    }
    barrier_wait(&my_barrier);
    return 0;
}

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        dpu_results.ERROR_TYPE_0 = 0;
		dpu_results.ERROR_TYPE_1 = 0;
		dpu_results.ERROR_TYPE_2 = 0;
		dpu_results.ERROR_TYPE_3 = 0;
        mem_reset();
        perfcounter_config(COUNT_CYCLES, 1);
        RADIX = param_global_hash_partitioning.partition_num - 1;
    }

    barrier_wait(&my_barrier);

    tuplePair_t* read_buffer = NULL;

    read_buffer = (tuplePair_t*)mem_alloc(BLOCK_SIZE2);

    if (tasklet_id == 0)
    {
        R_addr = (char*)MRAM_BASE_ADDR + param_global_hash_partitioning.table_r_start_byte;
        Partitioned_R_addr = (char*)MRAM_BASE_ADDR + param_global_hash_partitioning.partitioned_table_r_start_byte;
        R_partition_info_addr = (char*)MRAM_BASE_ADDR + param_global_hash_partitioning.partition_info_start_byte;
        R_histogram_addr = (char*)MRAM_BASE_ADDR + param_global_hash_partitioning.histogram_start_byte;
        write_buffer = (char*)mem_alloc(32 * 1024);
    }

    /* Setup Done */
    barrier_wait(&my_barrier);

    Do_Partition(
        tasklet_id, 
        R_addr, 
        Partitioned_R_addr, 
        R_partition_info_addr, 
        R_histogram_addr, 
        param_global_hash_partitioning.table_r_num_elem, 
        read_buffer);

    barrier_wait(&my_barrier);

    /* Partitioning Relation S */
    if (tasklet_id == 0)
    {
        NB_CYCLE = perfcounter_get();
        printf("Probe NB_CYCLE: %lu\n", NB_CYCLE);
        dpu_results.cycle_count = NB_CYCLE;

    

        // validate
        // for (int i = 0; i < param_global_hash_partitioning.partition_num; i++)
		// {
		// 	printf("%d, ", ((int32_t *)write_buffer)[i]);
		// }
        if (((int32_t *)write_buffer)[param_global_hash_partitioning.partition_num-1] != param_global_hash_partitioning.table_r_num_elem)
        {
            // printf("%d vs %d", ((int32_t *)write_buffer)[param_global_hash_partitioning.partition_num-1], param_global_hash_partitioning.table_r_num_elem);
            dpu_results.ERROR_TYPE_3 = 99;
        }
        else
        {
            printf("good.\n");
        }
    }

    barrier_wait(&my_barrier);

    return 0;
}