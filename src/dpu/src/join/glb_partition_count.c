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
__host glb_partition_count_arg param_glb_partition_count;
__host dpu_results_t dpu_results;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t NR_CYCLE = 0;

// Radix Value
uint32_t RADIX;

// TIDs start Addr
char* Tids_src_addr;
char* Tids_dest_addr;

// relation R start addr
char* R_addr;

// Partition Infos
char* R_partition_info_addr;
char* R_histogram_addr;

// Target Partition Size
char* write_buffer;

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
    uint32_t *wb = ((uint32_t *)write_buffer + idx);
    *wb += 1;
}

uint32_t GetIncrHistogram(int32_t idx)
{
    uint32_t *wb = ((uint32_t *)write_buffer + idx);
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


inline uint32_t tid_partition(int32_t tid)
{
    int32_t dpu_id;
    GET_RANK_DPU_FROM_TUPLE_ID(dpu_id, tid);
    return dpu_id;
}

// Global Vars for PartitionCalculation
uint32_t Source_Partiton_ID = 0;
uint32_t Max_Source_Partition_ID = 0;

int HashPartitionCalculationArray(
    uint32_t tasklet_id,
    uint32_t mram__source_addr, 
    uint32_t mram_partition_info_base_addr, 
    uint32_t mram_histogram_base_addr, 
    uint32_t num_elem,
    tuplePair_t *wram_read_buffer_payload,
    int num_partition)
{
    int RADIX = num_partition - 1;
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
        for (int i = 0; i < num_partition; i++)
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
            mutex_unlock(&(mutex_atomic[50])); break;
        }
        mutex_unlock(&(mutex_atomic[50]));

        int Read_size = BLOCK_SIZE2;                                                                                                                                                      
        int elem_size = ELEM_PER_BLOCK2;

        if (my_id == (Max_Source_Partition_ID - 1))
        {
            Read_size = (remained_elem << 3);
            elem_size = remained_elem;
        }
        // Read Data
        mram_read(
            (__mram_ptr void const *)(mram__source_addr + my_id * BLOCK_SIZE2), 
            wram_read_buffer_payload, 
            Read_size);

        for (int i = 0; i < elem_size; i++)
        {
            if (wram_read_buffer_payload[i].lvalue != 0)
            {

            // uint32_t val = (RADIX & wram_read_buffer[i]);
            uint32_t val = (RADIX & glb_partition_hash(wram_read_buffer_payload[i].lvalue));

            #ifdef VALIDATION
            if (wram_read_buffer_payload[i].lvalue == 0)
            {
                // pass
                dpu_results.ERROR_TYPE_0 = 2;
            }
            #endif

            mutex_lock(&(mutex_atomic[val & 0x1F]));
            IncrHistogram(val);
            mutex_unlock(&(mutex_atomic[val & 0x1F]));
            }
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
        
        for (int i = 1; i < num_partition; i++)
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

    barrier_wait(&my_barrier);
    return 0;
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
        
        write_buffer = (char*)mem_alloc(2048 * 4);
        memset(write_buffer, 0, param_glb_partition_count.partition_num*sizeof(int32_t));
    }
    barrier_wait(&my_barrier);
    

    Key64_t *read_buffer_payload = NULL;
    TupleID64_t *read_buffer_tid = NULL;
    TupleID64_t *read_buffer_pair = NULL;

    barrier_wait(&my_barrier);

    // Allocate Buffers
    read_buffer_payload = (tuplePair_t*)mem_alloc(BLOCK_SIZE2);

    HashPartitionCalculationArray(
        tasklet_id,
        MRAM_BASE_ADDR + param_glb_partition_count.input_offset, 
        MRAM_BASE_ADDR + param_glb_partition_count.partition_info_start_byte, 
        MRAM_BASE_ADDR + param_glb_partition_count.histogram_start_byte, 
        param_glb_partition_count.elem_num,
        read_buffer_payload,
        param_glb_partition_count.partition_num);
    
    barrier_wait(&my_barrier);

    /* Partitioning Relation S */
    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        printf("INSTR: %d\n", NR_CYCLE);
        dpu_results.cycle_count = NR_CYCLE;
    }

    barrier_wait(&my_barrier);
    return 0;
}
