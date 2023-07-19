#pragma once

#include <vector>
#include <unordered_map>

#define NUM_DPU_RANK 64 // # DPU per rank - Comment
#define DPUWISE_TUPLEID_OFFSET 0x800000
#define MRAM_SIZE 0x4000000 // 64MB
#define BANK_CHUNK_SIZE 0x20000


// DPU Side Functions
enum FUNCTION_TYPES {
    MICROBENCHMARK_MRAM_BANDWIDTH = 0,
    MICROBENCHMARK_WRAM_BANDWIDTH,
    MICROBENCHMARK_OPS,
    MICROBENCHMARK_HASH,
    DPU_FUNC_LOCAL_HASH_PARTITIONING,
    DPU_FUNC_GLOBAL_HASH_PARTITIONING,
    DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE,
    DPU_FUNC_PHJ_PROBE_HASH_TABLE,
    DPU_FUNC_NPHJ_BUILD_HASH_TABLE,
    DPU_FUNC_NPHJ_PROBE_HASH_TABLE,
    DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING,
    DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING,
    DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE,
    DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE,
    DPU_FUNC_MPSM_JOIN_PARTITIONING,
    DPU_FUNC_MPSM_JOIN_PROBE,
    DPU_FUNC_NESTED_LOOP_JOIN,
    DPU_FUNC_MPSM_JOIN_SORT,
    DPU_FUNC_MPSM_JOIN_RADIX_SORT,
    DPU_FUNC_MPSM_JOIN_MERGE_SORT,
    DPU_FUNC_MPSM_JOIN_QUICK_SORT,
    DPU_FUNC_MPSM_JOIN_PROBE_ALL,
    DPU_FUNC_GLB_PARTITION_COUNT,
    DPU_FUNC_GLB_PARTITION_PACKET,
    DPU_FUNC_FINISH_JOIN,
    HOST_FUNC_INVALIDATE_STACKNODE,
    HOST_FUNC_ROTATE_AND_STREAM,
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM,
    HOST_FUNC_ROTATE_AND_STREAM_OPT,
    HOST_FUNC_LOAD_COLUMN,
    HOST_FUNC_SEND_DATA_OPT,
    HOST_FUNC_RECV_DATA_OPT,
    HOST_FUNC_SEND_DATA_UPMEMLIB,
    HOST_FUNC_RECV_DATA_UPMEMLIB,
    COMPOUND_FUNC_UPMEM_JOIN,
    COMPOUND_FUNC_RNS_JOIN,
    COMPOUND_FUNC_GLB_PARTITION,
    COMPOUND_FUNC_GLB_PARTITION_CPU,
    COMPOUND_FUNC_SORT_MERGE_JOIN,
    COMPOUND_FUNC_SORT_MERGE_JOIN_ALL,
    TEST_FUNC_VALIDATE_PACKETWISE_DATA,
    TEST_FUNC_VALIDATE_LOCAL_PARTITIONED_DATA,
    TEST_FUNC_ROTATE_AND_CACHE_BYPASS,
    TEST_FUNC_PIM2HOST,
    TEST_FUNC_HOST2PIM,
    CONTROL_FUNC_SYNC_THREADS,
};

typedef struct {
    uint32_t lvalue;
    uint32_t rvalue;
} kv_pair_t;

namespace pidjoin
{
    enum DataType
    {
        NOT_SPECIFIED,
        INTEGER,
        INTEGER8,
        INTEGER32,
        CHAR,
        VARCHAR,
        DATE,
        DECIMAL,
        TUPLEID,
    };

    typedef std::unordered_map<std::string, int> ENCODING_TABLE_t;
    typedef std::vector<std::vector<char *>> RankwiseMemoryBankBuffers_t; // start address of each DPUs? - Comment
    typedef std::vector<std::vector<int>> RankwiseMemoryBankFilledBytes_t; // how much data is in each DPUs - Comment
    typedef std::vector<void *> DPUKernelParams_t;
    typedef std::vector<std::vector<void *>> DPUKernelParamsVector_t;

    /* RankwiseMemoryBankBufferPair_t contains 64 buffers and 64 int values to indicate how many byte data are stored for each buffer. */
    typedef std::pair<RankwiseMemoryBankBuffers_t *, RankwiseMemoryBankFilledBytes_t *> RankwiseMemoryBankBufferPair_t;
    typedef std::pair<std::vector<kv_pair_t*>, std::vector<int64_t>> ResultBuffers_t;
}
