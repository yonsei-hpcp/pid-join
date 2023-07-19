#include "join_internals.hpp"


#include <algorithm>
#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>


namespace pidjoin
{
    std::unordered_map<std::string, int> operator_name_map =
    {
        // DPU Functions
        {"DPU_FUNC_LOCAL_HASH_PARTITIONING", DPU_FUNC_LOCAL_HASH_PARTITIONING},
        {"DPU_FUNC_GLOBAL_HASH_PARTITIONING", DPU_FUNC_GLOBAL_HASH_PARTITIONING},
        {"DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE", DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE},
        {"DPU_FUNC_PHJ_PROBE_HASH_TABLE", DPU_FUNC_PHJ_PROBE_HASH_TABLE},
        {"DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING", DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING},
        {"DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING", DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING},
        {"DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE", DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE},
        {"DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE", DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE},
        {"DPU_FUNC_NPHJ_BUILD_HASH_TABLE", DPU_FUNC_NPHJ_BUILD_HASH_TABLE},
        {"DPU_FUNC_NPHJ_PROBE_HASH_TABLE", DPU_FUNC_NPHJ_PROBE_HASH_TABLE},
	    {"DPU_FUNC_GLB_PARTITION_COUNT", DPU_FUNC_GLB_PARTITION_COUNT},
        {"DPU_FUNC_GLB_PARTITION_PACKET", DPU_FUNC_GLB_PARTITION_PACKET},
        {"DPU_FUNC_MPSM_JOIN_PARTITIONING", DPU_FUNC_MPSM_JOIN_PARTITIONING},
        {"DPU_FUNC_MPSM_JOIN_PROBE", DPU_FUNC_MPSM_JOIN_PROBE},
        {"DPU_FUNC_NESTED_LOOP_JOIN", DPU_FUNC_NESTED_LOOP_JOIN},
        {"DPU_FUNC_MPSM_JOIN_SORT", DPU_FUNC_MPSM_JOIN_SORT},
        {"DPU_FUNC_MPSM_JOIN_MERGE_SORT", DPU_FUNC_MPSM_JOIN_MERGE_SORT},
        {"DPU_FUNC_MPSM_JOIN_RADIX_SORT", DPU_FUNC_MPSM_JOIN_RADIX_SORT},
        {"DPU_FUNC_MPSM_JOIN_QUICK_SORT", DPU_FUNC_MPSM_JOIN_QUICK_SORT},
        {"DPU_FUNC_MPSM_JOIN_PROBE_ALL", DPU_FUNC_MPSM_JOIN_PROBE_ALL},
        {"DPU_FUNC_FINISH_JOIN", DPU_FUNC_FINISH_JOIN},
        // Host Functions
        {"HOST_FUNC_INVALIDATE_STACKNODE", HOST_FUNC_INVALIDATE_STACKNODE}, 
        {"HOST_FUNC_ROTATE_AND_STREAM", HOST_FUNC_ROTATE_AND_STREAM},
        {"HOST_FUNC_ROTATE_AND_STREAM_OPT", HOST_FUNC_ROTATE_AND_STREAM_OPT},
        {"HOST_FUNC_CALCULATE_PAGE_HISTOGRAM", HOST_FUNC_CALCULATE_PAGE_HISTOGRAM},
        {"HOST_FUNC_LOAD_COLUMN", HOST_FUNC_LOAD_COLUMN},
        {"HOST_FUNC_RECV_DATA_UPMEMLIB", HOST_FUNC_RECV_DATA_UPMEMLIB},
        {"HOST_FUNC_SEND_DATA_UPMEMLIB", HOST_FUNC_SEND_DATA_UPMEMLIB},
        {"HOST_FUNC_SEND_DATA_OPT", HOST_FUNC_SEND_DATA_OPT},
        {"HOST_FUNC_RECV_DATA_OPT", HOST_FUNC_RECV_DATA_OPT},
        // Compound Functions
        {"COMPOUND_FUNC_RNS_JOIN", COMPOUND_FUNC_RNS_JOIN},
        {"COMPOUND_FUNC_GLB_PARTITION", COMPOUND_FUNC_GLB_PARTITION},
        {"COMPOUND_FUNC_GLB_PARTITION_CPU", COMPOUND_FUNC_GLB_PARTITION_CPU},
        {"COMPOUND_FUNC_UPMEM_JOIN", COMPOUND_FUNC_UPMEM_JOIN},
        {"COMPOUND_FUNC_SORT_MERGE_JOIN", COMPOUND_FUNC_SORT_MERGE_JOIN},
        {"COMPOUND_FUNC_SORT_MERGE_JOIN_ALL", COMPOUND_FUNC_SORT_MERGE_JOIN_ALL},
        // Tester Functions
        {"TEST_FUNC_VALIDATE_PACKETWISE_DATA", TEST_FUNC_VALIDATE_PACKETWISE_DATA},
        {"TEST_FUNC_VALIDATE_LOCAL_PARTITIONED_DATA", TEST_FUNC_VALIDATE_LOCAL_PARTITIONED_DATA},
        {"TEST_FUNC_ROTATE_AND_CACHE_BYPASS", TEST_FUNC_ROTATE_AND_CACHE_BYPASS},
        {"TEST_FUNC_PIM2HOST", TEST_FUNC_PIM2HOST},
        {"TEST_FUNC_HOST2PIM", TEST_FUNC_HOST2PIM},
        // Control functions
        {"CONTROL_FUNC_SYNC_THREADS", CONTROL_FUNC_SYNC_THREADS},
        // Microbenchmark
        {"MICROBENCHMARK_MRAM_BANDWIDTH", MICROBENCHMARK_MRAM_BANDWIDTH},
        {"MICROBENCHMARK_WRAM_BANDWIDTH", MICROBENCHMARK_WRAM_BANDWIDTH},
        {"MICROBENCHMARK_OPS", MICROBENCHMARK_OPS},
        {"MICROBENCHMARK_HASH", MICROBENCHMARK_HASH},
    };

    std::unordered_map<int, std::string> param_binary_name_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, "src/dpu/bin/a2a_srj_local_partition"},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, "src/dpu/bin/a2a_srj_global_partition"},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, "src/dpu/bin/a2a_srj_build_linear_probe"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, "src/dpu/bin/a2a_srj_probe"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "src/dpu/bin/a2a_srj_probe_inner_mt_linear_probe"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "src/dpu/bin/a2a_packetwise_srj_local_partition"},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "src/dpu/bin/a2a_packetwise_srj_global_partition"},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, "src/dpu/bin/a2a_packetwise_srj_build"},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, "src/dpu/bin/no_partitioned_hash_join_build"},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, "src/dpu/bin/no_partitioned_hash_join_probe"},
	    {DPU_FUNC_GLB_PARTITION_COUNT, "src/dpu/bin/glb_partition_count"},
        {DPU_FUNC_MPSM_JOIN_PARTITIONING, "src/dpu/bin/mpsm_join_partition"},
        {DPU_FUNC_MPSM_JOIN_PROBE, "src/dpu/bin/mpsm_join_probe"},
        {DPU_FUNC_GLB_PARTITION_PACKET, "src/dpu/bin/glb_partition_packet"},
        {MICROBENCHMARK_MRAM_BANDWIDTH, "src/dpu/bin/microbenchmark_mram_bandwidth"},
        {MICROBENCHMARK_WRAM_BANDWIDTH, "src/dpu/bin/microbenchmark_wram_bandwidth"},
        {MICROBENCHMARK_OPS, "src/dpu/bin/microbenchmark_ops"},
        {MICROBENCHMARK_HASH, "src/dpu/bin/microbenchmark_hash"},
        {DPU_FUNC_NESTED_LOOP_JOIN, "src/dpu/bin/nested_loop_join"},
        {DPU_FUNC_MPSM_JOIN_SORT, "src/dpu/bin/mpsm_join_sort"},
        {DPU_FUNC_MPSM_JOIN_RADIX_SORT, "src/dpu/bin/mpsm_join_radix_sort"},
        {DPU_FUNC_MPSM_JOIN_MERGE_SORT, "src/dpu/bin/mpsm_join_merge_sort"},
        {DPU_FUNC_MPSM_JOIN_QUICK_SORT, "src/dpu/bin/mpsm_join_quick_sort"},
        {DPU_FUNC_MPSM_JOIN_PROBE_ALL, "src/dpu/bin/mpsm_join_probe_all"},
        {DPU_FUNC_FINISH_JOIN, "src/dpu/bin/finish_join"},
    };

    std::unordered_map<int, std::string> param_name_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, "param_local_hash_partitioning"},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, "param_global_hash_partitioning"},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, "param_phj_build_hash_table"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, "param_phj_probe_hash_table"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "param_phj_probe_hash_table_inner"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "param_packetwise_local_hash_partitioning"},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "param_packetwise_global_hash_partitioning"},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, "param_packetwise_phj_build_hash_table"},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, "param_no_partitioned_join_build_hash_table"},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, "param_no_partitioned_join_probe_hash_table"},
	    {DPU_FUNC_GLB_PARTITION_COUNT, "param_glb_partition_count"},
        {DPU_FUNC_GLB_PARTITION_PACKET, "param_glb_partition_packet"},
        {MICROBENCHMARK_MRAM_BANDWIDTH, "param_microbenchmark"},
        {MICROBENCHMARK_WRAM_BANDWIDTH, "param_microbenchmark"},
        {MICROBENCHMARK_OPS, "param_microbenchmark"},
        {MICROBENCHMARK_HASH, "param_microbenchmark"},
        {DPU_FUNC_MPSM_JOIN_PARTITIONING, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_PROBE, "param_sort_merge_probe"},
        {DPU_FUNC_NESTED_LOOP_JOIN, "param_nested_loop_join"},
        {DPU_FUNC_MPSM_JOIN_SORT, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_RADIX_SORT, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_MERGE_SORT, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_QUICK_SORT, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_PROBE_ALL, "param_sort_merge_probe_all"},
        {DPU_FUNC_FINISH_JOIN, "param_finish_join"},
    };

    std::unordered_map<int, int> param_size_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, sizeof(hash_local_partitioning_arg)},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, sizeof(hash_global_partitioning_arg)},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, sizeof(hash_phj_build_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, sizeof(hash_phj_probe_arg)},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, sizeof(packetwise_hash_local_partitioning_arg)},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, sizeof(packetwise_hash_global_partitioning_arg)},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, sizeof(packetwise_hash_phj_build_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, sizeof(hash_phj_probe_arg)},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, sizeof(no_partitioned_join_build_hash_table_arg)},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, sizeof(no_partitioned_join_build_hash_table_arg)},
        {DPU_FUNC_GLB_PARTITION_COUNT, sizeof(glb_partition_count_arg)},
        {DPU_FUNC_FINISH_JOIN,sizeof(finish_join_arg)},
        {DPU_FUNC_GLB_PARTITION_PACKET, sizeof(glb_partition_packet_arg)},
        {MICROBENCHMARK_MRAM_BANDWIDTH, sizeof(microbenchmark_arg)},
        {MICROBENCHMARK_WRAM_BANDWIDTH, sizeof(microbenchmark_arg)},
        {MICROBENCHMARK_OPS, sizeof(microbenchmark_arg)},
        {MICROBENCHMARK_HASH, sizeof(microbenchmark_arg)},
        {DPU_FUNC_MPSM_JOIN_PARTITIONING, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_PROBE, sizeof(sort_merge_probe_arg)},
        {DPU_FUNC_NESTED_LOOP_JOIN, sizeof(nested_loop_join_arg)},
        {DPU_FUNC_MPSM_JOIN_SORT, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_RADIX_SORT, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_MERGE_SORT, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_QUICK_SORT, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_PROBE_ALL, sizeof(sort_merge_probe_all_arg)},
    };

    std::unordered_map<int, std::string> param_return_var_map = 
    {
	{DPU_FUNC_LOCAL_HASH_PARTITIONING, "param_local_hash_partitioning_return"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "param_packetwise_local_hash_partitioning_return"},
	    {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "param_packetwise_global_hash_partitioning_return"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "param_hash_phj_probe_return"},
        {MICROBENCHMARK_MRAM_BANDWIDTH, "param_microbenchmark_return"},
        {MICROBENCHMARK_WRAM_BANDWIDTH, "param_microbenchmark_return"},
        {MICROBENCHMARK_OPS, "param_microbenchmark_return"},
        {MICROBENCHMARK_HASH, "param_microbenchmark_return"},
        {DPU_FUNC_MPSM_JOIN_PARTITIONING, "param_sort_merge_partitioning_return"},
        {DPU_FUNC_MPSM_JOIN_PROBE, "param_sort_merge_probe_return"},
    };
    
    std::unordered_map<int, int> param_return_size_map = 
    {
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, sizeof(packetwise_hash_local_partitioning_return_arg)},
	    {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, sizeof(packetwise_hash_global_partitioning_return_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, sizeof(hash_phj_probe_return_arg)},
        {MICROBENCHMARK_MRAM_BANDWIDTH, sizeof(microbenchmark_return_arg)},
        {MICROBENCHMARK_WRAM_BANDWIDTH, sizeof(microbenchmark_return_arg)},
        {MICROBENCHMARK_OPS, sizeof(microbenchmark_return_arg)},
        {MICROBENCHMARK_HASH, sizeof(microbenchmark_return_arg)},
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, sizeof(hash_local_partitioning_return_arg)},
        {DPU_FUNC_MPSM_JOIN_PARTITIONING, sizeof(sort_merge_partitioning_return_arg)},
        {DPU_FUNC_MPSM_JOIN_PROBE, sizeof(sort_merge_probe_return_arg)},
    };
}
