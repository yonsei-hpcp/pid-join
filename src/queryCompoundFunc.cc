#include "join_internals.hpp"

#include "iostream"
#include <time.h>
#include "typeinfo"
#include <cmath>
#include <assert.h>

using namespace pidjoin;

void JoinOperator::Execute_COMPOUND_FUNC_RNS_JOIN(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node,
    std::vector<TimeStamp_t> *perThreadTimeStamps)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto key = (*op_node)["key"];
    auto type = (*op_node)["join_type"];
    auto debug_key = (*op_node)["debug_key"];
    auto packet_size = (*op_node)["packet_size"];

    int tuple_size_ht = sizeof(int64_t);
    int tuple_size_probe = sizeof(int64_t);

    

    if (this->join_instance->join_algorithm != "")
    {
        join_alg = this->join_instance->join_algorithm;
        
    }
    
    if (rank_id == 0)
        printf("join_alg: %s\n", join_alg.c_str());

    /**
     * Strings Setup
     */

    // HOST_FUNC_ROTATE_AND_STREAM
    std::string R_rnc_join_key("R_rnc_join_key");
    // HOST_FUNC_ROTATE_AND_STREAM
    std::string S_rnc_join_key("S_rnc_join_key");

    // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
    std::string R_packetwise_local_partitioned_join_key("R_packetwise_local_partitioned_join_key");
    std::string R_packetwise_local_partitioned_partition_info("R_packetwise_local_partitioned_partition_info");
    std::string R_packetwise_local_partition_num("R_packetwise_local_partition_num");

    // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
    std::string S_packetwise_local_partitioned_join_key("S_packetwise_local_partitioned_join_key");
    std::string S_packetwise_local_partitioned_partition_info("S_packetwise_local_partitioned_partition_info");

    // DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE
    std::string hash_table("hash_table");
    // DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE
    std::string join_result(outputs[0].asCString());

    Json::Value HOST_FUNC_ROTATE_AND_STREAM_var_r;
    Json::Value HOST_FUNC_ROTATE_AND_STREAM_var_s;

    HOST_FUNC_ROTATE_AND_STREAM_var_r["input"] = inputs[0].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_r["output"] = R_rnc_join_key;
    HOST_FUNC_ROTATE_AND_STREAM_var_r["histogram"] = inputs[1].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_r["packet_size"] = packet_size;

    HOST_FUNC_ROTATE_AND_STREAM_var_s["input"] = inputs[2].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_s["output"] = S_rnc_join_key;
    HOST_FUNC_ROTATE_AND_STREAM_var_s["histogram"] = inputs[3].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_s["packet_size"] = packet_size;

    perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "HOST_FUNC_ROTATE_AND_STREAM"});
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
    Execute_HOST_FUNC_ROTATE_AND_STREAM(idp_handler, rank_id, params, &HOST_FUNC_ROTATE_AND_STREAM_var_r);
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    // invalidate Nodes
    {
        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(inputs[0].asString());
        Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
    }
    perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "HOST_FUNC_ROTATE_AND_STREAM"});
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);

    Execute_HOST_FUNC_ROTATE_AND_STREAM(idp_handler, rank_id, params, &HOST_FUNC_ROTATE_AND_STREAM_var_s);

    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    // invalidate Nodes
    {
        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(inputs[2].asString());
        Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
    }

    // sync
    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

    if (join_alg == "phj")
    {
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs.append(R_rnc_join_key);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_partition_info);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(true);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs.append(S_rnc_join_key);
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_Tuple_ID);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(false);

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["tuple_size"] = tuple_size_ht;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["packet_size"] = packet_size;

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["tuple_size"] = tuple_size_probe;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["packet_size"] = packet_size;

        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs.append(hash_table);
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;

        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(hash_table);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs.append(join_result);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["join_type"] = type;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Local_partitioning"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r);
        Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }
        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Build"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_BUILD_HASH_TABLE(idp_handler, rank_id, params, &DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Probe"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_PROBE_HASH_TABLE_INNER(idp_handler, rank_id, params, &DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_partition_info);
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(hash_table);

        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;

        Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
    }
    else if (join_alg == "nl")
    {
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var;
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(R_rnc_join_key);
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(S_rnc_join_key);
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_outputs.append(outputs[0]);

        DPU_FUNC_NESTED_LOOP_JOIN_var["inputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["outputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["packet_size"] = packet_size;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "JOIN"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_NESTED_LOOP_JOIN(idp_handler, rank_id, params, &DPU_FUNC_NESTED_LOOP_JOIN_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }
    else if (join_alg == "smj")
    {
        auto algorithm = (*op_node)["algorithm"];
        // To be fixed
        algorithm[0] = "quick";
        algorithm[1] = 0;

        /* DPU_FUNC_MPSM_JOIN_SORT_R */

        Json::Value DPU_FUNC_MPSM_JOIN_SORT_R_var;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_R_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_R_var_outputs;

        // input : DPU_FUNC_MPSM_JOIN_SORT_R
        std::string R_sort_algorithm(algorithm[0].asString());
        DPU_FUNC_MPSM_JOIN_SORT_R_var_inputs.append(R_rnc_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_R_var_inputs.append(R_sort_algorithm);
        DPU_FUNC_MPSM_JOIN_SORT_R_var_inputs.append(algorithm[1].asBool());

        // output : DPU_FUNC_MPSM_JOIN_SORT_R
        std::string R_sorted_join_key("R_sorted_join_key");
        DPU_FUNC_MPSM_JOIN_SORT_R_var_outputs.append(R_sorted_join_key);

        DPU_FUNC_MPSM_JOIN_SORT_R_var["inputs"] = DPU_FUNC_MPSM_JOIN_SORT_R_var_inputs;
        DPU_FUNC_MPSM_JOIN_SORT_R_var["outputs"] = DPU_FUNC_MPSM_JOIN_SORT_R_var_outputs;
        DPU_FUNC_MPSM_JOIN_SORT_R_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_SORT_R
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_SORT_R"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_SORT(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_SORT_R_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        /* DPU_FUNC_MPSM_JOIN_SORT_R */

        /* DPU_FUNC_MPSM_JOIN_SORT_S */

        Json::Value DPU_FUNC_MPSM_JOIN_SORT_S_var;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_S_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_S_var_outputs;

        // input : DPU_FUNC_MPSM_JOIN_SORT_S
        std::string S_sort_algorithm(algorithm[0].asString());
        DPU_FUNC_MPSM_JOIN_SORT_S_var_inputs.append(S_rnc_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_S_var_inputs.append(S_sort_algorithm);
        DPU_FUNC_MPSM_JOIN_SORT_S_var_inputs.append(algorithm[1].asBool());

        // output : DPU_FUNC_MPSM_JOIN_SORT_S
        std::string S_sorted_join_key("S_sorted_join_key");
        DPU_FUNC_MPSM_JOIN_SORT_S_var_outputs.append(S_sorted_join_key);

        DPU_FUNC_MPSM_JOIN_SORT_S_var["inputs"] = DPU_FUNC_MPSM_JOIN_SORT_S_var_inputs;
        DPU_FUNC_MPSM_JOIN_SORT_S_var["outputs"] = DPU_FUNC_MPSM_JOIN_SORT_S_var_outputs;
        DPU_FUNC_MPSM_JOIN_SORT_S_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_SORT_S
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_SORT_S"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_SORT(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_SORT_S_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        /* DPU_FUNC_MPSM_JOIN_SORT_S */

        /* DPU_FUNC_MPSM_JOIN_PROBE */

        Json::Value DPU_FUNC_MPSM_JOIN_PROBE_ALL_var;
        Json::Value DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_output;

        // input : DPU_FUNC_MPSM_JOIN_PROBE
        DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_inputs.append(R_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_inputs.append(S_sorted_join_key);
        // output : DPU_FUNC_MPSM_JOIN_PROBE
        std::string join_result(outputs[0].asString());
        DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_output.append(join_result);

        DPU_FUNC_MPSM_JOIN_PROBE_ALL_var["inputs"] = DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_inputs;
        DPU_FUNC_MPSM_JOIN_PROBE_ALL_var["outputs"] = DPU_FUNC_MPSM_JOIN_PROBE_ALL_var_output;

        // Execute MPSM_JOIN_PROBE
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_PROBE"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_PROBE_ALL(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_PROBE_ALL_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }

    Json::Value DPU_FUNC_FINISH_JOIN_var;
    DPU_FUNC_FINISH_JOIN_var["join_output"] = outputs[0];
    perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "FINISH_J"});
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
    Execute_FINISH_JOIN(idp_handler, rank_id, params, &DPU_FUNC_FINISH_JOIN_var);
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
}

void JoinOperator::Execute_COMPOUND_FUNC_GLB_PARTITION(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    if (partition_type.asInt() == 0)
    {
        printf("%sError: Set partition_type\n", KRED);
        exit(-1);
    }

    std::string Tuple_ID = inputs[0].asCString();
    std::string Join_Key = inputs[1].asCString();
    ;

    // Execute_GLB_PARTITION_COUNT
    std::string gp_partition_info("gp_partition_info");
    std::string gp_histogram("gp_histogram");

    // Execute_GLB_PARTITION_PACKET
    std::string packetwise_global_partitioned_tid("packetwise_partitioned_tid");

    Json::Value DPU_FUNC_GLB_PARTITION_COUNT_var;
    DPU_FUNC_GLB_PARTITION_COUNT_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_partition_info);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_COUNT_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_COUNT_var["partition_type"] = (*op_node)["partition_type"];

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["inputs"].append(gp_histogram);
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["outputs"].append(outputs[1].asCString());
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["partition_type"] = partition_type;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["packet_size"] = packet_size;

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_GLB_PARTITION_PACKET_var;
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Tuple_ID);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(outputs[1].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_PACKET_var["outputs"].append(outputs[0].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_PACKET_var["partition_type"] = (*op_node)["partition_type"];

    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_partition_info);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_histogram);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);

    // Count the element to go each partitions(DPUs)
    Execute_GLB_PARTITION_COUNT(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_COUNT_var);
    // Count and calculate appropriate packet size and number
    Execute_MT_SAFE_HOST_FUNC_CALCULATE_PAGE_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var);
    Execute_GLB_PARTITION_PACKET(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_PACKET_var);
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
}
