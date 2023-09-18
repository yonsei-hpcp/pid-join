#include "join_internals.hpp"

#include "iostream"
#include <time.h>
#include "typeinfo"

using namespace pidjoin;

static void HandleDeprecated(const char *func_name)
{
    printf("%sError: %s is Deprecated.\n", KCYN, func_name);
    exit(-1);
}

void JoinOperator::Execute_GLOBAL_HASH_PARTITIONING(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_GLOBAL_HASH_PARTITIONING);

    IDPArgumentHandler::ConfigureGlobalPartitioningArgRank(
        rank_id, idp_handler, *params,
        NUM_DPU_RANK * join_instance->num_rank_allocated,
        inputs[0].asCString(),
        outputs[0].asCString(),
        outputs[1].asCString(),
        outputs[2].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_GLOBAL_HASH_PARTITIONING);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_GLOBAL_HASH_PARTITIONING, *params);
}

void JoinOperator::Execute_PHJ_BUILD_HASH_TABLE(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE);

    IDPArgumentHandler::ConfigurePHJHashTableBuildHorizontalArgRank(
        rank_id, idp_handler, *params,
        inputs[0].asCString(),
        NULL,
        inputs[1].asCString(),
        outputs[0].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE);
    idp_handler->RunKernel(rank_id);

    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, *params);

    // CHAEMIN
    // idp_handler->ReadLog(rank_id);
}

void JoinOperator::Execute_PHJ_PROBE_HASH_TABLE_INNER(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    struct timespec timer_start[4];
    struct timespec timer_stop[4];

    // clock_gettime(CLOCK_MONOTONIC, timer_start + 0);

    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto type = (*op_node)["join_type"];

    int join_type = JOIN_TYPE_EQUI;
    std::string join_type_as_string = type.asString();

    idp_handler->LoadProgram(rank_id, DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE);

    IDPArgumentHandler::ConfigurePHJHashTableProbeInnerHorizontalArgRank(
        rank_id, idp_handler, *params,
        join_type,
        inputs[0].asCString(),
        inputs[1].asCString(),
        inputs[2].asCString(),
        outputs[0].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE);
    idp_handler->RunKernel(rank_id);

    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, *params);

    // idp_handler->ReadLog(rank_id);
}

void JoinOperator::Execute_NESTED_LOOP_JOIN(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = (*op_node)["packet_size"].asInt();

    int num_ranks = this->join_instance->num_rank_allocated;

    // Local Partition P
    idp_handler->LoadProgram(rank_id, DPU_FUNC_NESTED_LOOP_JOIN);

    IDPArgumentHandler::ConfigureNestedLoopJoinArgRank(
        rank_id, idp_handler, *params,
        num_ranks,
        packet_size,
        inputs[0].asCString(),
        inputs[1].asCString(),
        outputs[0].asCString());

    idp_handler->PrintStackInfoBrief(rank_id);

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_NESTED_LOOP_JOIN);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_NESTED_LOOP_JOIN, *params);
}

void JoinOperator::Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto setup = (*op_node)["setup"];
    auto tuple_size = (*op_node)["tuple_size"];
    auto packet_size = this->join_instance->packet_size;

    int num_ranks = this->join_instance->num_rank_allocated;

    int32_t *local_partition_nums;

    GlobalBuffer_t *tbuff = this->join_instance->GetOrAllocateGlobalBuffer(
        NUM_DPU_RANK * sizeof(int32_t),
        (std::to_string(rank_id) + setup[0].asString()).c_str());

    local_partition_nums = (int32_t *)tbuff->aligned_data;

    if (setup[1].asBool() == true)
    {
        memset(local_partition_nums, 0, NUM_DPU_RANK * sizeof(int32_t));
    }
    // Local Partition P
    idp_handler->LoadProgram(rank_id, DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING);

    int targetlevel = 0;
    int num_dpus = num_ranks * NUM_DPU_RANK;
    while (num_dpus >>= 1)
        ++targetlevel;

    IDPArgumentHandler::ConfigurePacketwiseLocalPartitioningArgRank(
        rank_id, idp_handler, *params,
        num_ranks,
        packet_size,
        local_partition_nums,
        targetlevel,
        tuple_size.asInt(),
        inputs[0].asCString(),
        NULL,
        outputs[0].asCString(),
        outputs[1].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING);
    idp_handler->RunKernel(rank_id);

    if (setup[1].asBool() == false)
    {
        local_partition_nums = NULL;
    }

    idp_handler->HandleKernelResults(
        this->join_instance,
        rank_id,
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING,
        *params,
        (char *)local_partition_nums);

    // idp_handler->ReadLog(rank_id);
}

void JoinOperator::Execute_GLB_PARTITION_COUNT(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_GLB_PARTITION_COUNT);

    IDPArgumentHandler::ConfigureGlobalPartitioningCountArgRank(
        rank_id, idp_handler, *params,
        packet_size,
        this->join_instance->num_rank_allocated * NUM_DPU_RANK,
        partition_type.asInt(),
        inputs[0].asCString(),
        outputs[0].asCString(),
        outputs[1].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_GLB_PARTITION_COUNT);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_GLB_PARTITION_COUNT, *params);
}

void JoinOperator::Execute_GLB_PARTITION_PACKET(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_GLB_PARTITION_PACKET);

    // printf("inputs[2].asCString(): %s\n", inputs[2].asCString());

    if (inputs.size() == 3)
    {
        IDPArgumentHandler::ConfigureGlobalPartitioningPacketArgRank(
            rank_id, idp_handler, *params,
            packet_size,
            this->join_instance->num_rank_allocated * NUM_DPU_RANK,
            partition_type.asInt(),
            this->join_instance->GetMemoryBankBuffers(inputs[2].asCString())->first,
            inputs[0].asCString(),
            inputs[1].asCString(),
            inputs[2].asCString(),
            NULL,
            outputs[0].asCString());
    }
    else if (inputs.size() == 4)
    {
        IDPArgumentHandler::ConfigureGlobalPartitioningPacketArgRank(
            rank_id, idp_handler, *params,
            packet_size,
            this->join_instance->num_rank_allocated * NUM_DPU_RANK,
            partition_type.asInt(),
            this->join_instance->GetMemoryBankBuffers(inputs[2].asCString())->first,
            inputs[0].asCString(),
            inputs[1].asCString(),
            inputs[2].asCString(),
            inputs[3].asCString(),
            outputs[0].asCString());
    }
    else
    {
        printf("%sError!: inputs.size() : %d\n", KRED, inputs.size());
        exit(-1);
    }

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_GLB_PARTITION_PACKET);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_GLB_PARTITION_PACKET, *params);

    // CHAEMIN
    // idp_handler->ReadLog(rank_id);
}

void JoinOperator::Execute_MPSM_JOIN_PARTITIONING(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_PARTITIONING);

    sort_merge_partitioning_arg arg_form;

    IDPArgumentHandler::ConfigureSortMergeJoinPartitioningArgRank(
        rank_id, idp_handler, *params,
        inputs[0].asCString(),
        outputs[0].asCString(),
        outputs[1].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_PARTITIONING);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_PARTITIONING, *params);
}

void JoinOperator::Execute_MPSM_JOIN_PROBE(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_PROBE);

    sort_merge_probe_arg arg_form;

    IDPArgumentHandler::ConfigureSortMergeJoinProbeArgRank(
        rank_id, idp_handler, *params,
        inputs[0].asCString(),
        inputs[1].asCString(),
        inputs[2].asCString(),
        outputs[0].asCString(),
        outputs[1].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_PROBE);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_PROBE, *params);
}

void JoinOperator::Execute_MPSM_JOIN_SORT(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = (*op_node)["packet_size"].asInt();

    sort_merge_partitioning_arg arg_form;

    if (inputs[1].asString().compare("radix") == 0)
        idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_RADIX_SORT);
    else if (inputs[1].asString().compare("merge") == 0)
        idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_MERGE_SORT);
    else 
        idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_QUICK_SORT);

    IDPArgumentHandler::ConfigureSortMergeJoinSortArgRank(
        rank_id, idp_handler, *params,
        packet_size,
        inputs[0].asCString(),
        outputs[0].asCString(),
        inputs[2].asBool());

    if (inputs[1].asString().compare("radix") == 0)
    {
        idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_RADIX_SORT);
        idp_handler->RunKernel(rank_id);
        idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_RADIX_SORT, *params);
    }
    else if (inputs[1].asString().compare("merge") == 0)
    {
        idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_MERGE_SORT);
        idp_handler->RunKernel(rank_id);
        idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_MERGE_SORT, *params);
    }
    else
    {
        idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_QUICK_SORT);
        idp_handler->RunKernel(rank_id);
        idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_QUICK_SORT, *params);
    }

    // idp_handler->ReadLog(0);
}

void JoinOperator::Execute_MPSM_JOIN_PROBE_ALL(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_MPSM_JOIN_PROBE_ALL);

    sort_merge_probe_arg arg_form;

    IDPArgumentHandler::ConfigureSortMergeJoinProbeAllArgRank(
        rank_id, idp_handler, *params,
        inputs[0].asCString(),
        inputs[1].asCString(),
        outputs[0].asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_MPSM_JOIN_PROBE_ALL);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_MPSM_JOIN_PROBE_ALL, *params);

    // idp_handler->ReadLog(0);
}


void JoinOperator::Execute_FINISH_JOIN(IDPHandler*idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto join_output = (*op_node)["join_output"];

    idp_handler->LoadProgram(rank_id, DPU_FUNC_FINISH_JOIN);

    IDPArgumentHandler::ConfigureFinishJoinArgRank(
        rank_id, idp_handler, *params, join_output.asCString());

    idp_handler->LoadParameter(rank_id, *params, DPU_FUNC_FINISH_JOIN);
    idp_handler->RunKernel(rank_id);
    idp_handler->HandleKernelResults(this->join_instance, rank_id, DPU_FUNC_FINISH_JOIN, *params);
}
