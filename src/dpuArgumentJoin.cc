#include "idpHandler.hpp"


#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <thread>
#include <string.h>



namespace pidjoin
{
    ///////////////////////////////////////////////////////////////////////////////
    // IDPArgumentHandler
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////

    void *IDPArgumentHandler::ConfigureGlobalPartitioningArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int partition_num,
        const char *input_arr_name,
        const char *partitioned_input_arr_name,
        const char *partition_info_name,
        const char *histogram_name)
    {
        arg_rank.clear();
        hash_global_partitioning_arg *arg = (hash_global_partitioning_arg *)malloc(sizeof(hash_global_partitioning_arg) * NUM_DPU_RANK);

        StackNode *input_arr_node = idp_handler->FindNode(rank_id, input_arr_name);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Push partition info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(rank_id, partition_info_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push partitioned Payload result Node
        // Push partitioned tid result Node
        StackNode *partition_result_node = idp_handler->PushStackNodeAligned(
            rank_id,
            partitioned_input_arr_name,
            input_arr_node->data_bytes,
            input_arr_node->block_byte,
            8192);

        arg[0].table_r_start_byte = input_arr_node->start_byte;
        arg[0].partitioned_table_r_start_byte = partition_result_node->start_byte;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].table_r_num_elem = input_arr_node->data_bytes[0] / sizeof(tuplePair_t);
        
        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].table_r_num_elem = input_arr_node->data_bytes[i] / sizeof(tuplePair_t);
            
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigureGlobalPartitioningPrevArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int partition_num,
        const char *tid_buffer_name,
        const char *payload_buffer_name,
        const char *partitioned_tid_name,
        const char *partitioned_payload_name,
        const char *histogram_name)
    {
        arg_rank.clear();
        hash_global_partitioning_prev_arg *arg = (hash_global_partitioning_prev_arg *)malloc(sizeof(hash_global_partitioning_prev_arg) * NUM_DPU_RANK);

        StackNode *tid_buffer_node = idp_handler->FindNode(rank_id, tid_buffer_name);
        StackNode *payload_buffer_node = idp_handler->FindNode(rank_id, payload_buffer_name);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push partitioned payload result Node
        StackNode *partition_payload_result_node = idp_handler->PushStackNode(rank_id, partitioned_payload_name, payload_buffer_node->data_bytes, payload_buffer_node->block_byte);
        // Push partitioned tid result Node
        StackNode *partition_tid_result_node = idp_handler->PushStackNode(rank_id, partitioned_tid_name, tid_buffer_node->data_bytes, tid_buffer_node->block_byte);

        arg[0].tid_start_byte = tid_buffer_node->start_byte;
        arg[0].table_r_start_byte = payload_buffer_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partitioned_tid_start_byte = partition_tid_result_node->start_byte;
        arg[0].partitioned_r_start_byte = partition_payload_result_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].table_r_num_elem = payload_buffer_node->data_bytes[0] / sizeof(Key64_t);

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].table_r_num_elem = payload_buffer_node->data_bytes[i] / sizeof(Key64_t);

            arg_rank.push_back((char *)(arg + i));
        }
        return NULL;
    }

    void *IDPArgumentHandler::ConfigurePacketwiseGlobalPartitioningArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int partition_num,
        int packet_size,
        RankwiseMemoryBankBuffers_t *imm_hist_buffers,
        const char *packet_histogram_name,
        const char *partition_info_name,
        const char *tid_buffer_name,
        const char *payload_buffer_name,
        const char *aligned_tid_name,
        const char *aligned_payload_name)
    {
        arg_rank.clear();
        packetwise_hash_global_partitioning_arg *arg = (packetwise_hash_global_partitioning_arg *)malloc(sizeof(packetwise_hash_global_partitioning_arg) * NUM_DPU_RANK);

        auto &histogram_buff = imm_hist_buffers->at(rank_id);
        char *hist_ptr = histogram_buff.at(0);

        int total_packets = 0;

        for (int p = 0; p < (partition_num) / NUM_DPU_RANK; p++)
        {
            total_packets += ((int64_t *)hist_ptr)[p];
        }
        total_packets *= NUM_DPU_RANK;

        StackNode *histogram_node = idp_handler->FindNode(rank_id, packet_histogram_name);
        StackNode *partition_info_node = idp_handler->FindNode(rank_id, partition_info_name);

        StackNode *tid_buffer_node = idp_handler->FindNode(rank_id, tid_buffer_name);
        StackNode *payload_buffer_node = idp_handler->FindNode(rank_id, payload_buffer_name);

        // Push aligned Result Node
        StackNode *payload_result_node = idp_handler->PushStackNodeAligned(rank_id, aligned_payload_name, NULL, total_packets * packet_size, 8192);

        arg[0].dpu_id = 0;
        arg[0].rank_id = rank_id;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].tid_start_byte = tid_buffer_node->start_byte;
        arg[0].payload_start_byte = payload_buffer_node->start_byte;
        // arg[0].result_tid_start_byte = tid_result_node->start_byte;
        arg[0].result_payload_start_byte = payload_result_node->start_byte;
        arg[0].table_num_elem = tid_buffer_node->data_bytes[0] / sizeof(TupleID64_t);
        arg[0].partition_num = partition_num;

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].table_num_elem = tid_buffer_node->data_bytes[i] / sizeof(TupleID64_t);

            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureLocalPartitioningArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int total_rank,
        int *partition_nums,
        int shift_val,
        int tuple_size,
        const char *input_arr_name,
        const char *partitioned_arr_name,
        const char *partition_info_name)
    {
        arg_rank.clear();
        hash_local_partitioning_arg *arg =
            (hash_local_partitioning_arg *)malloc(
                sizeof(hash_local_partitioning_arg) * NUM_DPU_RANK);

        StackNode *input_arr_node = idp_handler->FindNode(rank_id, input_arr_name);

        // Add Data Node // NEED TO BE UPDATED AFTER LOCAL PARTITONING
        StackNode *partitioned_arr_node = idp_handler->PushStackNode(
            rank_id, partitioned_arr_name,
            input_arr_node->data_bytes, input_arr_node->block_byte);
        // Add Partition Info Node // NEED TO BE UPDATED AFTER LOCAL PARTITONING
        StackNode *partition_info_node = idp_handler->PushStackNode(rank_id, partition_info_name, NULL, -1);

        arg[0].input_arr_start_byte = input_arr_node->start_byte;
        arg[0].input_data_bytes = input_arr_node->data_bytes[0];

        // printf("arg[0].input_data_bytes: %d\n", arg[0].input_data_bytes);

        arg[0].partitioned_input_arr_start_byte = partitioned_arr_node->start_byte;
        arg[0].result_partition_info_start_byte = partition_info_node->start_byte;
        arg[0].do_calculate_partition_num = partition_nums[0];
        arg[0].shift_val = shift_val;
        arg[0].total_rank = total_rank;
        arg[0].tuple_size = tuple_size;

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            arg[i] = arg[0];
            arg[i].input_data_bytes = input_arr_node->data_bytes[i];
            arg[i].do_calculate_partition_num = partition_nums[i];
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigurePacketwiseLocalPartitioningArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int total_rank,
        int packet_size,
        int *partition_nums,
        int shift_value,
        int tuple_size,
        const char *rnc_packets_joinkey_name,
        const char *generated_tid_name,
        const char *partitioned_key_name,
        const char *partition_info_name)
    {
        arg_rank.clear();
        packetwise_hash_local_partitioning_arg *arg = (packetwise_hash_local_partitioning_arg *)malloc(sizeof(packetwise_hash_local_partitioning_arg) * NUM_DPU_RANK);

        StackNode *rnc_packets_joinkey_node = idp_handler->FindNode(rank_id, rnc_packets_joinkey_name);

        // Add Data Node // NEED TO BE UPDATED AFTER LOCAL PARTITONING
        StackNode *partitioned_key_node = idp_handler->PushStackNode(
            rank_id, partitioned_key_name, rnc_packets_joinkey_node->data_bytes, rnc_packets_joinkey_node->block_byte);
        // Add Partition Info Node // NEED TO BE UPDATED AFTER LOCAL PARTITONING
        StackNode *partition_info_node = idp_handler->PushStackNode(rank_id, partition_info_name, NULL, -1);

        arg[0].total_rank = total_rank;
        arg[0].num_packets = (rnc_packets_joinkey_node->block_byte / packet_size);
        arg[0].packet_size = packet_size;
        arg[0].packet_start_byte = rnc_packets_joinkey_node->start_byte;
        // arg[0].generated_local_tid_start_byte = generated_tid_node->start_byte;
        arg[0].partitioned_result_start_byte = partitioned_key_node->start_byte;
        arg[0].result_partition_info_start_byte = partition_info_node->start_byte;
        arg[0].shift_val = shift_value;
        arg[0].tuple_size = tuple_size;
        // 0 means do partitioning
        arg[0].do_calculate_partition_num = partition_nums[0];

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].do_calculate_partition_num = partition_nums[i];
            // printf("partition_nums: %d\n", partition_nums[i]);
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigurePHJHashTableBuildHorizontalArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *r_partitioned_payload_name,
        const char *r_partitioned_tid_name, // Deprecated
        const char *r_partitioned_info_name,
        const char *hash_table_name)
    {
        arg_rank.clear();
        hash_phj_build_arg *arg = (hash_phj_build_arg *)malloc(sizeof(hash_phj_build_arg) * NUM_DPU_RANK);

        StackNode *r_partitioned_payload_node = idp_handler->FindNode(rank_id, r_partitioned_payload_name);
        // StackNode *r_partitioned_tid_node = idp_handler->FindNode(rank_id,r_partitioned_tid_name);
        StackNode *r_partitioned_info_node = idp_handler->FindNode(rank_id, r_partitioned_info_name);

        int partition_sizes[NUM_DPU_RANK];
        int max_partition_size = 0;

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            // printf("r_partitioned_info_node->data_bytes[%d]: %d\n", i, r_partitioned_info_node->data_bytes[i]);
            partition_sizes[i] = (r_partitioned_info_node->data_bytes[i] / sizeof(int32_t)) * 32768;
            // partition_sizes[i] = (r_partitioned_info_node->data_bytes[i] / sizeof(int32_t)) * 2048;

            if (max_partition_size < partition_sizes[i])
            {
                max_partition_size = partition_sizes[i];
            }
        }

        // Add Hash Table Node
        StackNode *hash_table_node = idp_handler->PushStackNode(rank_id, hash_table_name, partition_sizes, max_partition_size);

        arg[0].parted_R_offset = r_partitioned_payload_node->start_byte;
        // arg[0].parted_Tid_R_offset = r_partitioned_tid_node->start_byte;
        arg[0].HT_offset = hash_table_node->start_byte;
        arg[0].parted_R_info_offset = r_partitioned_info_node->start_byte;
        arg[0].partition_num = (r_partitioned_info_node->data_bytes[0] / sizeof(int32_t));
        arg[0].R_num = r_partitioned_payload_node->data_bytes[0] / sizeof(Key64_t);
        // Add arg
        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].partition_num = (r_partitioned_info_node->data_bytes[i] / sizeof(int32_t));
            arg[i].R_num = r_partitioned_payload_node->data_bytes[i] / sizeof(Key64_t);
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigurePHJHashTableProbeInnerHorizontalArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int join_type,
        const char *hash_table_name,
        const char *partitioned_s_arr_name,
        const char *s_partition_info_name,
        const char *result_tids_name)
    {
        arg_rank.clear();
        hash_phj_probe_arg *arg = (hash_phj_probe_arg *)malloc(sizeof(hash_phj_probe_arg) * NUM_DPU_RANK);

        StackNode *hash_table_node = idp_handler->FindNode(rank_id, hash_table_name);
        StackNode *partitioned_s_arr_node = idp_handler->FindNode(rank_id, partitioned_s_arr_name);
        StackNode *s_partition_info_node = idp_handler->FindNode(rank_id, s_partition_info_name);

        int div_val;
        int mul_val;

        // FIXME
        if (join_type == JOIN_TYPE_EQUI)
        {
            div_val = sizeof(tuplePair_t);
            mul_val = sizeof(tuplePair_t);
        }
        else
        {
            printf("%sError: Not Avaliable Join Type\n", KRED);
            exit(-1);
        }

        int data_bytes[NUM_DPU_RANK];
        int block_bytes;

        for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
        {
            data_bytes[dpu] = partitioned_s_arr_node->data_bytes[dpu] * mul_val / div_val;
        }
        block_bytes = partitioned_s_arr_node->block_byte * mul_val / div_val;

        // Add Result Node
        StackNode *result_tids_node = idp_handler->PushStackNodeAligned(
            rank_id,
            result_tids_name,
            data_bytes,
            block_bytes, 8192);

        arg[0].parted_S_offset = partitioned_s_arr_node->start_byte;
        arg[0].HT_offset = hash_table_node->start_byte;
        arg[0].parted_S_info_offset = s_partition_info_node->start_byte;
        arg[0].Result_offset = result_tids_node->start_byte;
        arg[0].partition_num = s_partition_info_node->data_bytes[0] / sizeof(int32_t);
        arg[0].S_num = partitioned_s_arr_node->data_bytes[0] / div_val;
        arg[0].rank_id = rank_id;
        arg[0].key_table_type = join_type;
        arg[0].dpu_id = 0;
        arg[0].probe_type = 0;

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].partition_num = s_partition_info_node->data_bytes[i] / sizeof(int32_t);
            arg[i].S_num = partitioned_s_arr_node->data_bytes[i] / div_val;
            arg[i].dpu_id = i;
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigurePHJHashTableProbeHorizontalArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *hash_table_name,
        const char *partitioned_s_payload_name,
        const char *partitioned_s_tid_name,
        const char *s_partition_info_name,
        const char *result_tids_name)
    {
        arg_rank.clear();
        hash_phj_probe_arg *arg = (hash_phj_probe_arg *)malloc(sizeof(hash_phj_probe_arg) * NUM_DPU_RANK);

        StackNode *hash_table_node = idp_handler->FindNode(rank_id, hash_table_name);
        StackNode *partitioned_s_tid_node = idp_handler->FindNode(rank_id, partitioned_s_tid_name);
        StackNode *partitioned_s_payload_node = idp_handler->FindNode(rank_id, partitioned_s_payload_name);
        StackNode *s_partition_info_node = idp_handler->FindNode(rank_id, s_partition_info_name);

        // Add Result Node
        StackNode *result_tids_node = idp_handler->PushStackNode(rank_id, result_tids_name, partitioned_s_tid_node->data_bytes, partitioned_s_payload_node->block_byte);

        arg[0].parted_S_offset = partitioned_s_payload_node->start_byte;
        arg[0].parted_Tid_S_offset = partitioned_s_tid_node->start_byte;
        arg[0].HT_offset = hash_table_node->start_byte;
        arg[0].parted_S_info_offset = s_partition_info_node->start_byte;
        arg[0].Result_offset = result_tids_node->start_byte;
        arg[0].partition_num = s_partition_info_node->data_bytes[0] / sizeof(int32_t);
        arg[0].S_num = partitioned_s_tid_node->data_bytes[0] / sizeof(TupleID64_t);
        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].partition_num = s_partition_info_node->data_bytes[i] / sizeof(int32_t);
            arg[i].S_num = partitioned_s_tid_node->data_bytes[i] / sizeof(TupleID64_t);

            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        if (rank_id == 1)
        {
            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                printf("arg[%d].S_num:%d\n", i, arg[i].S_num);
            }
        }

        return NULL;
    }

    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureNPHJBuildHorizontalArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *hash_table_name,
        const char *R_name)
    {

        return NULL;
    }


    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureNPHJProbeHorizontalArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *hash_table_name,
        const char *S_name,
        const char *Result_name)
    {
        return NULL;
    }


    /////////////////////////////////////////////////////////////////////
    // Sort Merge (MPSM) Join
    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureSortMergeJoinPartitioningArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *rnc_packets_joinkey_name,
        const char *sorted_joinkey_name,
        const char *histogram_addr_name)
    {
        arg_rank.clear();
        sort_merge_partitioning_arg *arg = (sort_merge_partitioning_arg *)malloc(sizeof(sort_merge_partitioning_arg) * NUM_DPU_RANK);

        StackNode *rnc_packets_joinkey_node = idp_handler->FindNode(rank_id, rnc_packets_joinkey_name);

        int data_bytes[NUM_DPU_RANK];
        int block_byte = rnc_packets_joinkey_node->block_byte / 1024;
        for (int i = 0; i < NUM_DPU_RANK; i++)
            data_bytes[i] = rnc_packets_joinkey_node->data_bytes[i] / 1024;

        // Push Histogram Address Node
        StackNode *sorted_joinkey_node = idp_handler->PushStackNodeAligned(rank_id, sorted_joinkey_name,
                                                                       rnc_packets_joinkey_node->data_bytes, rnc_packets_joinkey_node->block_byte, 8192);
        StackNode *histogram_addr_node = idp_handler->PushStackNodeAligned(rank_id, histogram_addr_name, data_bytes, block_byte, 8192);

        arg[0].num_packets = rnc_packets_joinkey_node->block_byte / RNS_PAGE_SIZE_128B;
        arg[0].r_packet_start_byte = rnc_packets_joinkey_node->start_byte;
        arg[0].r_sorted_start_byte = sorted_joinkey_node->start_byte;
        arg[0].histogram_addr_start_byte = histogram_addr_node->start_byte;

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureSortMergeJoinProbeArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *rnc_packets_joinkey_name,
        const char *partitioned_joinkey_name,
        const char *histogram_addr_name,
        const char *sorted_joinkey_name,
        const char *result_tids_name)
    {
        arg_rank.clear();
        sort_merge_probe_arg *arg = (sort_merge_probe_arg *)malloc(sizeof(sort_merge_probe_arg) * NUM_DPU_RANK);

        StackNode *rnc_packets_joinkey_node = idp_handler->FindNode(rank_id, rnc_packets_joinkey_name);
        StackNode *partitioned_joinkey_node = idp_handler->FindNode(rank_id, partitioned_joinkey_name);
        StackNode *histogram_addr_node = idp_handler->FindNode(rank_id, histogram_addr_name);

        StackNode *sorted_joinkey_node = idp_handler->PushStackNodeAligned(rank_id, sorted_joinkey_name,
                                                                       rnc_packets_joinkey_node->data_bytes, rnc_packets_joinkey_node->block_byte, 8192);
        StackNode *result_node = idp_handler->PushStackNodeAligned(rank_id, result_tids_name,
                                                               rnc_packets_joinkey_node->data_bytes, rnc_packets_joinkey_node->block_byte, 8192);

        arg[0].num_packets = rnc_packets_joinkey_node->block_byte / RNS_PAGE_SIZE_128B;
        arg[0].s_packet_start_byte = rnc_packets_joinkey_node->start_byte;
        arg[0].s_sorted_start_byte = sorted_joinkey_node->start_byte;
        arg[0].r_partitioned_start_byte = partitioned_joinkey_node->start_byte;
        arg[0].result_probe_start_byte = result_node->start_byte;
        arg[0].histogram_addrs = histogram_addr_node->start_byte;
        arg[0].r_total_bytes = partitioned_joinkey_node->data_bytes[0];

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].r_total_bytes = partitioned_joinkey_node->data_bytes[i];
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureSortMergeJoinSortArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        const char *rnc_packets_joinkey_name,
        const char *sorted_joinkey_name,
        bool data_skewness)
    {
        arg_rank.clear();
        sort_merge_partitioning_arg *arg = (sort_merge_partitioning_arg *)malloc(sizeof(sort_merge_partitioning_arg) * NUM_DPU_RANK);

        StackNode *rnc_packets_joinkey_node = idp_handler->FindNode(rank_id, rnc_packets_joinkey_name);

        int data_bytes[NUM_DPU_RANK];
        int block_bytes;

        for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++) data_bytes[dpu] = sizeof(uint32_t);
        block_bytes = sizeof(uint32_t);

        // Push sorted joinkey Address Node
        StackNode *sorted_joinkey_node = idp_handler->PushStackNodeAligned(rank_id, sorted_joinkey_name,
            rnc_packets_joinkey_node->data_bytes, rnc_packets_joinkey_node->block_byte, 8192);

        arg[0].r_packet_start_byte = rnc_packets_joinkey_node->start_byte;
        arg[0].r_sorted_start_byte = sorted_joinkey_node->start_byte;
        arg[0].num_packets = rnc_packets_joinkey_node->data_bytes[0] / packet_size;
        arg[0].data_skewness = data_skewness;
        arg[0].packet_size = packet_size;

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].num_packets = rnc_packets_joinkey_node->data_bytes[i] / packet_size;
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }


    /////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler::ConfigureSortMergeJoinProbeAllArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char *r_sorted_joinkey_name,
        const char *s_sorted_joinkey_name,
        const char *result_tids_name)
    {
        arg_rank.clear();
        sort_merge_probe_all_arg *arg = (sort_merge_probe_all_arg *)malloc(sizeof(sort_merge_probe_all_arg) * NUM_DPU_RANK);

        StackNode *r_sorted_joinkey_node = idp_handler->FindNode(rank_id, r_sorted_joinkey_name);
        StackNode *s_sorted_joinkey_node = idp_handler->FindNode(rank_id, s_sorted_joinkey_name);

        StackNode *result_node = idp_handler->PushStackNodeAligned(rank_id, result_tids_name,
                s_sorted_joinkey_node->data_bytes, s_sorted_joinkey_node->block_byte, 8192);

        arg[0].r_sorted_start_byte = r_sorted_joinkey_node->start_byte;
        arg[0].s_sorted_start_byte = s_sorted_joinkey_node->start_byte;
        arg[0].result_probe_start_byte = result_node->start_byte;
        arg[0].r_total_bytes = r_sorted_joinkey_node->data_bytes[0];
        arg[0].s_total_bytes = s_sorted_joinkey_node->data_bytes[0];

        arg_rank.push_back((char *)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].r_total_bytes = r_sorted_joinkey_node->data_bytes[i];
            arg[i].s_total_bytes = s_sorted_joinkey_node->data_bytes[i];
            // Add arg
            arg_rank.push_back((char *)(arg + i));
        }

        return NULL;
    }

}
