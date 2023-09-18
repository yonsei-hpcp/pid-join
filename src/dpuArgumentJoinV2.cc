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
    void *IDPArgumentHandler::ConfigureGlobalPartitioningPacketArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        RankwiseMemoryBankBuffers_t * imm_hist_buffers,
        const char *input_buffer_name1, // tid
        const char *local_histogram_name,
        const char *packet_histogram_name,
        const char *input_buffer_name2, // Payload
        const char *result_buffer_name)
    {
        arg_rank.clear();
        glb_partition_packet_arg *arg = (glb_partition_packet_arg *)malloc(sizeof(glb_partition_packet_arg) * NUM_DPU_RANK);

        int div_val = 8;
        int packet_div = 1;

        partition_type = 1;

        StackNode *input_buffer_node1;

        if (partition_type == 1)
        {
            input_buffer_node1 = idp_handler->FindNode(rank_id, input_buffer_name2);
        }
        else
        {
            input_buffer_node1 = idp_handler->FindNode(rank_id, input_buffer_name1);
        }

        StackNode *local_histogram_node = idp_handler->FindNode(rank_id, local_histogram_name);
        StackNode *packet_histogram_node = idp_handler->FindNode(rank_id, packet_histogram_name);
        StackNode *input_buffer_node2 = NULL;

        auto& histogram_buff = imm_hist_buffers->at(rank_id);
        char* hist_ptr = histogram_buff.at(0);

        int total_packets = 0;

        for (int p = 0; p < (partition_num) / NUM_DPU_RANK; p++)
        {
            total_packets += ((int64_t*)hist_ptr)[p];
            // printf("((int64_t*)hist_ptr)[%d]: %d\n", p, ((int64_t*)hist_ptr)[p]);
        }
        total_packets *= NUM_DPU_RANK;
        // printf("total_packets: %d\n", total_packets);
        StackNode *result_buffer_node = idp_handler->PushStackNodeAligned(
            rank_id, result_buffer_name, NULL, total_packets * packet_size, 8192);

        arg[0].dpu_id = 0;
        arg[0].rank_id = rank_id;
        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset1 = input_buffer_node1->start_byte;
        arg[0].elem_num = input_buffer_node1->data_bytes[0] / div_val;
        arg[0].histogram_start_byte = local_histogram_node->start_byte;
        arg[0].packet_histogram_start_byte = packet_histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].result_offset = result_buffer_node->start_byte;
        arg[0].packet_size = packet_size;
        // printf("PARTITION: arg[%d].elem_num: %d\n", 0, arg[0].elem_num);
        
        // printf("arg_rank: size: %d\n", arg_rank.size());
        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].elem_num = input_buffer_node1->data_bytes[i] / div_val;
            arg[i].elem_num = input_buffer_node1->data_bytes[0] / div_val;
            // printf("PARTITION: arg[%d].elem_num: %d\n", i, arg[i].elem_num);
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }
    
    void *IDPArgumentHandler::ConfigureGlobalPartitioningCountArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        const char *input_buffer_name,
        const char *partition_info_name,
        const char *histogram_name)
    {
        partition_type = 1;

        arg_rank.clear();
        glb_partition_count_arg *arg = (glb_partition_count_arg *)malloc(sizeof(glb_partition_count_arg) * NUM_DPU_RANK);

        int div_val = 8;

        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Push partition info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(
            rank_id, partition_info_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(
            rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));

        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / div_val;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        
        // printf("COUNT: arg[%d].elem_num: %d\n", 0, arg[0].elem_num);

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            // arg[i].elem_num = input_buffer_node->data_bytes[i] / div_val;
            arg[i].elem_num = input_buffer_node->data_bytes[0] / div_val;
            // printf("COUNT: arg[%d].elem_num: %d\n", i, arg[i].elem_num);
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureDebuggerArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int total_rank,
        int debugger_op_type,
        const char *target_addr1,
        const char *target_addr2)
    {
        arg_rank.clear();
        debugger_arg *arg = (debugger_arg *)malloc(sizeof(debugger_arg) * NUM_DPU_RANK);

        StackNode *target_addr1_node;
        StackNode *target_addr2_node;

        if (target_addr1 != NULL)
        {
            target_addr1_node = idp_handler->FindNode(rank_id, target_addr1);
        }

        if (target_addr2 != NULL)
        {
            target_addr2_node = idp_handler->FindNode(rank_id, target_addr2);
        }

        arg[0].debugger_op = debugger_op_type;
        arg[0].check_target_addr1 = target_addr1_node->start_byte;
        arg[0].check_target_elem_num1 = target_addr1_node->data_bytes[0] / sizeof(int64_t);
        arg[0].check_target_addr2 = target_addr2_node->start_byte;
        arg[0].check_target_elem_num2 = target_addr2_node->data_bytes[0] / sizeof(int64_t);
        arg[0].rank_id = rank_id;
        arg[0].dpu_id = 0;
        arg[0].total_rank = total_rank;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].check_target_elem_num1 = target_addr1_node->data_bytes[i] / sizeof(int64_t);
            arg[i].check_target_elem_num2 = target_addr2_node->data_bytes[i] / sizeof(int64_t);
            
            arg_rank.push_back((char*)(arg + i));
            
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigureNestedLoopJoinArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int packet_size,
            const char *R_packets_name,
            const char *S_packets_name,
            const char *result_node_name)
    {
        arg_rank.clear();
        nested_loop_join_arg* arg = (nested_loop_join_arg *)malloc(sizeof(nested_loop_join_arg) * NUM_DPU_RANK);

        StackNode *R_stacknode = idp_handler->FindNode(rank_id, R_packets_name);
        StackNode *S_stacknode = idp_handler->FindNode(rank_id, S_packets_name);

        printf("R_packets_name: %s\n", R_packets_name);
        printf("S_packets_name: %s\n", S_packets_name);
        printf("packet_size: %d\n", packet_size);

        StackNode *result_node = idp_handler->PushStackNodeAligned(
            rank_id, result_node_name, S_stacknode->data_bytes, S_stacknode->block_byte, 8192);
        
        // exit(-1);
        arg[0].packet_size = packet_size;
        arg[0].R_num_packets = R_stacknode->data_bytes[0] / packet_size;
        printf("arg[0].R_num_packets: %d\n", arg[0].R_num_packets);
        arg[0].R_packet_start_byte = R_stacknode->start_byte;
        arg[0].result_start_byte = result_node->start_byte;
        arg[0].S_num_packets = S_stacknode->data_bytes[0] / packet_size;
        printf("arg[0].S_num_packets: %d\n", arg[0].S_num_packets);
        printf("S Elem: %d\n", arg[0].S_num_packets * packet_size / 8);
        arg[0].S_packet_start_byte = S_stacknode->start_byte;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].R_num_packets = R_stacknode->data_bytes[i] / packet_size;
            arg[i].S_num_packets = S_stacknode->data_bytes[i] / packet_size;
            
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void* IDPArgumentHandler::ConfigureFinishJoinArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char * join_result_name)
    {
        arg_rank.clear();
        finish_join_arg* arg = (finish_join_arg *)malloc(sizeof(finish_join_arg) * NUM_DPU_RANK);

        StackNode *join_result_node = idp_handler->FindNode(rank_id, join_result_name);

        int64_t leftover_bytes = 0;

        arg[0].effective_bytes = join_result_node->data_bytes[0];
        arg[0].join_result_start_byte = join_result_node->start_byte;
        arg[0].max_bytes = join_result_node->block_byte;
        
        leftover_bytes += (arg[0].max_bytes - arg[0].effective_bytes);
        
        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].effective_bytes = join_result_node->data_bytes[i];
            
            leftover_bytes += (arg[i].max_bytes - arg[i].effective_bytes);
            arg_rank.push_back((char*)(arg + i));
        }

        // printf("leftover_bytes: %ld ELEM: %ld\n", leftover_bytes, leftover_bytes / sizeof(tuplePair_t));
        return NULL;
    }

}
