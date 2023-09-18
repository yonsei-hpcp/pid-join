#include "join_internals.hpp"

#include "iostream"
#include <time.h>
#include "typeinfo"
#include <mutex>
#include <atomic>
#include <algorithm>
#include <math.h>

#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <inttypes.h>

#ifdef INTEL_ITTNOTIFY_API
#include <ittnotify.h>
#endif

using namespace pidjoin;

static void CalculatePageHistogram(
    JoinInstance *join_instance,
    IDPHandler &idp_handler,
    int rank_id, int total_rank, int partition_num, // rank_id is probably source rank id - Comment
    RankwiseMemoryBankBufferPair_t &input_histogram_buffers,
    RankwiseMemoryBankBufferPair_t &output_dpu_xfer_packet_buffers,
    int data_type_size, int packet_size);

/* Append m_timeStamps of JoinInstance */
void JoinInstance::CollectLog(std::vector<TimeStamp_t> &timeStamps)
{
    pthread_mutex_lock(&this->timeline_mutex);
    m_timeStamps.insert(m_timeStamps.end(), timeStamps.begin(), timeStamps.end());
    pthread_mutex_unlock(&this->timeline_mutex);
}

void JoinInstance::RecordLogs()
{
    if (this->timeline_ptr == NULL)
    {
        for (int i = 0; i < m_timeStamps.size(); i++)
        {
            float time_start = (m_timeStamps[i].start_time.tv_sec * 1000.0) - (m_timer_init.tv_sec * 1000.0) + (m_timeStamps[i].start_time.tv_nsec / 1000000.0) - (m_timer_init.tv_nsec / 1000000.0);
            float time_stop = (m_timeStamps[i].end_time.tv_sec * 1000.0) - (m_timer_init.tv_sec * 1000.0) + (m_timeStamps[i].end_time.tv_nsec / 1000000.0) - (m_timer_init.tv_nsec / 1000000.0);
            std::string time_stamp_line = std::to_string(m_timeStamps[i].rank_id) + "-" + std::to_string(i) + "-" + std::to_string(time_start) + "-" + std::to_string(time_stop) + "-" + std::string(m_timeStamps[i].node_name) + "-" + m_timeStamps[i].inputs + "\n";
            printf("%s", time_stamp_line.c_str());
        }

        int64_t total_miss_count = 0;
        int64_t total_num_tuples = 0;
        int64_t total_xfer_byte = 0;
        int64_t total_xfer_packet_num = 0;

        std::map<std::string, int64_t> rnc_xfer_bytes;
        std::map<std::string, int64_t> rnc_num_packets;

        // Print Stat Logs
        for (int rank_id = 0; rank_id < this->num_rank_allocated; rank_id++)
        {
            for (auto &log : this->rankwiseStatLogs[rank_id])
            {
                if (log.second.STAT_TYPE == STAT_TYPE_JOIN)
                {
                    total_miss_count += log.second.miss_count;
                    total_num_tuples += log.second.num_tuples;
                }
                else if (log.second.STAT_TYPE == STAT_TYPE_RNS)
                {
                    total_xfer_byte += log.second.xfer_byte;
                    total_xfer_packet_num += log.second.xfer_packet_num;

                    if (rnc_xfer_bytes.find(log.first) != rnc_xfer_bytes.end())
                    {
                        rnc_xfer_bytes.insert(std::make_pair(log.first, 0));
                    }
                    if (rnc_num_packets.find(log.first) != rnc_num_packets.end())
                    {
                        rnc_num_packets.insert(std::make_pair(log.first, 0));
                    }

                    rnc_xfer_bytes[log.first] += (log.second.xfer_byte);
                    rnc_num_packets[log.first] += (log.second.xfer_packet_num);
                }
            }
        }

        for (auto &rnc_num_packet : rnc_num_packets)
        {
            fprintf(stdout, "rnc_xfer_bytes %ld rnc_num_packets %ld\n", rnc_xfer_bytes[rnc_num_packet.first], rnc_num_packet.second);
        }

        // fprintf(stdout, "MISSRATE %f\n", (total_miss_count * 1.0) / (total_num_tuples * 1.0 + total_miss_count * 1.0));
        // fprintf(stdout, "FILL FACTOR %f\n", (total_xfer_byte * 1.0) / (total_xfer_packet_num * 128.0));
    }
    else
    {
        for (int i = 0; i < m_timeStamps.size(); i++)
        {
            float time_start = (m_timeStamps[i].start_time.tv_sec * 1000.0) - (m_timer_init.tv_sec * 1000.0) + (m_timeStamps[i].start_time.tv_nsec / 1000000.0) - (m_timer_init.tv_nsec / 1000000.0);
            float time_stop = (m_timeStamps[i].end_time.tv_sec * 1000.0) - (m_timer_init.tv_sec * 1000.0) + (m_timeStamps[i].end_time.tv_nsec / 1000000.0) - (m_timer_init.tv_nsec / 1000000.0);
            std::string time_stamp_line = std::to_string(m_timeStamps[i].rank_id) + "-" + std::to_string(i) + "-" + std::to_string(time_start) + "-" + std::to_string(time_stop) + "-" + std::string(m_timeStamps[i].node_name) + "-" + m_timeStamps[i].inputs + "\n";
            fprintf(this->timeline_ptr, "%s", time_stamp_line.c_str());
        }

        int64_t total_miss_count = 0;
        int64_t total_num_tuples = 0;
        int64_t total_xfer_byte = 0;
        int64_t total_xfer_packet_num = 0;

        std::map<std::string, int64_t> rnc_xfer_bytes;
        std::map<std::string, int64_t> rnc_num_packets;

        // Print Stat Logs
        for (int rank_id = 0; rank_id < this->num_rank_allocated; rank_id++)
        {
            int RNS_index = 0;
            for (auto &log : this->rankwiseStatLogs[rank_id])
            {
                // printf("%s: ", log.first.c_str());

                if (log.second.STAT_TYPE == STAT_TYPE_JOIN)
                {
                    total_miss_count += log.second.miss_count;
                    total_num_tuples += log.second.num_tuples;
                    // printf("hit_count %d ", log.second.hit_count);
                    // printf("miss_count %d ", log.second.miss_count);
                    // printf("num_tuples %d ", log.second.num_tuples);
                    // printf("\n");
                }
                else if (log.second.STAT_TYPE == STAT_TYPE_RNS)
                {
                    total_xfer_byte += log.second.xfer_byte;
                    total_xfer_packet_num += log.second.xfer_packet_num;

                    if (rnc_xfer_bytes.find(log.first) != rnc_xfer_bytes.end())
                    {
                        rnc_xfer_bytes.insert(std::make_pair(log.first, 0));
                    }
                    if (rnc_num_packets.find(log.first) != rnc_num_packets.end())
                    {
                        rnc_num_packets.insert(std::make_pair(log.first, 0));
                    }

                    rnc_xfer_bytes[log.first] += (log.second.xfer_byte);
                    rnc_num_packets[log.first] += (log.second.xfer_packet_num);
                    fprintf(this->timeline_ptr, "[%dth-RNS] TX of Rank %d (packet#, byte) : %ld , %ld | Max TX Byte of DPU : %ld | UPMEM TX byte : %ld\n",
                            RNS_index, rank_id, log.second.xfer_packet_num * log.second.packet_size, log.second.xfer_byte, log.second.max_xfer_byte, log.second.upmem_tx_byte);
                    RNS_index += 1;
                }
            }
        }

        for (auto &rnc_num_packet : rnc_num_packets)
        {
            fprintf(this->timeline_ptr, "%s rnc_xfer_bytes %ld rnc_num_packets %ld\n", rnc_num_packet.first.c_str(), rnc_xfer_bytes[rnc_num_packet.first], rnc_num_packet.second);
        }

        // printf("miss_count %d ", total_miss_count);
        // printf("num_tuples %d ", total_num_tuples);
        // printf("\n");
        fprintf(this->timeline_ptr, "MISSRATE %f\n", (total_miss_count * 1.0) / (total_num_tuples * 1.0));
        fprintf(this->timeline_ptr, "FILL FACTOR %f\n", (total_xfer_byte * 1.0) / (total_xfer_packet_num * 128.0));
    }
}

void JoinInstance::AddJobIntoPriorityQueue(rotate_n_stream_job_t *new_job)
{
    RNS_Job_Queue_t *rnc_job_queue = ((IDPHandler *)(new_job->idp_handler))->rnc_job_q;

    int ret = pthread_mutex_lock(&(rnc_job_queue->mutex));
    if (ret != 0)
    {
        printf("%s %d Error pthread Lock Error. %s:%d\n", KRED, ret, __FILE__, __LINE__);
        exit(-1);
    }

    ret = pthread_mutex_lock(&rotate_and_stream_mutex);
    if (ret != 0)
    {
        printf("%s %d Error pthread Lock Error. %s:%d\n", KRED, ret, __FILE__, __LINE__);
        exit(-1);
    }

    ((IDPHandler *)(new_job->idp_handler))->AddRNSJob(new_job, false);

    pthread_mutex_unlock(&rotate_and_stream_mutex);
    pthread_mutex_unlock(&(rnc_job_queue->mutex));
}

static int UpdateSharedHistogramFixedPriority(GlobalBuffer_t *hist_buff, int64_t value, int num_ranks, int my_rank_id)
{
    int mram_dst_offset_packet = 0;

    pthread_mutex_lock(&(hist_buff->lock));

    int64_t *target_accuml_dat = (int64_t *)(hist_buff->aligned_data);

    bool overflow = true;

    // if previous rank does not set the data, then, sleep.
    if (my_rank_id != 0)
    {
        while (true)
        {
            if (target_accuml_dat[my_rank_id] == 0)
            {
                // Sleep
                pthread_cond_wait(&(hist_buff->cond), &(hist_buff->lock));
            }
            else
            {
                break;
            }
        }
    }
    else
    {
        target_accuml_dat[0] = 0;
    }

    // Update histogram
    target_accuml_dat[my_rank_id + 1] = target_accuml_dat[my_rank_id] + value;
    mram_dst_offset_packet = target_accuml_dat[my_rank_id];

    pthread_cond_broadcast(&(hist_buff->cond));
    pthread_mutex_unlock(&(hist_buff->lock));

    return mram_dst_offset_packet;
}

enum job_priority_func_opt_t
{
    RANK_FIRST,
    CH_LOAD_BAL,
    JOB_FIRST,
    DEFAULT
};

int job_priority_func(IDPHandler *idp_handler, int src_rank, int target_rank, job_priority_func_opt_t option)
{
    int src_ch = idp_handler->GetChannelID(src_rank);
    int tar_ch = idp_handler->GetChannelID(target_rank);
    int priority; // smaller value first priority
#ifndef RNS_NEW_QUE
    if (option == DEFAULT)
    {
        priority = 0;
    }
    else if (option == RANK_FIRST)
    {
        // execute smaller channel ID first. higher priority if src_ch == tar_ch
        int smaller_ch = src_ch < tar_ch ? src_ch : tar_ch;
        priority = smaller_ch * 2;
        priority += src_ch != tar_ch ? 1 : 0;
    }
    else if (option == CH_LOAD_BAL)
    {
        if (src_ch == tar_ch)
            priority = 0;
        else if ((src_ch == 9 && tar_ch == 10) || (src_ch == 11 && tar_ch == 12))
            priority = 1;
        else if ((src_ch == 10 && tar_ch == 9) || (src_ch == 12 && tar_ch == 11))
            priority = 1;
        else if ((src_ch == 9 && tar_ch == 12) || (src_ch == 10 && tar_ch == 11))
            priority = 2;
        else if ((src_ch == 12 && tar_ch == 9) || (src_ch == 11 && tar_ch == 10))
            priority = 2;
        else if ((src_ch == 9 && tar_ch == 11) || (src_ch == 10 && tar_ch == 12))
            priority = 3;
        else if ((src_ch == 11 && tar_ch == 9) || (src_ch == 12 && tar_ch == 10))
            priority = 3;
        else
        {
            // hard coded number of channel = 4, range 9 ~ 12
            printf("%sError! Unexpected range of channel ID. currently only supports 9~12 \n", KCYN);
            exit(-1);
        }
    }
#else
    if (option == JOB_FIRST)
    {
        // execute different channels first. or same job first
        // Different priority queue handles this
        priority = 0;
    }
#endif
    else
    {
        printf("%sError! Undefiend priority option\n", KCYN);
        exit(-1);
    }

    return priority;
}

/**
 * @brief
 *
 * @param idp_handler
 * @param rank_id
 * @param params
 * @param op_node
 */
static int ITER__ = 0;
static int synchro_cnt = 0;

void JoinOperator::Execute_HOST_FUNC_ROTATE_AND_STREAM(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    // printf("Execute_HOST_FUNC_ROTATE_AND_STREAM Started Rank: %d\n", rank_id);
    int num_ranks = join_instance->num_rank_allocated;

    idp_handler->SetRNSRank(rank_id);

    sem_wait(&this->join_instance->rotate_and_stream_semaphore);

    pthread_mutex_lock(&(this->join_instance->thread_queue_mutex));

    if (idp_handler->rnc_job_q->ROTATE_AND_STREAM_JOBS == 0)
    {
        idp_handler->rnc_job_q->ROTATE_AND_STREAM_JOBS = num_ranks * num_ranks;
        ITER__++;
    }

    pthread_mutex_unlock(&(this->join_instance->thread_queue_mutex));
    join_instance->tot_jobs_per_rank = num_ranks * 2;

    // Load Vars
    auto histogram = (*op_node)["histogram"];
    auto input = (*op_node)["input"];
    auto output = (*op_node)["output"];
    int packet_size = this->join_instance->packet_size;

    int job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_128B;

    if (packet_size == 128)
        job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_128B;
    else if (packet_size == 64)
        job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_64B;
    else if (packet_size == 32)
        job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_32B;
    else if (packet_size == 16)
        job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_16B;
    else if (packet_size == 8)
        job_type = DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_8B;
    else
    {
        printf("%sError! Not appropriate packet size.\n", KRED);
        exit(-1);
    }

    // Find Node
    StackNode *input_node = idp_handler->FindNode(rank_id, input.asCString());
    // std::cout << rank_id << " Allocate Node output.asCString(): " << output.asCString() << std::endl;
    StackNode *output_node = idp_handler->PushStackNodeAligned(rank_id, output.asCString(), NULL, -1, 8192);

    ////////////////////////////////////////////////////////////
    // Find Offset
    int64_t src_base_offset = input_node->start_byte;

    RankwiseMemoryBankBufferPair_t *rankwise_mbank_buff_histogram_pair = join_instance->GetMemoryBankBuffers(histogram.asCString());

    if (rankwise_mbank_buff_histogram_pair == NULL)
    {
        printf("histogram.asCString(): %s\n", histogram.asCString());
        exit(-1);
    }
    std::vector<char *> &rankwise_mbank_buff_histogram_buff = rankwise_mbank_buff_histogram_pair->first->at(rank_id);

    //////////////////////////////////
    // Setting up histograms
    //////////////////////////////////
    int64_t *packet_nums_hist = (int64_t *)(rankwise_mbank_buff_histogram_buff[0]);
    int64_t *local_temp_packet_nums_accuml;
    int64_t packet_nums_total = 0;

    GlobalBuffer_t *other_ranks_local_temp_packet_accmuls[num_ranks];
    GlobalBuffer_t *other_ranks_packet_accmuls[num_ranks];
    GlobalBuffer_t *other_ranks_buffers[num_ranks];
    StackNode *output_nodes[num_ranks];

    output_nodes[rank_id] = output_node;

    ////////////////////////////////////////////////////////////////////////////

    int64_t packet_offset_size = (int64_t)NUM_DPU_RANK * num_ranks * NUM_DPU_RANK * packet_size;

    for (int r = 0; r < num_ranks; r++)
    {
        // setup as null
        other_ranks_buffers[r] = NULL;

        other_ranks_local_temp_packet_accmuls[r] = join_instance->GetOrAllocateGlobalBuffer(
            sizeof(int64_t) * (num_ranks + 1),
            (std::to_string(r) + "_local_temp_accumls" + std::to_string(ITER__)).c_str(),
            true);

        if (r == rank_id)
        {
            pthread_mutex_lock(&other_ranks_local_temp_packet_accmuls[r]->lock);
            local_temp_packet_nums_accuml = (int64_t *)(other_ranks_local_temp_packet_accmuls[r]->aligned_data);
            local_temp_packet_nums_accuml[0] = 0;

            for (int rr = 1; rr < (num_ranks + 1); rr++)
            {
                local_temp_packet_nums_accuml[rr] = local_temp_packet_nums_accuml[rr - 1] + packet_nums_hist[rr - 1];
            }
            packet_nums_total = local_temp_packet_nums_accuml[num_ranks];

            pthread_mutex_unlock(&other_ranks_local_temp_packet_accmuls[r]->lock);
        }

        // Last Element is for calculating the total Page Number
        other_ranks_packet_accmuls[r] = join_instance->GetOrAllocateGlobalBuffer(
            sizeof(int64_t) * (num_ranks + 1),
            (std::to_string(r) + "_target_accumls" + std::to_string(ITER__)).c_str(),
            true);
    }

    // Write Into Target Ranks,
    bool ranks_processed[num_ranks] = {
        false,
    };

    lock_ *lock = new lock_();
    int my_order = join_instance->ReadyQueueLineFixedPriority(rank_id, lock);

    ////////////////////////////////////////////////////////////////////////////
    // Job Creation
    ////////////////////////////////////////////////////////////////////////////

    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

    for (int target_rank = 0; target_rank < num_ranks; target_rank++)
    {
        rotate_n_stream_job_t *job1 = new rotate_n_stream_job_t;

        int priority = job_priority_func(idp_handler, rank_id, target_rank, DEFAULT);

        // if (target_rank == rank_id)
        // {
        //     // Allocate My array
        //     if (other_ranks_buffers[rank_id] == NULL)
        //     {
        //         other_ranks_buffers[rank_id] = join_instance->GetOrAllocateGlobalBuffer(
        //             1024,
        //             (std::to_string(rank_id) + "_host_temp_buffer" + std::to_string(ITER__)).c_str());
        //     }
        // }

        // //////////////////////////////////////////////////////////////////////////
        // // Add Job Type 1 (DPU -> DPU)
        // //////////////////////////////////////////////////////////////////////////
        // // Wait
        // other_ranks_buffers[target_rank] = join_instance->WaitForGlobalBuffer(
        //     (std::to_string(target_rank) + "_host_temp_buffer" + std::to_string(ITER__)).c_str());

        // Find output Node
        // std::cout << "Try Find output.asCString(): " << output.asCString() << std::endl;
        output_nodes[target_rank] = idp_handler->FindNode(target_rank, output.asCString());

        int mram_dst_offset_packet = UpdateSharedHistogramFixedPriority(
            other_ranks_packet_accmuls[target_rank],
            packet_nums_hist[target_rank],
            num_ranks,
            rank_id);

        // printf("Packet Num to copy: %ld\n", packet_nums_hist[target_rank]);

        JoinInstance::InitRnSJob(
            job1, priority,
            rank_id, target_rank,
            src_base_offset + (local_temp_packet_nums_accuml[target_rank] * NUM_DPU_RANK * packet_size),   // mram src offset
            output_nodes[target_rank]->start_byte + (mram_dst_offset_packet * NUM_DPU_RANK * packet_size), // mram dst offset
            job_type, 0,
            packet_nums_hist[target_rank], // src_packet_num
            NULL,                          // Not Used for job 1
            idp_handler);

        join_instance->AddJobIntoPriorityQueue(job1);

        ranks_processed[target_rank] = true;
    }

    GlobalBuffer_t *my_accmuls = other_ranks_packet_accmuls[rank_id];
    int output_node_data_byte = this->join_instance->WaitJobDoneQueueLine(my_order, my_accmuls, num_ranks, packet_size);
    idp_handler->UpdateStackNode(rank_id, output_node, output_node_data_byte);
}

std::vector<std::string> split(std::string input, char delimiter)
{
    std::vector<std::string> answer;
    std::stringstream ss(input);
    std::string temp;

    while (std::getline(ss, temp, delimiter))
    {
        answer.push_back(temp);
    }

    return answer;
}

std::vector<int> parse_target_rank(int src_ids, std::string src_dst_pairs, int *to_receive)
{
    std::vector<std::string> pairs = split(src_dst_pairs, '/');
    std::vector<int> target_ranks;

    for (std::string str_pair : pairs)
    {
        std::vector<std::string> vec_pair = split(str_pair, '_');
        if (std::stoi(vec_pair[0]) == src_ids)
            target_ranks.push_back(std::stoi(vec_pair[1]));
        if (std::stoi(vec_pair[1]) == src_ids)
            *to_receive += 1;
    }
    return target_ranks;
}

static int UpdateSharedHistogram(GlobalBuffer_t *hist_buff, int64_t value, int num_ranks)
{
    int mram_dst_offset_packet = 0;

    pthread_mutex_lock(&(hist_buff->lock));

    int64_t *target_accuml_dat = (int64_t *)(hist_buff->aligned_data);

    bool overflow = true;
    for (int rr = 1; rr < num_ranks + 1; rr++)
    {
        if (target_accuml_dat[rr] == 0)
        {
            target_accuml_dat[rr] = target_accuml_dat[rr - 1] + value;
            mram_dst_offset_packet = target_accuml_dat[rr - 1];
            overflow = false;
            break;
        }
    }

    if (overflow)
    {
        printf("%sERROR: Histogram overflow\n", KRED);
        printf("HISTOGRAM\n");

        for (int rr = 0; rr < num_ranks; rr++)
        {
            printf("%ld, ", target_accuml_dat[rr]);
        }
        printf("\n");
        exit(-1);
    }

    pthread_mutex_unlock(&(hist_buff->lock));

    return mram_dst_offset_packet;
}

static int ITER = 0;

void JoinInstance::ResetQueueLine(void)
{
    // pthread_mutex_lock(&(this->thread_queue_mutex));

    auto pair = std::pair<int, lock_ *>(-1, NULL);
    std::fill(this->thread_queue_line.begin(), this->thread_queue_line.end(), pair);
    this->curr_queued = 0;

    rns_rank_thread_orders.clear();
    job_done_map.clear();

    for (int i = 0; i < this->num_rank_allocated; i++)
    {
        // RemoveGlobalBuffer((std::to_string(i)+"_host_temp_buffer" + std::to_string(ITER)).c_str());
        // RemoveGlobalBuffer((std::to_string(i)+ "_local_temp_accumls" + std::to_string(ITER)).c_str());
        // RemoveGlobalBuffer((std::to_string(i)+ "_target_accumls" + std::to_string(ITER)).c_str());
    }

    for (int i = 0; i < this->num_rank_allocated; i++)
    {
        sem_post(&this->rotate_and_stream_semaphore);
    }

    // pthread_mutex_unlock(&(this->thread_queue_mutex));
}

void JoinOperator::Execute_HOST_FUNC_INVALIDATE_STACKNODE(IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];

    for (int i = 0; i < inputs.size(); i++)
    {
        const char *input = inputs[i].asCString();

        idp_handler->RemoveStackNode(rank_id, input);
    }
}

void JoinOperator::Execute_MT_SAFE_HOST_FUNC_CALCULATE_PAGE_HISTOGRAM(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{

    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto partition_type = (*op_node)["partition_type"];
    auto packet_size = this->join_instance->packet_size;

    int datatype_size = 8;

    partition_type == 1;

    // Page Calculation Setup
    int total_rank = this->join_instance->num_rank_allocated;
    int partition_num = total_rank * NUM_DPU_RANK;
    int result_histogram_size_byte = total_rank * sizeof(int64_t);

    RankwiseMemoryBankBufferPair_t *rankwise_mbank_buff_pair_histogram = join_instance->AllocateEmptyMemoryBankBuffersRankwise(rank_id, inputs[0].asCString());

    idp_handler->RetrieveData(
        rank_id,
        (*rankwise_mbank_buff_pair_histogram->first)[rank_id],
        (*rankwise_mbank_buff_pair_histogram->second)[rank_id],
        inputs[0].asCString());

    // Allocate Histogram
    RankwiseMemoryBankBufferPair_t *rankwise_mbank_buff_result_histogram = join_instance->AllocateMemoryBankBuffersRankwise(
        result_histogram_size_byte, rank_id, outputs[0].asCString());

    CalculatePageHistogram(
        this->join_instance,
        *idp_handler, rank_id, total_rank, partition_num,
        *rankwise_mbank_buff_pair_histogram,
        *rankwise_mbank_buff_result_histogram,
        datatype_size,
        packet_size);

    int64_t *buff = (int64_t *)(*rankwise_mbank_buff_result_histogram->first)[rank_id].at(0);

    // Push Result Node
    idp_handler->StoreDataAligned(
        rank_id,
        (*rankwise_mbank_buff_result_histogram->first)[rank_id],
        (*rankwise_mbank_buff_result_histogram->second)[rank_id],
        result_histogram_size_byte,
        outputs[0].asCString(), 8192);
}

void JoinOperator::Execute_HOST_FUNC_LOAD_COLUMN(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    // this->join_instance->GetPartitionedColumnBuffersGenTup(
    //     storage_path.c_str(),
    //     this->join_instance->num_rank_allocated,
    //     table_name.asCString(),
    //     col_name.asCString());

    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);
}

// int rank_id,
// std::vector<char*> &buffers,
// std::vector<int> &data_bytes,
// int block_size,
// const char *buffer_name

void JoinOperator::Execute_HOST_FUNC_RECV_DATA_OPT(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    auto node_name = (*op_node)["node_name"];

    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

    int64_t block_size = 0;

    // Get Stack Node
    StackNode *node = idp_handler->FindNode(rank_id, node_name.asCString());
    block_size = node->block_byte;
    int64_t xfer_bytes = block_size * NUM_DPU_RANK;

    if (rank_id == 0)
    {
        this->join_instance->result_bufferpool.first.resize(this->join_instance->num_rank_allocated);
        this->join_instance->result_bufferpool.second.resize(this->join_instance->num_rank_allocated);

        int64_t rank_bytes_prefix_sum[this->join_instance->num_rank_allocated + 1];
        rank_bytes_prefix_sum[0] = 0;
        int64_t summ_bytes = 0;

        for (int r = 0; r < this->join_instance->num_rank_allocated; r++)
        {
            StackNode *r_node = idp_handler->FindNode(r, node_name.asCString());
            int64_t r_block_size = r_node->block_byte;
            int64_t r_xfer_bytes = r_block_size * NUM_DPU_RANK;
            // summ_bytes += r_xfer_bytes;
            rank_bytes_prefix_sum[r + 1] = rank_bytes_prefix_sum[r] + r_xfer_bytes;

            this->join_instance->result_bufferpool.first.at(r) = (kv_pair_t *)aligned_alloc(64, r_xfer_bytes);
            this->join_instance->result_bufferpool.second.at(r) = r_xfer_bytes;
        }
    }

    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

    int64_t *buff = (int64_t *)(this->join_instance->result_bufferpool.first.at(rank_id));

    pthread_mutex_t rank_mutex;
    pthread_cond_t rank_cond;

    pthread_mutex_init(&rank_mutex, NULL);
    pthread_cond_init(&rank_cond, NULL);

    int running_jobs_per_rank = 1;

    rotate_n_stream_job_t job;
    this->join_instance->InitXferJob(
        &job,
        false,
        rank_id,
        node->start_byte,
        buff,
        xfer_bytes,
        idp_handler,
        &rank_mutex,
        &rank_cond,
        &running_jobs_per_rank);

    idp_handler->SetRNSRank(rank_id);

    this->join_instance->AddJobIntoPriorityQueue(&job);

    // Waiting for all rank jobs to finish
    WaitUSGJobFinish(running_jobs_per_rank, rank_mutex, rank_cond);
    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);
}

void JoinOperator::Execute_HOST_FUNC_SEND_DATA_OPT(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
#ifdef INTEL_ITTNOTIFY_API
    __itt_resume();
    usleep(10);
#endif

    // Check if last rank
    bool last_rank = false;
    if ((this->join_instance->num_rank_allocated - 1) == rank_id)
    {
        last_rank = true;
    }
    
    struct timespec timer_start[3];
    struct timespec timer_stop[3];

    // clock_gettime(CLOCK_MONOTONIC, timer_start + 0);

    auto buffer_name_json = (*op_node)["buffer_name"];

    int block_size = 0;

    std::string buff_name = (buffer_name_json.asString());

    RankwiseMemoryBankBufferPair_t *rankwise_mbank_buff_col = this->join_instance->GetMemoryBankBuffers(buff_name.c_str());

    if (rankwise_mbank_buff_col == nullptr)
    {
        std::cout << "Error Cannot find buffers." << std::endl;
        exit(-1);
    }

    int data_bytes[64];
    int total_bytes = 0;

    for (int d = 0; d < NUM_DPU_RANK; d++)
    {
        data_bytes[d] = rankwise_mbank_buff_col->second->at(rank_id).at(d);

        if (block_size < data_bytes[d])
            block_size = data_bytes[d];

        total_bytes += data_bytes[d];
    }

    printf("USG: total_bytes: %d\n", total_bytes);

    int64_t *buff = (int64_t *)(rankwise_mbank_buff_col->first->at(rank_id).at(0));

    int blks = block_size * 64;

    StackNode *node = idp_handler->PushStackNodeAligned(rank_id, buff_name.c_str(), data_bytes, block_size, 8192);

    pthread_mutex_t rank_mutex;
    pthread_cond_t rank_cond;

    pthread_mutex_init(&rank_mutex, NULL);
    pthread_cond_init(&rank_cond, NULL);

    int running_jobs_per_rank = 1;

    rotate_n_stream_job_t job;

    this->join_instance->InitXferJob(
        &job,
        true,
        rank_id,
        node->start_byte,
        buff,
        blks,
        idp_handler,
        &rank_mutex,
        &rank_cond,
        &running_jobs_per_rank);

    idp_handler->SetRNSRank(rank_id);

    this->join_instance->AddJobIntoPriorityQueue(&job);

    // Waiting for all rank jobs to finish
    WaitUSGJobFinish(running_jobs_per_rank, rank_mutex, rank_cond);
    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);
}

#define MIN_JOBS_PER_RANK 1

void JoinOperator::WaitUSGJobFinish(const int &num_jobs, pthread_mutex_t &mutex, pthread_cond_t &cond)
{
    while (true)
    {
        pthread_mutex_lock(&mutex);
        if (num_jobs <= 0)
            break;
        pthread_cond_wait(&cond, &mutex);
        pthread_mutex_unlock(&mutex);
    }
    pthread_mutex_unlock(&mutex);
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
}

void JoinOperator::Execute_CONTROL_FUNC_SYNC_THREADS(
    IDPHandler *idp_handler, int rank_id, DPUKernelParams_t *params, Json::Value *op_node)
{
    pthread_mutex_lock(&this->join_instance->thread_sync_mutex);

    this->join_instance->sync_value++;

    if (this->join_instance->sync_value < this->join_instance->num_rank_allocated)
    {
        pthread_cond_wait(&this->join_instance->thread_sync_cond, &this->join_instance->thread_sync_mutex);
    }
    else
    {
        this->join_instance->sync_value = 0;
        pthread_cond_broadcast(&this->join_instance->thread_sync_cond);
    }

    pthread_mutex_unlock(&this->join_instance->thread_sync_mutex);
}

static void CalculatePageHistogram(
    JoinInstance *join_instance,
    IDPHandler &idp_handler,
    int rank_id, int total_rank, int partition_num, // rank_id is probably source rank id - Comment
    RankwiseMemoryBankBufferPair_t &input_histogram_buffers,
    RankwiseMemoryBankBufferPair_t &output_dpu_xfer_packet_buffers,
    int data_type_size, int packet_size)
{
    RankwiseMemoryBankBuffers_t *src_histogram = input_histogram_buffers.first;

    // FIXME
    int packet_elem = packet_size / data_type_size;

    // Return Value
    int64_t *dpu_xfer_packet = (int64_t *)(output_dpu_xfer_packet_buffers.first->at(rank_id)[0]);

    memset(dpu_xfer_packet, 0, total_rank * sizeof(int64_t));

#ifdef COLLECT_LOGS
    int64_t xfer_byte = 0, upmem_tx_byte = 0, max_tx_byte_per_rank = 0;
#endif
    // find max xfer size to DPUs in each dst rank from a src_rank - Comment
    for (int dpu_src = 0; dpu_src < NUM_DPU_RANK; dpu_src++)
    {
#ifdef COLLECT_LOGS
        int64_t xfer_byte_per_dpu = 0;
#endif
        // per src dpu xfer size(in number of elements) to each dst dpu - Comment
        int32_t *curr_xfer_sizes = (int32_t *)src_histogram->at(rank_id).at(dpu_src);

        for (int dst_rank = 0; dst_rank < total_rank; dst_rank++)
        {
            for (int dst_dpu = 0; dst_dpu < NUM_DPU_RANK; dst_dpu++)
            {
#ifdef COLLECT_LOGS
                xfer_byte += (curr_xfer_sizes[dst_rank * NUM_DPU_RANK + dst_dpu] * data_type_size);
                xfer_byte_per_dpu += (curr_xfer_sizes[dst_rank * NUM_DPU_RANK + dst_dpu] * data_type_size);
#endif
                if (dpu_xfer_packet[dst_rank] < curr_xfer_sizes[dst_rank * NUM_DPU_RANK + dst_dpu])
                {
                    dpu_xfer_packet[dst_rank] = curr_xfer_sizes[dst_rank * NUM_DPU_RANK + dst_dpu];
                }
            }
        }
#ifdef COLLECT_LOGS
        if (xfer_byte_per_dpu > max_tx_byte_per_rank)
            max_tx_byte_per_rank = xfer_byte_per_dpu;
#endif
    }

#ifdef COLLECT_LOGS
    upmem_tx_byte += max_tx_byte_per_rank * NUM_DPU_RANK;
    int64_t xfer_packet_num = 0, max_xfer_byte = 0;
#endif

    for (int dst_rank = 0; dst_rank < total_rank; dst_rank++)
    {
        if (dpu_xfer_packet[dst_rank] == 0)
        {
            dpu_xfer_packet[dst_rank] = 1;
        }
        else
        {
            int num_packet = (dpu_xfer_packet[dst_rank] / packet_elem);
            int leftover = (dpu_xfer_packet[dst_rank] % packet_elem);
            if (leftover > 0)
            {
                num_packet++;
            }
#ifdef COLLECT_LOGS
            if (max_xfer_byte < dpu_xfer_packet[dst_rank] * data_type_size)
                max_xfer_byte = dpu_xfer_packet[dst_rank] * data_type_size;
#endif
            dpu_xfer_packet[dst_rank] = num_packet;
        }

        // printf("[%d] to [%d]: %d\n", rank_id, dst_rank, dpu_xfer_packet[dst_rank]);

#ifdef COLLECT_LOGS
        xfer_packet_num += dpu_xfer_packet[dst_rank] * NUM_DPU_RANK * NUM_DPU_RANK;
#endif
    }

#ifdef COLLECT_LOGS
    join_instance->CreateRNSLog(rank_id, xfer_byte, max_xfer_byte, xfer_packet_num, upmem_tx_byte, packet_size);
#endif

    for (int dpu = 1; dpu < NUM_DPU_RANK; dpu++)
    {
        output_dpu_xfer_packet_buffers.first->at(rank_id)[dpu] = (char *)dpu_xfer_packet;
    }
}
