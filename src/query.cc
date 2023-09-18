#include "join_internals.hpp"

#include "iostream"
#include <time.h>
#include "typeinfo"
#include <mutex>
#include <semaphore.h>
#include <numa.h>

#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <stdio.h>
#include <unistd.h>

#ifdef INTEL_ITTNOTIFY_API
#include <ittnotify.h>
#endif

using namespace pidjoin;

////////////////////////////////////////////////////////////////////

void JoinInstance::InitRnSJob( // used in Execute_TEST_FUNC_ROTATE_AND_CACHE_BYPASS - Comment
    rotate_n_stream_job_t *job,
    float job_priority,
    int src_rank,
    int dst_rank,
    int mram_src_offset,
    int mram_dst_offset,
    int job_type,
    int prefetch_distance,
    float packet_num_2_copy,
    int bankchunk_8_offset,
    int num_repeat,
    void *host_buffer,
    IDPHandler*idp_handler)
{
    job->job_priority = job_priority;
    job->src_rank = src_rank;
    job->dst_rank = dst_rank;
    job->mram_src_offset = mram_src_offset;
    job->mram_dst_offset = mram_dst_offset;
    job->job_type = job_type;
    job->prefetch_distance = prefetch_distance;
    job->next_job = NULL;
    job->src_packet_num = packet_num_2_copy;
    job->bankchunk_8_offset = bankchunk_8_offset;
    job->num_repeat = num_repeat; 
    job->host_buffer = host_buffer;
    job->idp_handler = (void *)idp_handler;
}

void JoinInstance::InitRnSJob(
    rotate_n_stream_job_t *job,
    float job_priority,
    int src_rank,
    int dst_rank,
    int mram_src_offset,
    int mram_dst_offset,
    int job_type,
    int prefetch_distance,
    float packet_num_2_copy,
    void *host_buffer,
    IDPHandler*idp_handler)
{
    job->job_priority = job_priority;
    job->src_rank = src_rank;
    job->dst_rank = dst_rank;
    job->mram_src_offset = mram_src_offset;
    job->mram_dst_offset = mram_dst_offset;
    job->job_type = job_type;
    job->prefetch_distance = prefetch_distance;
    job->next_job = NULL;
    job->src_packet_num = packet_num_2_copy;
    job->num_repeat = 1; // - Comment
    job->host_buffer = host_buffer;
    job->idp_handler = (void *)idp_handler;
    job->cache_bypass = true;
}

void JoinInstance::InitXferJob(
    rotate_n_stream_job_t *job,
    bool store_data, // is store to PIM
    int dst_rank,
    int mram_dst_offset,
    void *host_buffer, // buffer at host memory to rd/wr
    int64_t xfer_bytes,
    IDPHandler*idp_handler,
    pthread_mutex_t *mutex_ptr,
    pthread_cond_t *cond_ptr,
    int *running_jobs_per_rank)
{
    if (store_data)
    {
        job->job_priority = 0;
        job->src_rank = 0;
        job->dst_rank = dst_rank;
        job->mram_dst_offset = mram_dst_offset;
        job->job_type = DPU_TRANSFER_JOB_TYPE_UNORDERED_SCATTER;
        job->next_job = NULL;
        job->xfer_bytes = xfer_bytes;
        job->host_buffer = host_buffer;
        job->idp_handler = (void *)idp_handler;
        job->mutex = mutex_ptr;
        job->cond = cond_ptr;
        job->running_jobs_per_rank = running_jobs_per_rank;
    }
    else
    {
        job->job_priority = 0;
        job->src_rank = dst_rank;
        job->dst_rank = 0;
        job->mram_src_offset = mram_dst_offset;
        job->job_type = DPU_TRANSFER_JOB_TYPE_UNORDERED_GATHER;
        job->next_job = NULL;
        job->xfer_bytes = xfer_bytes;
        job->host_buffer = host_buffer;
        job->idp_handler = (void *)idp_handler;
        job->mutex = mutex_ptr;
        job->cond = cond_ptr;
        job->running_jobs_per_rank = running_jobs_per_rank;
    }

}

ResultBuffers_t JoinInstance::ExecuteJoin(std::string join_type)
{
    this->join_operator->join_alg = join_type;
    if (join_type == "smj" or join_type == "phj")
    {
        this->join_operator->ExecuteJoin();
        return this->result_bufferpool;
    }
    else
    {
        std::cout << "Not supoorted join algorithm " << join_type << std::endl;
        exit(-1);
    }
}

void JoinInstance::LoadColumn(void* data_ptr, int64_t num_tuples, std::string col_name)
{
    int num_ranks = this->num_rank_allocated;
    uint8_t* data_ptr_uint8 = (uint8_t*)data_ptr;
    RankwiseMemoryBankBufferPair_t* imm_pair = this->AllocateEmptyMemoryBankBuffers(num_ranks, col_name.c_str());
    
    for (int rank_id = 0; rank_id < num_ranks; rank_id++)
    {
        imm_pair->first->at(rank_id).resize(NUM_DPU_RANK);
        imm_pair->second->at(rank_id).resize(NUM_DPU_RANK);
    }

    int64_t origin_tuple_cnt = num_tuples;
    int64_t origin_byte_size = num_tuples * sizeof(tuplePair_t);
    
    std::cout << "origin_tuple_cnt: " << origin_tuple_cnt << std::endl;
    std::cout << "origin_byte_size: " << origin_byte_size << std::endl;

    int64_t alignment = num_ranks * NUM_DPU_RANK * 8;
    int64_t offset = (alignment) - (origin_byte_size % alignment);
    
    int64_t total_byte_size = origin_byte_size + offset;
    int64_t total_tuple_cnt = total_byte_size / sizeof(tuplePair_t);

    int64_t tuple_per_dpu = (total_tuple_cnt) / (num_ranks * NUM_DPU_RANK);
    int64_t byte_per_dpu = tuple_per_dpu * sizeof(tuplePair_t);

    int offseted_dpus = offset / byte_per_dpu;
    if (offset % byte_per_dpu > 0)
    {
        offseted_dpus++;
    }

    int dpu_start_offseted = (num_ranks * NUM_DPU_RANK) - offseted_dpus;
    
    
    std::cout << "dpu_start_offseted: " << dpu_start_offseted << std::endl;

    std::cout << "offset: " << offset << std::endl;

    std::cout << "tuple_per_dpu: " << tuple_per_dpu << std::endl;
    std::cout << "byte_per_dpu: " << byte_per_dpu << std::endl;


    int64_t data_ctr = 0;
    // Read File

    int dpu_count = 0;
    for (int rank_id = 0; rank_id < num_ranks; rank_id++)
    {
        std::vector<char*>& write_buffers = std::ref(imm_pair->first->at(rank_id));
        std::vector<int>& block_bytes = std::ref(imm_pair->second->at(rank_id));
        // write_buffers[0] = (char*)aligned_alloc(NUM_DPU_RANK, byte_per_dpu * NUM_DPU_RANK);
        
        for (int dpu_id = 0; dpu_id < NUM_DPU_RANK; dpu_id++)
        {
            if (dpu_start_offseted <= dpu_count)
            {
                write_buffers[dpu_id] = (char*)malloc(byte_per_dpu);
                if (data_ctr >= origin_byte_size)
                {
                    memset(write_buffers[dpu_id], 0, byte_per_dpu); 
                }
                else
                {
                    for (int i = 0; i < byte_per_dpu / sizeof(int64_t); i++)
                    {
                        if (origin_byte_size - (data_ctr + i * sizeof(int64_t)))
                        {
                            ((int64_t*)(write_buffers[dpu_id]))[i] = ((int64_t*)(data_ptr_uint8 + data_ctr))[i];
                        }
                        else
                        {
                            ((int64_t*)(write_buffers[dpu_id]))[i] = 0;
                        }
                    }

                }
                // write_buffers[dpu_id] = (char*)(data_ptr_uint8 + data_ctr);
                data_ctr += (byte_per_dpu);
                block_bytes[dpu_id] = (byte_per_dpu);
            }
            else
            {

                // memcpy(write_buffers[dpu_id], data_ptr_uint8 + data_ctr, byte_per_dpu);
                write_buffers[dpu_id] = (char*)(data_ptr_uint8 + data_ctr);
                data_ctr += (byte_per_dpu);
                block_bytes[dpu_id] = (byte_per_dpu);
                // printf("block_bytes[%d]: %d\n", dpu_id, block_bytes[dpu_id]);
            }
            dpu_count++;
        }
    } 

    std::cout << "data_ctr " << data_ctr  << std::endl;
    std::cout << "total_tuple_cnt * sizeof(tuplePair_t)" << total_tuple_cnt * sizeof(tuplePair_t)  << std::endl;
}
    
    
    /////////////////////////
    /////////////////////////

    // int64_t total_tuple_leftover = total_tuple_cnt % (num_ranks * NUM_DPU_RANK);
    // int64_t tuple_per_dpu = total_tuple_cnt / (num_ranks * NUM_DPU_RANK);
    // int64_t byte_per_dpu = tuple_per_dpu * sizeof(tuplePair_t);
    
    // if (total_tuple_leftover > 0)
    //     byte_per_dpu += sizeof(tuplePair_t);

    // printf("total_byte_size: %ld\n", total_byte_size);
    // printf("total_tuple_cnt: %ld\n", total_tuple_cnt);
    // printf("total_tuple_leftover: %ld\n", total_tuple_leftover);
    // printf("tuple_per_dpu: %ld\n", tuple_per_dpu);
    // printf("byte_per_dpu: %ld\n", byte_per_dpu);

    // printf("| Start Loading Column %s\n", col_name.c_str());

    // int dpu_ctr = 0;
    // int64_t data_ctr = 0;
    // // Read File
    // for (int rank_id = 0; rank_id < num_ranks; rank_id++)
    // {
    //     std::vector<char*>& write_buffers = std::ref(imm_pair->first->at(rank_id));
    //     std::vector<int>& block_bytes = std::ref(imm_pair->second->at(rank_id));
    //     // write_buffers[0] = (char*)aligned_alloc(NUM_DPU_RANK, byte_per_dpu * NUM_DPU_RANK);
        
    //     for (int dpu_id = 0; dpu_id < NUM_DPU_RANK; dpu_id++)
    //     {
    //         // write_buffers[dpu_id] = write_buffers[0] + byte_per_dpu * dpu_id;

    //         if (total_tuple_leftover == 0)
    //         {
    //             // memcpy(write_buffers[dpu_id], data_ptr_uint8 + data_ctr, byte_per_dpu);
    //             write_buffers[dpu_id] = (char*)(data_ptr_uint8 + data_ctr);
    //             data_ctr += (byte_per_dpu);
    //             block_bytes[dpu_id] = (byte_per_dpu);
    //         }
    //         else
    //         {
    //             if (dpu_ctr >= total_tuple_leftover)
    //             {
    //                 // memcpy(write_buffers[dpu_id], data_ptr_uint8 + data_ctr, byte_per_dpu - sizeof (tuplePair_t));
    //                 write_buffers[dpu_id] = (char*)(data_ptr_uint8 + data_ctr);
    //                 if ((intptr_t)(write_buffers[dpu_id]) % 8 != 0)
    //                     printf("write_buffers[%d]: 0x%lx aligned?: %d\n", dpu_id, write_buffers[dpu_id]);
    //                 data_ctr += (byte_per_dpu - sizeof (tuplePair_t));
    //                 block_bytes[dpu_id] = (byte_per_dpu - sizeof (tuplePair_t));
    //                 // printf("block_bytes[%d]: %d\n", dpu_id, block_bytes[dpu_id]);
    //             }
    //             else
    //             {
    //                 // memcpy(write_buffers[dpu_id], data_ptr_uint8 + data_ctr, byte_per_dpu);
    //                 write_buffers[dpu_id] = (char*)(data_ptr_uint8 + data_ctr);
    //                 if ((intptr_t)(write_buffers[dpu_id]) % 8 != 0)
    //                     printf("write_buffers[%d]: 0x%lx aligned?: %d\n", dpu_id, write_buffers[dpu_id]);
    //                 data_ctr += (byte_per_dpu);
    //                 block_bytes[dpu_id] = (byte_per_dpu);
    //                 // printf("block_bytes[%d]: %d\n", dpu_id, block_bytes[dpu_id]);
    //             }    
    //         }
            
    //         dpu_ctr++;
    //     } 
    // }
// }


std::pair<int, lock_ *> JoinInstance::GetQueueElemByIndex(int index)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));
    auto ret = this->thread_queue_line.at(index);
    pthread_mutex_unlock(&(this->thread_queue_mutex));
    return ret;
}

std::pair<int, lock_ *> JoinInstance::GetQueueElemByRankID(int rank_id)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));
    for (auto &e : this->thread_queue_line)
    {
        if (e.first == rank_id)
        {
            pthread_mutex_unlock(&(this->thread_queue_mutex));
            return e;
        }
    }
    printf("%sError; reaches end.\n", KRED);
    exit(-1);
}

// Returns its order
int JoinInstance::ReadyQueueLine(int rank_id, lock_ *lock)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));
    int my_order = curr_queued;
    this->thread_queue_line[curr_queued] = std::make_pair(rank_id, lock);
    this->rns_rank_thread_orders.push_back(rank_id);
    curr_queued++;
    pthread_mutex_unlock(&(this->thread_queue_mutex));
    return my_order;
}

// change thread_queue_line[rank_id] and rns_rank_thread_orders - Comment
int JoinInstance::ReadyQueueLineFixedPriority(int rank_id, lock_ *lock)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));

    this->thread_queue_line[rank_id] = std::make_pair(rank_id, lock);

    if (this->rns_rank_thread_orders.size() == 0)
    {
        this->rns_rank_thread_orders.resize(this->num_rank_allocated);

        for (int r = 0; r < this->num_rank_allocated; r++)
        {
            this->rns_rank_thread_orders[r] = r;
        }
    }

    pthread_mutex_unlock(&(this->thread_queue_mutex));
    return rank_id;
}

/*
 * Called at the end of each rank thread of RNS - Comment
 * Wait for all the jobs a rank has to do finish. e.g. send to all other ranks and receive from all other ranks
 */
int JoinInstance::WaitJobDoneQueueLine(int index, GlobalBuffer_t *my_accmuls, int num_ranks, int packet_size)
{
    // printf("WaitJobDoneQueueLine: index:%d\n", index);
    auto elem = this->GetQueueElemByIndex(index); // get rank_id, lock pair from thread_queue_line - Comment
    auto lock = elem.second;

    int return_val = 0;
    // printf("%d WaitJobDoneQueueLine!!\n", index);

    pthread_mutex_lock(&lock->mutex);

    if (lock->job_counter == this->tot_jobs_per_rank)
    {
        // printf("bypassing11111111111111111111111111111~\n");
    }
    else if (lock->job_counter > this->tot_jobs_per_rank)
    {
        printf("%sError!: lock->job_counter is bigger than this->num_rank_allocated\n", KRED);
        exit(-1);
    }
    else
    {
        pthread_cond_wait(&lock->cond, &lock->mutex); // wait for
    }

    if (my_accmuls != NULL) // - Comment
    {
        int64_t *dat = (int64_t *)(my_accmuls->aligned_data);
        int64_t num_packets_received = dat[num_ranks];
        int64_t output_node_data_byte = num_packets_received * NUM_DPU_RANK * packet_size;
        
            if (output_node_data_byte <= 0)
            {

                printf("%sError! output_node_data_byte:%ld <= 0\n", KRED, output_node_data_byte);

                printf("HISTOGRAM\n");

                for (int rr = 0; rr < num_ranks; rr++)
                {
                    printf("HISTOGRAM[%d]: %ld, ", rr, dat[rr]);
                }
                printf("\n");

                exit(-1);
        }
        else
        {
            return_val = output_node_data_byte;
        }
    }

    pthread_mutex_unlock(&lock->mutex);
    pthread_mutex_destroy(&lock->mutex);
    pthread_cond_destroy(&lock->cond);
    
    delete lock;

    return return_val;
}

int JoinInstance::GetJobStatus(int index)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));
    auto ret = this->thread_queue_line.at(index);
    int retval = ret.second->job_counter;
    pthread_mutex_unlock(&(this->thread_queue_mutex));
    return retval;
}

/*
 * Called when a job is end - Comment
 * add number of finished job of 'index' rank
 * return true if a rank jobs are all done
 */
bool JoinInstance::UpdateJobStatus(int index)
{
    pthread_mutex_lock(&(this->thread_queue_mutex));
    auto ret = this->thread_queue_line.at(index);
    if (ret.second == NULL)
    {
        printf("Error: %d ret.second is Null\n", index);
    }
    pthread_mutex_lock(&ret.second->mutex);
    ret.second->job_counter++;
    int retval = ret.second->job_counter;
    pthread_mutex_unlock(&ret.second->mutex);
    pthread_mutex_unlock(&(this->thread_queue_mutex));

    if (retval > this->tot_jobs_per_rank)
    {
        printf("%sError!: status is bigger than this->tot_jobs_per_rank job_counter: %d > tot_jobs_per_rank: %d\n", KRED, retval, this->tot_jobs_per_rank);
        exit(-1);
    }
    return (retval == this->tot_jobs_per_rank);
}

void JoinInstance::WakeUpQueueLine(int index)
{
    auto elem = this->GetQueueElemByIndex(index);
    auto lock = elem.second;
    pthread_mutex_lock(&thread_queue_mutex);
    // printf("WakeUpQueueLine:: %d\n", index);
    pthread_mutex_lock(&lock->mutex);
    pthread_cond_signal(&lock->cond);
    pthread_mutex_unlock(&lock->mutex);

    pthread_mutex_unlock(&thread_queue_mutex);
}


JoinInstance::JoinInstance(int num_rank_allocated_)
{
    std::string timeline_path = "notimeline";

    int xfer_worker_num = NUM_MAX_RANKS;
    this->join_algorithm = "";

    if (xfer_worker_num == 0)
        xfer_worker_num = 8;

    this->packet_size = 8;
    this->scale_factor = 0;
    this->zipf_factor = 0;

    if (timeline_path != "notimeline")
        this->timeline_ptr = fopen(timeline_path.c_str(), "w");
    else
        this->timeline_ptr = NULL;

    this->num_rank_allocated = num_rank_allocated_;

    printf("%s"
        "Packet Size: %d\n"
        "Xfer Worker: %d\n"
        "Zipf Factor: %lf\n"
        "Scale Factor: %d\n"
        "Timeline Path: %s\n", KCYN, packet_size, xfer_worker_num, zipf_factor, scale_factor, timeline_path.c_str());
    printf("%s", KWHT);
    fflush(stdout);

    rankwiseStatLogs.resize(this->num_rank_allocated);

    this->thread_queue_line.resize(this->num_rank_allocated);
    std::fill(this->thread_queue_line.begin(), this->thread_queue_line.end(), std::pair<int, lock_ *>(-1, NULL));
    curr_queued = 0;
    this->tot_jobs_per_rank = num_rank_allocated_ * 2; // just initial value for code backward compatibility - Comment

    this->rns_rank_thread_orders;

    this->join_operator = new JoinOperator(this);

    sem_init(&rotate_and_stream_semaphore, 0, this->num_rank_allocated);
    
    m_xfer_worker_num = xfer_worker_num;

    if (pthread_mutex_init(&timeline_mutex, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&timeline_mutex, NULL) Failed" << std::endl;
        exit(-1);
    }
    if (pthread_mutex_init(&thread_queue_mutex, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&thread_queue_mutex, NULL) Failed" << std::endl;
        exit(-1);
    }
    if (pthread_mutex_init(&global_buffer_mutex, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&global_buffer_mutex, NULL) Failed" << std::endl;
        exit(-1);
    }
    if (pthread_mutex_init(&rankwise_buff_mutex, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&rankwise_buff_mutex, NULL) Failed" << std::endl;
        exit(-1);
    }
    if (pthread_mutex_init(&instance_lock, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&instance_lock, NULL) Failed" << std::endl;
        exit(-1);
    }
    
    pthread_mutexattr_t attr2;
    pthread_mutexattr_settype(&attr2, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutexattr_init(&attr2);
    if (pthread_mutex_init(&job_done_mutex, &attr2) != 0)
    {
        printf("%sError pthread init Error. %s:%d\n", KRED, __FILE__, __LINE__);
        exit(-1);
    }
    pthread_mutexattr_t attr3;
    pthread_mutexattr_settype(&attr3, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutexattr_init(&attr3);

    pthread_mutexattr_t attr4;
    pthread_mutexattr_settype(&attr4, PTHREAD_MUTEX_ERRORCHECK);
    if (pthread_mutex_init(&rotate_and_stream_mutex, &attr4) != 0)
    {
        printf("%sError pthread init Error. %s:%d\n", KRED, __FILE__, __LINE__);
        exit(-1);
    }

    if (pthread_mutex_init(&thread_sync_mutex, NULL) != 0)
    {
        std::cout << "ERROR: pthread_mutex_init(&thread_sync_mutex, NULL) Failed" << std::endl;
        exit(-1);
    }
    if (pthread_cond_init(&thread_sync_cond, NULL) != 0)
    {
        std::cout << "ERROR: pthread_cond_init(&thread_sync_cond, NULL) Failed" << std::endl;
        exit(-1);
    }
    sync_value = 0;
}

void JoinInstance::CreateRNSLog(int rank_id, int64_t xfer_byte, int64_t max_xfer_byte, int64_t xfer_packet_num, int64_t upmem_tx_byte, int packet_size)
{
    static int rnc_count[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string name = "RNS_" + std::to_string(rnc_count[rank_id]);
    rnc_count[rank_id]++;

    stat_info_t si;
    si.STAT_TYPE = STAT_TYPE_RNS;
    si.xfer_byte = xfer_byte;
    si.max_xfer_byte = max_xfer_byte;
    si.xfer_packet_num = xfer_packet_num;
    si.upmem_tx_byte = upmem_tx_byte;
    si.packet_size = packet_size;
    this->rankwiseStatLogs[rank_id].insert(std::make_pair(name, si));
}

void JoinInstance::CreateJoinLog(int rank_id, int32_t num_tuples, int32_t hit_count, int32_t miss_count)
{
    static int join_count[32] = {
        0,
    };
    std::string name = "JOIN_" + std::to_string(join_count[rank_id]);
    join_count[rank_id]++;

    stat_info_t si;
    si.STAT_TYPE = STAT_TYPE_JOIN;
    si.miss_count = miss_count;
    si.num_tuples = num_tuples;
    si.hit_count = hit_count;

    this->rankwiseStatLogs[rank_id].insert(std::make_pair(name, si));
}

JoinInstance::~JoinInstance()
{
    if (this->timeline_ptr != NULL)
    {
        fclose(this->timeline_ptr);
    }

    for (auto &tbuff_pair : this->global_buffer_map)
    {
        GlobalBuffer_t *tbuff = tbuff_pair.second;
        free(tbuff->aligned_data);
        pthread_mutex_destroy(&tbuff->lock);
        pthread_cond_destroy(&tbuff->cond);
        delete tbuff;
    }

    this->global_buffer_map.clear();

    for (auto elem : this->rankwise_buffer_map)
    {
        auto &rankwise_mbank_buff_pair = elem.second;

        for (auto &rankwise_mbank_buff : *rankwise_mbank_buff_pair.first)
        {
            // FIXME
            // numa_free(rankwise_mbank_buff[0]);
        }

        delete rankwise_mbank_buff_pair.first;
        delete rankwise_mbank_buff_pair.second;
    }

    delete this->join_operator;

    pthread_mutex_destroy(&timeline_mutex);
    pthread_mutex_destroy(&job_done_mutex);
    pthread_mutex_destroy(&rotate_and_stream_mutex);
    pthread_mutex_destroy(&thread_queue_mutex);
    pthread_mutex_destroy(&global_buffer_mutex);
    pthread_mutex_destroy(&rankwise_buff_mutex);
    pthread_mutex_destroy(&instance_lock);
    pthread_mutex_destroy(&thread_sync_mutex);
    pthread_cond_destroy(&thread_sync_cond);

    sem_destroy(&rotate_and_stream_semaphore);
}

RankwiseMemoryBankBufferPair_t *JoinInstance::AllocateMemoryBankBuffersRankwise(
    int64_t size_byte, int rank_id, const char *name)
{
    // Find if buffer is already exists
    pthread_mutex_lock(&this->rankwise_buff_mutex);

    auto it = this->rankwise_buffer_map.find(std::string(name));
    
    if (it == this->rankwise_buffer_map.end())
    {
        RankwiseMemoryBankBuffers_t *rankwise_mbank_buffs = new RankwiseMemoryBankBuffers_t(NUM_MAX_RANKS);
        RankwiseMemoryBankFilledBytes_t *rankwise_mbank_buff_filled_bytes = new RankwiseMemoryBankFilledBytes_t(NUM_MAX_RANKS);

        // for (int i = 0; i < rankwise_mbank_buffs->size(); i++)
        {
            std::vector<char *> &rankwise_mbank_buff = rankwise_mbank_buffs->at(rank_id);
            auto &rankwise_mbank_buff_filled_byte = rankwise_mbank_buff_filled_bytes->at(rank_id);

            rankwise_mbank_buff.resize(NUM_DPU_RANK);
            rankwise_mbank_buff_filled_byte.resize(NUM_DPU_RANK);

            // char *buffer = (char *)numa_alloc_onnode(size_byte * NUM_DPU_RANK, 1);
            char *buffer = (char *)aligned_alloc(64, size_byte * NUM_DPU_RANK);
            if (buffer == nullptr)
            {
                printf("%sError!: numa_alloc_onnode Failed.\n", KRED); exit(-1);
            }

            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                rankwise_mbank_buff[i] = buffer + i * size_byte;
                rankwise_mbank_buff_filled_byte[i] = size_byte;
            }
        }

        this->rankwise_buffer_map.insert(std::make_pair(std::string(name), std::make_pair(rankwise_mbank_buffs, rankwise_mbank_buff_filled_bytes)));
    }
    else
    {
        auto& pair = this->rankwise_buffer_map.at(std::string(name));

        // for (int i = 0; i < rankwise_mbank_buffs->size(); i++)
        {
            std::vector<char *> &rankwise_mbank_buff = pair.first->at(rank_id);
            auto &rankwise_mbank_buff_filled_byte = pair.second->at(rank_id);

            rankwise_mbank_buff.resize(NUM_DPU_RANK);
            rankwise_mbank_buff_filled_byte.resize(NUM_DPU_RANK);

            // char *buffer = (char *)numa_alloc_onnode(size_byte * NUM_DPU_RANK, 1);
            char *buffer = (char *)aligned_alloc(64, size_byte * NUM_DPU_RANK);
            if (buffer == nullptr)
            {
                printf("%sError!: numa_alloc_onnode Failed.\n", KRED); exit(-1);
            }

            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                rankwise_mbank_buff[i] = buffer + i * size_byte;
                rankwise_mbank_buff_filled_byte[i] = size_byte;
            }
        }
    }

    auto &ret = *(this->rankwise_buffer_map.find(std::string(name)));
    pthread_mutex_unlock(&this->rankwise_buff_mutex);

    return &(ret.second);
}

RankwiseMemoryBankBufferPair_t *JoinInstance::AllocateEmptyMemoryBankBuffersRankwise(
    int rank_id, const char *name)
{
    pthread_mutex_lock(&this->rankwise_buff_mutex);

    auto it = this->rankwise_buffer_map.find(std::string(name));
    
    if (it == this->rankwise_buffer_map.end())
    {
        RankwiseMemoryBankBuffers_t *rankwise_mbank_buffs = new RankwiseMemoryBankBuffers_t(this->num_rank_allocated);
        RankwiseMemoryBankFilledBytes_t *rankwise_mbank_buff_filled_bytes = new RankwiseMemoryBankFilledBytes_t(this->num_rank_allocated);

        for (int r = 0; r < this->num_rank_allocated; r++)
        {
            rankwise_mbank_buffs->at(r).resize(NUM_DPU_RANK);
            rankwise_mbank_buff_filled_bytes->at(r).resize(NUM_DPU_RANK);

            rankwise_mbank_buffs->at(r).at(0) = nullptr;
            rankwise_mbank_buff_filled_bytes->at(r).at(0) = 0;
        }
        this->rankwise_buffer_map.insert(std::make_pair(std::string(name), std::make_pair(rankwise_mbank_buffs, rankwise_mbank_buff_filled_bytes)));
    }

    auto &ret = *(this->rankwise_buffer_map.find(std::string(name)));
    pthread_mutex_unlock(&this->rankwise_buff_mutex);
    return &(ret.second);
}

RankwiseMemoryBankBufferPair_t *JoinInstance::AllocateMemoryBankBuffers(
    int64_t size_byte, int num_rank, const char *name)
{
    RankwiseMemoryBankBuffers_t *rankwise_mbank_buffs = new RankwiseMemoryBankBuffers_t(num_rank);
    RankwiseMemoryBankFilledBytes_t *rankwise_mbank_buff_filled_bytes = new RankwiseMemoryBankFilledBytes_t(num_rank);

    for (int i = 0; i < rankwise_mbank_buffs->size(); i++)
    {
        std::vector<char *> &rankwise_mbank_buff = rankwise_mbank_buffs->at(i);
        auto &rankwise_mbank_buff_filled_byte = rankwise_mbank_buff_filled_bytes->at(i);

        rankwise_mbank_buff.resize(NUM_DPU_RANK);
        rankwise_mbank_buff_filled_byte.resize(NUM_DPU_RANK);

        char *buffer = (char *)malloc(size_byte * NUM_DPU_RANK);

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            rankwise_mbank_buff[i] = buffer + i * size_byte;
            rankwise_mbank_buff_filled_byte[i] = size_byte;
        }
    }

    this->rankwise_buffer_map.insert(std::make_pair(std::string(name), std::make_pair(rankwise_mbank_buffs, rankwise_mbank_buff_filled_bytes)));

    auto &ret = *(this->rankwise_buffer_map.find(std::string(name)));
    return &(ret.second);
}

RankwiseMemoryBankBufferPair_t *JoinInstance::AllocateEmptyMemoryBankBuffers(int num_rank, const char *name)
{
    RankwiseMemoryBankBuffers_t *rankwise_mbank_buffs = new RankwiseMemoryBankBuffers_t(num_rank);
    RankwiseMemoryBankFilledBytes_t *rankwise_mbank_buff_filled_bytes = new RankwiseMemoryBankFilledBytes_t(num_rank);

    std::cout << "Allocated: " << name << std::endl;
    this->rankwise_buffer_map.insert(std::make_pair(std::string(name), std::make_pair(rankwise_mbank_buffs, rankwise_mbank_buff_filled_bytes)));

    auto &ret = *(this->rankwise_buffer_map.find(std::string(name)));
    return &(ret.second);
}

void JoinInstance::RemoveMemoryBankBuffers(const char *name)
{
    auto it = rankwise_buffer_map.find(std::string(name));
    if (it == rankwise_buffer_map.end())
    {
        return;
    }
    else
    {
        auto elem = (*it);
        auto rankwise_mbank_buff_pair = elem.second;

        for (auto rankwise_mbank_buff : *rankwise_mbank_buff_pair.first)
        {
            free(rankwise_mbank_buff[0]);
        }

        delete rankwise_mbank_buff_pair.first;
        delete rankwise_mbank_buff_pair.second;
    }

    this->rankwise_buffer_map.erase(std::string(name));
}


////////////////////////////////////////////////
// Temporal Buffer Management
////////////////////////////////////////////////

GlobalBuffer_t *JoinInstance::WaitForGlobalBuffer(const char *name)
{
    std::string name_str = std::string(name);
    pthread_mutex_lock(&this->global_buffer_mutex);
    printf("Try Wait...%s\n", name);
    if (this->global_buffer_map.find(name_str) == this->global_buffer_map.end())
    {
        GlobalBuffer_t *tbuff = new GlobalBuffer_t;

        if (pthread_mutex_init(&tbuff->lock, NULL) != 0)
        {
            std::cout << "Error pthread_mutex_init(&tbuff->lock, NULL) Failed\n"; exit(-1);
        }
        if (pthread_cond_init(&tbuff->cond, NULL) != 0)
        {
            std::cout << "Error pthread_cond_init(&tbuff->cond, NULL) Failed\n"; exit(-1);
        }

        this->global_buffer_map.insert(std::make_pair(std::string(name), tbuff));
        // Not Allocate Data Here
        pthread_cond_wait(&tbuff->cond, &this->global_buffer_mutex);
        printf("Done Wait...%s`````````````\n", name);
        pthread_mutex_unlock(&this->global_buffer_mutex);
        return tbuff;
    }
    else
    {
        GlobalBuffer_t *tbuff = this->global_buffer_map.at(name_str);
        if (tbuff->allocated == true)
        {
            printf("Done Wait...%s~~~~~~~~~~~~~~~~~~~\n", name);
            pthread_mutex_unlock(&this->global_buffer_mutex);
            return tbuff;
        }
        else
        {
            this->global_buffer_map.insert(std::make_pair(std::string(name), tbuff));

            pthread_cond_wait(&tbuff->cond, &this->global_buffer_mutex);
            printf("Done Wait...%s--------------\n", name);
            pthread_mutex_unlock(&this->global_buffer_mutex);

            return tbuff;
        }
    }
}

GlobalBuffer_t *JoinInstance::GetOrAllocateGlobalBuffer(int64_t size_byte, const char *name, bool do_memset)
{
    return AllocateGlobalBuffer(size_byte, name, do_memset);
}

GlobalBuffer_t *JoinInstance::GetOrAllocateGlobalBuffer(int64_t size_byte, const char *name)
{
    return AllocateGlobalBuffer(size_byte, name, false);
}

GlobalBuffer_t *JoinInstance::AllocateGlobalBuffer(int64_t size_byte, const char *name, bool do_memset)
{
    std::string name_str = std::string(name);
    pthread_mutex_lock(&this->global_buffer_mutex);

    if (this->global_buffer_map.find(name_str) == this->global_buffer_map.end())
    {
        // need to 64 aligned.
        GlobalBuffer_t *tbuff = new GlobalBuffer_t;

        if (pthread_mutex_init(&tbuff->lock, NULL) != 0)
        {
            std::cout << "Error pthread_mutex_init(&tbuff->lock, NULL) Failed\n"; exit(-1);
        }
        if (pthread_cond_init(&tbuff->cond, NULL) != 0)
        {
            std::cout << "Error pthread_cond_init(&tbuff->cond, NULL) Failed\n"; exit(-1);
        }

        tbuff->aligned_data = aligned_alloc(64, size_byte);
        tbuff->alloced_size = size_byte;
        tbuff->allocated = true;

        if (tbuff->aligned_data == nullptr)
        {
            printf("ERROR! Allocation Error!\n");
            exit(-1);
        }

        this->global_buffer_map.insert(std::make_pair(std::string(name), tbuff));

        if (do_memset)
            memset(tbuff->aligned_data, 0, size_byte);

        pthread_mutex_unlock(&this->global_buffer_mutex);
        return tbuff;
    }
    else
    {
        GlobalBuffer_t *tbuff = this->global_buffer_map.at(name_str);

        pthread_mutex_lock(&tbuff->lock);

        if (tbuff->allocated == false)
        {
            tbuff->aligned_data = aligned_alloc(64, size_byte);
            tbuff->alloced_size = size_byte;
            tbuff->allocated = true;

            if (tbuff->aligned_data == nullptr)
            {
                printf("ERROR! Allocation Error!\n");
                exit(-1);
            }

            if (do_memset)
                memset(tbuff->aligned_data, 0, size_byte);

            pthread_cond_broadcast(&tbuff->cond);
        }

        pthread_mutex_unlock(&tbuff->lock);
        pthread_mutex_unlock(&this->global_buffer_mutex);
        return tbuff;
    }
}

GlobalBuffer_t *JoinInstance::AllocateGlobalBuffer(int64_t size_byte, const char *name)
{
    std::string name_str = std::string(name);
    pthread_mutex_lock(&this->global_buffer_mutex);

    if (this->global_buffer_map.find(name_str) == this->global_buffer_map.end())
    {
        // need to 64 aligned.
        GlobalBuffer_t *tbuff = new GlobalBuffer_t;

        pthread_mutex_init(&tbuff->lock, NULL);
        pthread_cond_init(&tbuff->cond, NULL);

        tbuff->aligned_data = (int64_t*)aligned_alloc(64, size_byte);
        tbuff->alloced_size = size_byte;
        tbuff->allocated = true;

        if (tbuff->aligned_data == nullptr)
        {
            printf("ERROR! Allocation Error!\n");
            exit(-1);
        }

        this->global_buffer_map.insert(std::make_pair(std::string(name), tbuff));

        pthread_mutex_unlock(&this->global_buffer_mutex);

        return tbuff;
    }
    else
    {
        GlobalBuffer_t *tbuff = this->global_buffer_map.at(name_str);

        pthread_mutex_lock(&tbuff->lock);
        if (tbuff->allocated == false)
        {
            tbuff->aligned_data = (int64_t*)aligned_alloc(64, size_byte);
            tbuff->alloced_size = size_byte;
            tbuff->allocated = true;
            if (tbuff->aligned_data == nullptr)
            {
                printf("ERROR! Allocation Error!\n");
                exit(-1);
            }

            pthread_cond_signal(&tbuff->cond);
        }
        pthread_mutex_unlock(&tbuff->lock);
        pthread_mutex_unlock(&this->global_buffer_mutex);
        return tbuff;
    }
}

GlobalBuffer_t *JoinInstance::GetGlobalBuffer(const char *name)
{
    pthread_mutex_lock(&this->global_buffer_mutex);
    GlobalBuffer_t *tbuff = this->global_buffer_map.at(std::string(name));
    pthread_mutex_unlock(&this->global_buffer_mutex);

    return tbuff;
}

void JoinInstance::RemoveGlobalBuffer(const char *name)
{
    pthread_mutex_lock(&this->global_buffer_mutex);
    if (this->global_buffer_map.find(std::string(name)) != this->global_buffer_map.end())
    {
        GlobalBuffer_t *tbuff = this->global_buffer_map.at(std::string(name));
        pthread_mutex_lock(&tbuff->lock);
        free (tbuff->aligned_data);
        this->global_buffer_map.erase(std::string(name));
        pthread_mutex_unlock(&tbuff->lock);
        pthread_mutex_destroy(&tbuff->lock);
        pthread_cond_destroy(&tbuff->cond);
        delete tbuff;
    }
    pthread_mutex_unlock(&this->global_buffer_mutex);
}

RankwiseMemoryBankBufferPair_t *JoinInstance::GetMemoryBankBuffers(const char *name)
{
    pthread_mutex_lock(&this->rankwise_buff_mutex);
    auto it = rankwise_buffer_map.find(std::string(name));
    if (it == rankwise_buffer_map.end())
    {
        pthread_mutex_unlock(&this->rankwise_buff_mutex);
        return NULL;
    }
    else
    {
        RankwiseMemoryBankBufferPair_t *ret = &((*it).second);
        pthread_mutex_unlock(&this->rankwise_buff_mutex);
        return ret;
    }
}

RankwiseMemoryBankBufferPair_t *JoinInstance::GetMemoryBankBuffers(std::string &name)
{
    return this->GetMemoryBankBuffers(name.c_str());
}

////////////////////////////////////////////////////////////////////

void JoinOperator::ExecuteJoin()
{
    IDPHandler idp_handler(this->join_instance->m_xfer_worker_num);
    idp_handler.SetJoinInstanceOnRNSJobQueue(this->join_instance);

    /***
     *  Allocate DPUs
     **/

    std::vector<int> rank_ids;
    idp_handler.AllocateRankPipeline(rank_ids, this->join_instance->num_rank_allocated);
    
    int num_ranks = rank_ids.size();
    
    idp_handler.PrintDPUInfo();

    // Load Program (load any program for the initialization. )
    idp_handler.LoadProgram(rank_ids, DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE);
    idp_handler.ClearMRAM(rank_ids);

    /***
     *  Execute Query Plans
     **/
    clock_gettime(CLOCK_MONOTONIC, &join_instance->m_timer_init);
    
    // Start Threads
    for (auto r : rank_ids)
    {
        // printf("Send Signal %d\n", r);
        idp_handler.rank_thread_handle[r].join_operator = this;
        pthread_mutex_lock(&(idp_handler.rank_thread_handle[r].thr_control.wait_mutex));
        pthread_cond_signal(&(idp_handler.rank_thread_handle[r].thr_control.wait_condition));
        pthread_mutex_unlock(&(idp_handler.rank_thread_handle[r].thr_control.wait_mutex));
    }

    while (true)
    {
        pthread_mutex_lock(&(idp_handler.master_thr_mutex));
        pthread_cond_wait(&(idp_handler.master_thr_condition), &(idp_handler.master_thr_mutex));

        bool all_terminated = true;

        for (auto r : rank_ids)
        {
            if (idp_handler.rank_thread_handle[r].curr_status != RANK_THREAD_STATUS_JOB_DONE)
            {
                all_terminated = false;
            }
        }

        pthread_mutex_unlock(&(idp_handler.master_thr_mutex));

        // Break Condition
        if (all_terminated)
            break;
    }

    for (auto r : rank_ids)
    {
        pthread_join(idp_handler.rank_threads[r], NULL);
        idp_handler.rank_thread_handle[r].curr_status = RANK_THREAD_STATUS_TERMINATED;
        pthread_mutex_destroy(&(idp_handler.rank_thread_handle[r].thr_control.wait_mutex));
        pthread_cond_destroy(&(idp_handler.rank_thread_handle[r].thr_control.wait_condition));
    }

    join_instance->RecordLogs();
}
