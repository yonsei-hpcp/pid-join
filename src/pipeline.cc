#include "idpHandler.hpp"

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
    void JoinInstance::AddJobToEndMap(rotate_n_stream_job_t *ended_job)
    {
        if (pthread_mutex_lock(&this->job_done_mutex) != 0)
        {
            printf("%sError pthread Lock Error. %s:%d\n", KRED, __FILE__, __LINE__);
            exit(-1);
        }

        IDPHandler *dpuhandler = (IDPHandler *)ended_job->idp_handler;

        bool status_src = this->UpdateJobStatus(ended_job->src_rank);
        bool status_dst = this->UpdateJobStatus(ended_job->dst_rank);

        if (status_src) // num_rank_allocated * 2 becuase each rank has 2 jobs, send and recieve. - Comment
        {
            this->WakeUpQueueLine(ended_job->src_rank);
        }

        if (status_dst)
        {
            this->WakeUpQueueLine(ended_job->dst_rank);
        }

        pthread_mutex_lock(&thread_queue_mutex);

        dpuhandler->rnc_job_q->ROTATE_AND_STREAM_JOBS--;
#ifdef RNS_NEW_QUE
        int src_ch = dpuhandler->GetChannelID(ended_job->src_rank);
        int dst_ch = dpuhandler->GetChannelID(ended_job->dst_rank);
        dpuhandler->m_jobs_per_ch[src_ch]--;
        dpuhandler->m_jobs_per_ch[dst_ch]--;
#endif

        if (dpuhandler->rnc_job_q->ROTATE_AND_STREAM_JOBS == 0)
        {
            this->ResetQueueLine();
        }

        pthread_mutex_unlock(&thread_queue_mutex);

        pthread_mutex_unlock(&this->job_done_mutex);
    }

    int IDPHandler::ProceedQueryPlan(
        JoinOperator *join_operator,
        int rank_id, /* rank_ids only contains one element which is rank id of this thread. */
        DPUKernelParams_t *param)
    {
        int progress = 0;

        if (join_operator == NULL)
        {
            printf("ERROR: join_operator is NULL\n");
            return 0;
        }

        std::vector<TimeStamp_t> perThreadTimeStamps_detail;

        // std::cout << "curr_query_tree->size(): " << curr_query_tree->size() << std::endl;

        for (; progress < curr_query_tree->size(); progress += 1)
        {
            Json::Value &query_node = (*curr_query_tree)[progress];
            std::string operator_name = query_node["operator"].asString();

            std::vector<TimeStamp_t> perThreadTimeStamps;
            perThreadTimeStamps.push_back(TimeStamp_t{rank_id, query_node, ""});

            if (operator_name_map.find(operator_name) == operator_name_map.end())
            {
                printf("Operand %s Not exists.\n", operator_name.c_str());
                return -1;
            }

            int operator_type = operator_name_map[operator_name];

            if (rank_id == 0)
                printf("Execute %s %d\n", operator_name.c_str(), progress);

            int num_prev_detail = perThreadTimeStamps_detail.size();
            clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps.back().start_time);

            switch (operator_type)
            {
                /////////////////////////////
                // DPU Functions
                ////////////////////////////
            case DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING:
                join_operator->Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_GLOBAL_HASH_PARTITIONING:
                join_operator->Execute_GLOBAL_HASH_PARTITIONING(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE:
                join_operator->Execute_PHJ_BUILD_HASH_TABLE(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE:
                join_operator->Execute_PHJ_PROBE_HASH_TABLE_INNER(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_MPSM_JOIN_PARTITIONING:
                join_operator->Execute_MPSM_JOIN_PARTITIONING(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_MPSM_JOIN_PROBE:
                join_operator->Execute_MPSM_JOIN_PROBE(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_MPSM_JOIN_SORT:
                join_operator->Execute_MPSM_JOIN_SORT(this, rank_id, param, &query_node);
                break;
            case DPU_FUNC_MPSM_JOIN_PROBE_ALL:
                join_operator->Execute_MPSM_JOIN_PROBE_ALL(this, rank_id, param, &query_node);
                break;
            /////////////////////////////
            // Host Functions
            /////////////////////////////
            case HOST_FUNC_ROTATE_AND_STREAM:
                join_operator->Execute_HOST_FUNC_ROTATE_AND_STREAM(this, rank_id, param, &query_node);
                break;
            case HOST_FUNC_CALCULATE_PAGE_HISTOGRAM:
                join_operator->Execute_MT_SAFE_HOST_FUNC_CALCULATE_PAGE_HISTOGRAM(this, rank_id, param, &query_node);
                break;
            case HOST_FUNC_INVALIDATE_STACKNODE:
                join_operator->Execute_HOST_FUNC_INVALIDATE_STACKNODE(this, rank_id, param, &query_node);
                break;
            case HOST_FUNC_LOAD_COLUMN:
                join_operator->Execute_HOST_FUNC_LOAD_COLUMN(this, rank_id, param, &query_node);
                break;
            case HOST_FUNC_SEND_DATA_OPT:
                join_operator->Execute_HOST_FUNC_SEND_DATA_OPT(this, rank_id, param, &query_node);
                break;
            case HOST_FUNC_RECV_DATA_OPT:
                join_operator->Execute_HOST_FUNC_RECV_DATA_OPT(this, rank_id, param, &query_node);
                break;

            /////////////////////////////
            // Control Functions
            /////////////////////////////
            case CONTROL_FUNC_SYNC_THREADS:
                join_operator->Execute_CONTROL_FUNC_SYNC_THREADS(this, rank_id, param, &query_node);
                break;
                /////////////////////////////
                // Compound Functions
                /////////////////////////////
            case COMPOUND_FUNC_RNS_JOIN:
                join_operator->Execute_COMPOUND_FUNC_RNS_JOIN(this, rank_id, param, &query_node, &perThreadTimeStamps_detail);
                break;
            case COMPOUND_FUNC_GLB_PARTITION:
                join_operator->Execute_COMPOUND_FUNC_GLB_PARTITION(this, rank_id, param, &query_node);
                break;
            default:
                printf("%sError: Not Available Execution Node %s.\n", KCYN, operator_name.c_str());
                exit(-1);
                break;
            }

            clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps.back().end_time);

            if (num_prev_detail == perThreadTimeStamps_detail.size())
                perThreadTimeStamps_detail.insert(perThreadTimeStamps_detail.end(), perThreadTimeStamps.begin(), perThreadTimeStamps.end());

            join_operator->Execute_CONTROL_FUNC_SYNC_THREADS(this, rank_id, param, &query_node);
        }
        join_operator->join_instance->CollectLog(perThreadTimeStamps_detail);

        return 0;
    }

    /* Main Thread Function of Rankwise Pipelining */
    void *rank_pipeline_execution_thr_func(void *arg)
    {
        rank_thread_handle_t *rank_thread_handle_arg = (rank_thread_handle_t *)arg;

        pthread_mutex_lock(&(rank_thread_handle_arg->thr_control.wait_mutex));
        pthread_cond_wait(&(rank_thread_handle_arg->thr_control.wait_condition), &(rank_thread_handle_arg->thr_control.wait_mutex));
        pthread_mutex_unlock(&(rank_thread_handle_arg->thr_control.wait_mutex));

        DPUKernelParams_t params;
        params.resize(NUM_DPU_RANK);
        JoinOperator *curr_join_operator = (JoinOperator *)rank_thread_handle_arg->join_operator;

        int ret = rank_thread_handle_arg->idp_handler->ProceedQueryPlan(
            curr_join_operator, rank_thread_handle_arg->rank_id, &params); // Start proceeding query plans - Comment

        if (ret != 0)
        {
            printf("%sError: Not allowed return value form ProceedQueryPlan\n", KCYN);
        }

        rank_thread_handle_arg->curr_status = RANK_THREAD_STATUS_JOB_DONE;

        rank_thread_handle_arg->idp_handler->SendPipelineDoneSignal();
        return NULL;
    }

    void IDPHandler::SendPipelineDoneSignal()
    {
        pthread_mutex_lock(&(this->master_thr_mutex));
        pthread_cond_signal(&(this->master_thr_condition));
        pthread_mutex_unlock(&(this->master_thr_mutex));
        // printf("Send Signal\n");
    }

    void IDPHandler::AllocateRankPipeline(std::vector<int> &rank_ids, int num)
    {
        this->AllocateRank(rank_ids, num);

        printf("Ranks to use: %d\n", rank_ids.size());

        for (int r = 0; r < rank_ids.size(); r++)
        {
            this->rank_thread_handle[r].idp_handler = this;
            pthread_mutex_init(&(this->rank_thread_handle[r].thr_control.wait_mutex), NULL);
            pthread_cond_init(&(this->rank_thread_handle[r].thr_control.wait_condition), NULL);
            this->rank_thread_handle[r].run_signal = RANK_THREAD_SIGNAL_NEXT_OP;
            this->rank_thread_handle[r].rank_id = r;

            // std::cout << "AllocateRankPipeline rank_pipeline_execution_thr_func Created" << std::endl;
            pthread_create(
                this->rank_threads + r,
                NULL,
                rank_pipeline_execution_thr_func,
                this->rank_thread_handle + r);
        }
    }

    int IDPHandler::AddRNSJob(const rotate_n_stream_job_t *new_job, bool lock)
    {
        if (lock)
        {
            int ret = pthread_mutex_lock(&(this->rnc_job_q->mutex));
            if (ret != 0)
            {
                printf("%s %d Error pthread Lock Error. %s:%d\n", KRED, ret, __FILE__, __LINE__);
                exit(-1);
            }
        }

        rnc_job_q->rnc_nqueue_add_job(this->rnc_job_q, new_job);

        pthread_cond_signal(&(this->rnc_job_q->cond));

        if (lock)
        {
            pthread_mutex_unlock(&(this->rnc_job_q->mutex));
        }
        return 0;
    }

    void IDPHandler::SetJoinInstanceOnRNSJobQueue(JoinInstance *instance)
    {
        this->rnc_job_q->join_instance = (void *)instance;
    }

    static void PrintQ(RNS_Job_Queue_t *job_queue)
    {
        rotate_n_stream_job_t *temp = job_queue->head_job;
        while (temp != NULL)
        {
            if (temp == job_queue->tail_job)
            {
                printf("%p", temp);
            }
            else
            {
                printf("%p --> ", temp);
            }
            temp = temp->next_job;
        }
        printf("\n");
    }

    static int counter = 0;
    rotate_n_stream_job_t *rnc_queue_next_job_host(RNS_Job_Queue_t *job_queue)
    {
        while (true)
        {
            if (pthread_mutex_lock(&(job_queue->mutex)) != 0)
            {
                printf("%sError pthread Lock Error. %s:%d\n", KRED, __FILE__, __LINE__);
                exit(-1);
            }

            rotate_n_stream_job_t *ret;

            if (job_queue->job_done == 1)
            {
                pthread_mutex_unlock(&(job_queue->mutex));
                return NULL;
            }

            RNS_job_priority_queue_t *job_q = (RNS_job_priority_queue_t *)(job_queue->job_queue_stl_ptr);

            if (job_q->empty())
            {
                pthread_cond_wait(&(job_queue->cond), &(job_queue->mutex));
                pthread_mutex_unlock(&(job_queue->mutex));
                continue;
            }

            ret = job_q->top();
            job_q->pop();
            IDPHandler *dpuhandler = (IDPHandler *)ret->idp_handler;
#ifdef RNS_NEW_QUE
            dpuhandler->m_jobs_per_ch[job_queue->channel_ids[ret->src_rank]]++;
            dpuhandler->m_jobs_per_ch[job_queue->channel_ids[ret->dst_rank]]++;
#endif

            pthread_mutex_unlock(&(job_queue->mutex));

            return ret;
        }
    }

    void return_job_host(struct RNS_Job_Queue_t *job_queue, const rotate_n_stream_job_t *ended_job_)
    {
        rotate_n_stream_job_t *ended_job = (rotate_n_stream_job_t *)ended_job_;
        JoinInstance *instance = (JoinInstance *)job_queue->join_instance;

        instance->AddJobToEndMap(ended_job);
    }

    void rnc_nqueue_add_job_host(RNS_Job_Queue_t *job_queue, const rotate_n_stream_job_t *new_job_)
    {
        // printf("JOB ADDED: %lf mram_src_offset:%d mram_dst_offset(in packet):%d src_packet_num:%d src_rank:%d dst_rank:%d\n",
        //     new_job_->job_priority, new_job_->mram_src_offset, new_job_->mram_dst_offset / 128, new_job_->src_packet_num, new_job_->src_rank, new_job_->dst_rank);

        rotate_n_stream_job_t *new_job = (rotate_n_stream_job_t *)new_job_;
        RNS_job_priority_queue_t *job_q = (RNS_job_priority_queue_t *)(job_queue->job_queue_stl_ptr);
        job_q->push(new_job);
    }

    int IDPHandler::InitRNS()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
        // pthread_mutexattr_
        if (pthread_mutex_init(&(this->rnc_job_q->mutex), &attr) != 0)
        {
            std::cout << "ERROR pthread_mutex_init(&(this->rnc_job_q->mutex), &attr) Failed\n";
            exit(-1);
        }
        if (pthread_cond_init(&(this->rnc_job_q->cond), NULL) != 0)
        {
            std::cout << "ERROR pthread_cond_init(&(this->rnc_job_q->cond), NULL) Failed\n";
            exit(-1);
        }

        printf("this->dpu_pairs.size(): %d\n", this->dpu_pairs.size());

        for (int i = 0; i < this->dpu_pairs.size(); i++)
        {
            std::pair<dpu_set_t *, pidjoin::IDPMemoryMgr *> &pair = this->dpu_pairs.at(i);
            dpu_set_t *set = pair.first;
            this->rnc_job_q->rank_sets[i] = set;
        }

        this->rnc_job_q->num_ranks = this->dpu_pairs.size();
        this->rnc_job_q->job_done = 0;
        this->rnc_job_q->rnc_queue_next_job = &rnc_queue_next_job_host;
        this->rnc_job_q->rnc_nqueue_add_job = &rnc_nqueue_add_job_host;
        this->rnc_job_q->return_job = &return_job_host;

        this->rnc_job_q->head_job = NULL; // new rotate_n_stream_job_t;
        this->rnc_job_q->tail_job = NULL; // new rotate_n_stream_job_t;

        this->rnc_job_q->curr_job_num = 0;

        this->rnc_job_q->total__ = 0;
        this->rnc_job_q->curr_priority__ = 0;
        this->rnc_job_q->ROTATE_AND_STREAM_JOBS = 0;

        // this->rnc_job_q->job_queue_stl_ptr = new std::queue<rotate_n_stream_job_t *>();
        this->rnc_job_q->job_queue_stl_ptr = new RNS_job_priority_queue_t();

        DPU_ASSERT(make_rnc_threads(this->rnc_job_q));

        return 0;
    }
}
