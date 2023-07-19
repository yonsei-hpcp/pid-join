#include "idpHandler.hpp"

#include "join_internals.hpp"

#include <algorithm>
#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#define NUM_DPU_RANK 64

std::string base_json = "{                                        \
    \"query_name\": \"\",                                         \
    \"tables\": [],                                               \
    \"query_tree\": [                                             \
        {                                                         \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",            \
            \"buffer_name\": \"left\"                             \
        },                                                        \
        {                                                         \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",            \
            \"buffer_name\": \"right\"                            \
        },                                                        \
        {                                                         \
            \"operator\": \"CONTROL_FUNC_SYNC_THREADS\"           \
        },                                                        \
        {                                                         \
            \"operator\": \"COMPOUND_FUNC_GLB_PARTITION\",        \
            \"inputs\": [                                         \
                \"unused\",                                       \
                \"left\"                                          \
            ],                                                    \
            \"outputs\": [                                        \
                \"Build_arr\",                                    \
                \"Build_hist\"                                    \
            ],                                                    \
            \"packet_size\": 8,                                   \
            \"partition_type\": 1                                 \
        },                                                        \
        {                                                         \
            \"operator\": \"COMPOUND_FUNC_GLB_PARTITION\",        \
            \"inputs\": [                                         \
                \"unused\",                                       \
                \"right\"                                         \
            ],                                                    \
            \"outputs\": [                                        \
                \"Probe_arr\",                                    \
                \"Probe_hist\"                                    \
            ],                                                    \
            \"packet_size\": 8,                                   \
            \"partition_type\": 1                                 \
        },                                                        \
        {                                                         \
            \"operator\": \"COMPOUND_FUNC_RNS_JOIN\",             \
            \"inputs\": [                                         \
                \"Build_arr\",                                    \
                \"Build_hist\",                                   \
                \"Probe_arr\",                                    \
                \"Probe_hist\"                                    \
            ],                                                    \
            \"outputs\": [                                        \
                \"join_result\"                                   \
            ],                                                    \
            \"packet_size\": 8,                                   \
            \"join_type\": \"\",                                  \
            \"comment\": \"first join\"                           \
        },                                                        \
        {                                                         \
            \"operator\": \"HOST_FUNC_RECV_DATA_OPT\",            \
            \"node_name\": \"join_result\"                        \
        }                                                         \
    ],                                                            \
    \"query_string\": []                                          \
}";

namespace pidjoin
{
#ifdef RNS_NEW_QUE
    bool RNS_job_priority_queue_t::compare(rotate_n_stream_job_t *a, rotate_n_stream_job_t *b)
    {
        IDPHandler *idp_handler = (IDPHandler *)a->idp_handler;
        int a_src_ch = idp_handler->GetChannelID(a->src_rank);
        int a_dst_ch = idp_handler->GetChannelID(a->dst_rank);
        int b_src_ch = idp_handler->GetChannelID(b->src_rank);
        int b_dst_ch = idp_handler->GetChannelID(b->dst_rank);

        int a_load = idp_handler->m_jobs_per_ch[a_src_ch] + idp_handler->m_jobs_per_ch[a_dst_ch];
        int b_load = idp_handler->m_jobs_per_ch[b_src_ch] + idp_handler->m_jobs_per_ch[b_dst_ch];
        return (a_load >= b_load); // execute job with lower channel load first
    }

    void RNS_job_priority_queue_t::push(rotate_n_stream_job_t *job)
    {
        _list.push_back(job);
    }

    rotate_n_stream_job_t *RNS_job_priority_queue_t::top()
    {
        rotate_n_stream_job_t *top_job_ptr;
        if (_type == "chBal")
        {
            auto smallest = _list.begin();
            for (auto it = _list.begin(); it != _list.end(); it++)
            {
                if (compare(*smallest, *it))
                    smallest = it;
            }
            _top_elem = smallest;
            top_job_ptr = *smallest;
        }
        else if (_type == "jobFirst")
        {
            auto prev_job = _list.begin();
            for (auto it = _list.begin(); it != _list.end(); it++)
            {
                if (_prev_pop.first == (*it)->src_rank && _prev_pop.second == (*it)->dst_rank)
                {
                    prev_job = it;
                }
            }
            _prev_pop.first = (*prev_job)->src_rank;
            _prev_pop.second = (*prev_job)->dst_rank;
            _top_elem = prev_job;
            top_job_ptr = *prev_job;
        }
#ifdef DEBUG_DPULOG
        printf("[%s] %d->%d\n", __func__, top_job_ptr->src_rank, top_job_ptr->dst_rank);
#endif
        return top_job_ptr;
    }

    void RNS_job_priority_queue_t::pop()
    {
        _list.erase(_top_elem);
    }

    bool RNS_job_priority_queue_t::empty()
    {
        return _list.empty();
    }

#endif

    /////////////////////////////////////////////////////
    // StackInfo
    /////////////////////////////////////////////////////

    void IDPMemoryMgr::RemoveNode(const char *col_name)
    {
        StackNode *node = NULL;
        int erase_idx = -1;
        for (int j = 0; j < this->stack_info.nodes.size(); j++)
        {
            std::string col_nam = col_name;
            if (col_nam == this->stack_info.nodes[j]->name)
            {
                node = this->stack_info.nodes.at(j);
                erase_idx = j;
                break;
            }
        }

        if (node == NULL)
        {
            PrintCurrentMramStatus();
            printf("%sCannot Find Node: %s\n", KRED, col_name);
            exit(-1);
        }

        /////////////////////////////////////////////////////////////////////////////////////
        // Update Free Space indexes

        int start_byte = node->start_byte;
        int prev_block_byte = node->block_byte;

        int num_indexes = this->mram_space_idxs.size();

        int curr_idx = -1;

        for (int i = 0; i < (num_indexes - 1); i++)
        {
            int start_byte_ = this->mram_space_idxs[i];

            if (start_byte_ == start_byte)
            {
                curr_idx = i;
                break;
            }
        }

        if (curr_idx == -1)
        {
            printf("%sError: Cannot Find Node for Removal %d.\n", KRED, start_byte);
            exit(-1);
        }

        // validate
        if (this->mram_space_idxs[curr_idx] != start_byte)
        {

            printf("%sError1: this->mram_space_idxs[curr_idx]: %d != start_byte %d.\n",
                   KRED, this->mram_space_idxs[curr_idx], start_byte);
            exit(-1);
        }

        if (this->mram_space_idxs[curr_idx + 1] != (start_byte + node->block_byte))
        {
            printf("%sError2: nodename:%s this->mram_space_idxs[curr_idx]:%d this->mram_space_idxs[curr_idx+1]:%d != start_byte + node->block_byte %d.\n",
                   KRED, node->name.c_str(), this->mram_space_idxs[curr_idx], this->mram_space_idxs[curr_idx + 1], start_byte + node->block_byte);
            PrintCurrentMramStatus();
            exit(-1);
        }

        bool prev_free;
        bool post_free = this->mram_free_space_indicates[curr_idx + 1].first;

        if (curr_idx != 0)
        {
            prev_free = this->mram_free_space_indicates[curr_idx - 1].first;
        }
        else
        {
            prev_free = false;
        }

        // Case 1 이전 space가 free space 일 떄 + 다음 space가 free space 일 때,
        if ((prev_free == true) && (post_free == true))
        {
            // x1 x2 x3 x4 => x1 x4
            // t f t  => t
            this->mram_space_idxs.erase(mram_space_idxs.begin() + curr_idx + 1);
            this->mram_space_idxs.erase(mram_space_idxs.begin() + curr_idx);

            this->mram_free_space_indicates.erase(mram_free_space_indicates.begin() + curr_idx + 1);
            this->mram_free_space_indicates.erase(mram_free_space_indicates.begin() + curr_idx);
        }

        // Case 2 다음 space만 free space 일 때,
        else if ((prev_free == false) && (post_free == true))
        {
            // x1 x2 x3 x4 => x1 x2 x4
            // f f t  => f t
            this->mram_space_idxs.erase(mram_space_idxs.begin() + curr_idx + 1);

            this->mram_free_space_indicates.erase(mram_free_space_indicates.begin() + curr_idx);
        }

        // Case 3 이전 space만 free space 일 때,
        else if ((prev_free == true) && (post_free == false))
        {
            // x1 x2 x3 x4 => x1 x3 x4
            // t f f  => t f
            this->mram_space_idxs.erase(mram_space_idxs.begin() + curr_idx);

            this->mram_free_space_indicates.erase(mram_free_space_indicates.begin() + curr_idx);
        }

        // Case 4 둘다 아닐 때,
        else if ((prev_free == false) && (post_free == false))
        {
            // x1 x2 x3 x4 => x1 x2 x3 x4
            // f f f  => f t f

            this->mram_free_space_indicates[curr_idx].first = true;
            this->mram_free_space_indicates[curr_idx].second = nullptr;
        }

        // Done
        delete node;
        this->stack_info.nodes.erase(this->stack_info.nodes.begin() + erase_idx);
    }

    struct StackNode *IDPMemoryMgr::FindNode(const char *col_name)
    {
        pthread_mutex_lock(&mem_mgr_lock);

        for (int j = 0; j < this->stack_info.nodes.size(); j++)
        {
            StackNode *node = this->stack_info.nodes.at(j);

            std::string col_nam = col_name;
            if (col_nam == this->stack_info.nodes[j]->name)
            {
                auto &dd = this->stack_info.nodes[j];
                pthread_mutex_unlock(&mem_mgr_lock);
                return dd;
            }
        }

        PrintCurrentMramStatus();
        printf("%sCannot Find Node : %s\n", KRED, col_name);
        exit(-1);
        return NULL;
    }

    struct StackNode *IDPMemoryMgr::FindNode(uint32_t offset)
    {
        pthread_mutex_lock(&mem_mgr_lock);
        for (int j = 0; j < this->stack_info.nodes.size(); j++)
        {
            if (offset == this->stack_info.nodes[j]->start_byte)
            {
                auto &dd = this->stack_info.nodes[j];
                pthread_mutex_unlock(&mem_mgr_lock);
                return dd;
            }
        }

        printf("%sCannot Find Node offset %u\n", KRED, offset);
        exit(-1);
        return NULL;
    }

    /**
     * @brief returns the compatible address
     *
     * @param block_byte
     * @return int
     */
    int IDPMemoryMgr::FindEmptySpace(StackNode *new_stacknode, int mram_align_size, int block_byte)
    {
        int num_idxes = this->mram_space_idxs.size();

        int base_candidate_start_address = -1;
        int candidate_start_address = -1;
        int candidate_idx = -1;
        int candidate_empty_space = 64 * 1024 * 1024;

        if (block_byte == -1)
        {
            candidate_empty_space = 0;

            for (int i = 0; i < (num_idxes - 1); i++)
            {
                if (mram_free_space_indicates[i].first)
                {
                    int base_start_address = this->mram_space_idxs[i];
                    int left_over = base_start_address % mram_align_size;
                    int start_addresss = base_start_address + ((mram_align_size - left_over) % mram_align_size);
                    int empty_size = this->mram_space_idxs[i + 1] - start_addresss;

                    int diff = empty_size;
                    if (diff > 0)
                    {
                        if ((diff) > candidate_empty_space)
                        {
                            candidate_empty_space = diff;
                            base_candidate_start_address = base_start_address;
                            candidate_start_address = start_addresss;
                            candidate_idx = i;
                        }
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < (num_idxes - 1); i++)
            {
                if (mram_free_space_indicates[i].first)
                {
                    int base_start_address = this->mram_space_idxs[i];
                    int left_over = base_start_address % mram_align_size;
                    int start_addresss = base_start_address + ((mram_align_size - left_over) % mram_align_size);
                    int empty_size = this->mram_space_idxs[i + 1] - start_addresss;

                    int diff = empty_size - block_byte;

                    if (diff > 0)
                    {
                        if ((diff) < candidate_empty_space)
                        {
                            candidate_empty_space = diff;
                            base_candidate_start_address = base_start_address;
                            candidate_start_address = start_addresss;
                            candidate_idx = i;
                        }
                    }
                }
            }
        }

        if (block_byte == -1)
        {
            block_byte = 8;
        }

        // No More empty Space
        if (candidate_start_address == -1)
        {
            printf("block_byte: %d\n", block_byte);
            PrintCurrentMramStatus();
            printf("%sNo More Empty Space on MRAM Error.\n", KRED);
            exit(-1);
        }
        else
        {
            if ((candidate_start_address + block_byte) == this->mram_space_idxs[candidate_idx + 1])
            {
                if ((base_candidate_start_address == candidate_start_address))
                {
                    if (this->mram_free_space_indicates[candidate_idx].first == false)
                    {
                        printf("%sError Stack Management.\n", KRED);
                        exit(-1);
                    }
                    this->mram_free_space_indicates[candidate_idx].first = false;
                    this->mram_free_space_indicates[candidate_idx].second = new_stacknode;
                }
                else
                {
                    this->mram_space_idxs.insert(mram_space_idxs.begin() + candidate_idx + 1, candidate_start_address);
                    this->mram_free_space_indicates.insert(mram_free_space_indicates.begin() + candidate_idx + 1, std::make_pair(false, new_stacknode));
                }
            }
            else
            {
                this->mram_space_idxs.insert(mram_space_idxs.begin() + candidate_idx + 1, candidate_start_address + block_byte);
                this->mram_free_space_indicates.insert(mram_free_space_indicates.begin() + candidate_idx + 1, std::make_pair(true, nullptr));

                if ((base_candidate_start_address == candidate_start_address))
                {
                    if (this->mram_free_space_indicates[candidate_idx].first == false)
                    {
                        printf("%sError Stack Management.\n", KRED);
                        exit(-1);
                    }
                    this->mram_free_space_indicates[candidate_idx].first = false;
                    this->mram_free_space_indicates[candidate_idx].second = new_stacknode;
                }
                else
                {
                    this->mram_space_idxs.insert(mram_space_idxs.begin() + candidate_idx + 1, candidate_start_address);
                    this->mram_free_space_indicates.insert(mram_free_space_indicates.begin() + candidate_idx + 1, std::make_pair(false, new_stacknode));
                }
            }
        }

        return candidate_start_address;
    }

    struct StackNode *IDPMemoryMgr::PushStackNodeAligned(
        const char *name, int *data_bytes, int block_byte, int mram_align_size, DataType dt, int dt_size)
    {
        pthread_mutex_lock(&mem_mgr_lock);

        StackNode *new_stackNode = new StackNode();

        if (block_byte == 0)
        {
            printf("%sError! name:%s block_byte: %d |||| block_byte is zero\n",
                   KRED, name, block_byte);
            exit(-1);
        }
        int start_addrss = FindEmptySpace(new_stackNode, mram_align_size, block_byte);

        // printf("start_addrss: %d block_bytes: %d\n", start_addrss, block_byte);
        this->stack_info.nodes.push_back(new_stackNode);
        auto new_node = this->stack_info.nodes.back();
        new_node->name.assign(name);
        new_node->data_type = dt;
        new_node->data_type_size = dt_size;
        new_node->start_byte = start_addrss;

        if (data_bytes != NULL)
        {
            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                if (data_bytes[dpu] > block_byte)
                {
                    printf("%sError! name:%s new_node->data_bytes[dpu]:%d > block_byte: %d\n",
                           KRED, name, data_bytes[dpu], block_byte);
                    exit(-1);
                }
                else
                {
                    new_node->data_bytes[dpu] = data_bytes[dpu];
                }
            }
        }
        else
        {
            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                new_node->data_bytes[dpu] = block_byte;
            }
        }

        if (block_byte == -1)
        {
            new_node->block_byte = 8;
        }
        else
        {
            new_node->block_byte = block_byte;
        }

        pthread_mutex_unlock(&mem_mgr_lock);
        return new_stackNode;
    }

    struct StackNode *IDPMemoryMgr::PushStackNodeAligned(const char *name, int *data_bytes, int block_byte, int mram_align_size)
    {
        return PushStackNodeAligned(name, data_bytes, block_byte, mram_align_size, DataType::NOT_SPECIFIED, -1);
    }

    struct StackNode *IDPMemoryMgr::PushStackNode(const char *name, int *data_bytes, int block_byte)
    {
        printf("%sDeprecated!\n", KCYN);
        exit(-1);
        return NULL;
    }

    /////////////////////////////////////////////////////
    // IDPHandler
    /////////////////////////////////////////////////////

#define MAN_RANK_NUM 16
    void IDPHandler::AllocateRank(std::vector<int> &rank_ids, int num)
    {
        int all_rank_id = 0;

        dpu_pairs.resize(num);

        for (int new_rank_id = 0; new_rank_id < (num) && all_rank_id < 40;)
        {
            struct dpu_set_t *dpu_set = (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
            IDPMemoryMgr *immgr = new IDPMemoryMgr();
            int alloced_dpus = 0;
            DPU_ASSERT(dpu_alloc_custom(NUM_DPU_RANK, NULL, dpu_set, &alloced_dpus));
            
            if (alloced_dpus == NUM_DPU_RANK)
            {

                    rank_ids.push_back(new_rank_id);
                    dpu_pairs[new_rank_id] = std::pair<struct dpu_set_t *, IDPMemoryMgr *>(dpu_set, immgr);
                    new_rank_id++;
            }
            else
            {
                printf("%d-th rank is defect: %d DPUs alloced.\n", all_rank_id, alloced_dpus);
            }
            all_rank_id++;
        }

        LoadProgram(std::ref(rank_ids), DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE);
        InitRNS();
    }

    void IDPHandler::LoadProgram(int rank_id, int param)
    {
        DPU_ASSERT(dpu_load(*(this->dpu_pairs[rank_id].first), param_binary_name_map[param].c_str(), NULL));
    }

    void IDPHandler::LoadProgram(std::vector<int> &rank_ids, int param)
    {
        if (rank_ids.size() == 1)
        {
            DPU_ASSERT(dpu_load(*(this->dpu_pairs[rank_ids[0]].first), pidjoin::param_binary_name_map[param].c_str(), NULL));
            return;
        }

        ////////////////////////////////////////////////////////////////////////////////

        std::thread *thrs[rank_ids.size()];

        for (int rank_id : rank_ids)
        {
            thrs[rank_id] = new std::thread([](int rank_id_, IDPHandler *idp_handler, int param_)
                                            { DPU_ASSERT(dpu_load(*(idp_handler->dpu_pairs[rank_id_].first), pidjoin::param_binary_name_map[param_].c_str(), NULL)); },
                                            rank_id, this, param);
        }

        for (int rank_id : rank_ids)
        {
            thrs[rank_id]->join();
            delete thrs[rank_id];
        }
    }

    int IDPHandler::GetEncodedData(std::string &attr_name, std::string &key_name)
    {
        int ret_val;
        pthread_mutex_lock(&encoding_table_mutex);

        // Find encoding table
        if (encoding_table_map.find(attr_name) != encoding_table_map.end())
        {
            ENCODING_TABLE_t *t = encoding_table_map.at(attr_name);
            if (t->find(key_name) != t->end())
            {
                ret_val = t->at(key_name);
            }
            else
            {
                printf("%sError! Cannot Find Encoding Elem: %s\n", KCYN, key_name.c_str());
                exit(-1);
            }
        }
        else
        {
            printf("%sError! Cannot Find Encoding Table: %s\n", KCYN, attr_name.c_str());
            exit(-1);
        }

        pthread_mutex_unlock(&encoding_table_mutex);
        return ret_val;
    }
    void IDPHandler::PrintStackInfo(std::vector<int> &rank_ids)
    {
        for (int rank_id : rank_ids)
        {
            this->PrintStackInfo(rank_id);
        }
    }

    void IDPHandler::PrintStackInfo(int rank_id)
    {
        int i = 0;

        for (auto &node : this->dpu_pairs[rank_id].second->stack_info.nodes)
        {
            printf("%s\n", node->name.c_str());
            printf("node.start_byte: %d\n", node->start_byte);
            printf("node.block_byte: %d\n", node->block_byte);
            printf("node.databytes\n");
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    printf("%2d:%8d | ", i * 8 + j, node->data_bytes[i * 8 + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    void IDPHandler::PrintStackInfoBrief(int rank_id)
    {
        int i = 0;

        for (auto &node : this->dpu_pairs[rank_id].second->stack_info.nodes)
        {
            printf("%s\n", node->name.c_str());
            printf("node.start_byte: %d\n", node->start_byte);
            printf("node.block_byte: %d\n", node->block_byte);
            // printf("node.databytes\n");
            // for (int i = 0; i < 8; i++)
            // {
            //     for (int j = 0; j < 8; j++)
            //     {
            //         printf("%2d:%8d | ", i * 8 + j, node->data_bytes[i * 8 + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
        }
    }

    void IDPHandler::PrintDPUInfo(void)
    {
        printf("this->dpu_pairs.size(): %ld\n", this->dpu_pairs.size());
        for (int i = 0; i < this->dpu_pairs.size(); i++)
        {
            // printf("%d,", i, this->dpu_pairs[i].first.kind);
        }
    }

    void IDPHandler::LoadParameter(int rank_id, DPUKernelParams_t &params_rank, int DPU_FUNC_TYPE)
    {
        int i = 0;
        struct dpu_set_t dpu;

        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(params_rank[i])));
            i++;
        }

        // printf("param_name_map[DPU_FUNC_TYPE].c_str()::%s\n", param_name_map[DPU_FUNC_TYPE].c_str());
        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_TO_DPU,
            param_name_map[DPU_FUNC_TYPE].c_str(),
            0,
            param_size_map[DPU_FUNC_TYPE],
            DPU_XFER_DEFAULT));
    }

    void IDPHandler::RunKernel(int rank_id)
    {
        // DPU_ASSERT(dpu_launch(*(this->dpu_pairs[rank_id].first), DPU_SYNCHRONOUS));
        DPU_ASSERT(dpu_launch(*(this->dpu_pairs[rank_id].first), DPU_ASYNCHRONOUS));
        DPU_ASSERT(dpu_sync(*(this->dpu_pairs[rank_id].first)));
    }

    void IDPHandler::RunKernelAsync(int rank_id)
    {
        DPU_ASSERT(dpu_launch(*(this->dpu_pairs[rank_id].first), DPU_ASYNCHRONOUS));
    }

    void IDPHandler::RunKernel(std::vector<int> &rank_ids)
    {
        for (int rank_id : rank_ids)
        {
            RunKernel(rank_id);
        }
    }

    void IDPHandler::WaitKernel(int rank_id)
    {
        DPU_ASSERT(dpu_sync(*(this->dpu_pairs[rank_id].first)));
    }

    void IDPHandler::WaitKernel(std::vector<int> &rank_ids)
    {
        for (auto rank_id : rank_ids)
        {
            this->WaitKernel(rank_id);
        }
    }

    void IDPHandler::ReadLog(int rank_id)
    {
        pthread_mutex_lock(&this->print_log_lock);
        struct dpu_set_t dpu;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
        pthread_mutex_unlock(&this->print_log_lock);
    }

    void IDPHandler::ReadLog(int rank_id, int dpu_id)
    {
#ifdef DEBUG_DPULOG
        struct dpu_set_t dpu;
        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            if (i == dpu_id)
            {
                DPU_ASSERT(dpu_log_read(dpu, stdout));
            }
            i++;
        }
#endif
    }

    void IDPHandler::HandleKernelResults(JoinInstance *join_instance, int rank_id, int DPU_FUNC_TYPE, DPUKernelParams_t &params)
    {
        HandleKernelResults(join_instance, rank_id, DPU_FUNC_TYPE, params, NULL);
    }

    void IDPHandler::HandleKernelResults(JoinInstance *join_instance, int rank_id, int DPU_FUNC_TYPE, DPUKernelParams_t &params, char *ptr)
    {
        switch (DPU_FUNC_TYPE)
        {
        case DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING:
        {
            packetwise_hash_local_partitioning_return_arg *return_arg = (packetwise_hash_local_partitioning_return_arg *)malloc(sizeof(packetwise_hash_local_partitioning_return_arg) * NUM_DPU_RANK);

            int i = 0;
            struct dpu_set_t dpu_set;

            DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu_set)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu_set, (void *)(return_arg + i)));
                i++;
            }

            DPU_ASSERT(dpu_push_xfer(
                *(this->dpu_pairs[rank_id].first),
                DPU_XFER_FROM_DPU,
                param_return_var_map[DPU_FUNC_TYPE].c_str(),
                0,
                param_return_size_map[DPU_FUNC_TYPE],
                DPU_XFER_DEFAULT));

            /////////////////////////////////////////////////////////////////////

            packetwise_hash_local_partitioning_arg *arg = (packetwise_hash_local_partitioning_arg *)(params[0]);

            int tuple_size = arg->tuple_size;

            StackNode *partitioned_result_node = this->FindNode(rank_id, arg->partitioned_result_start_byte);
            StackNode *partition_info_node = this->FindNode(rank_id, arg->result_partition_info_start_byte);

            int max_partition = 0;
            int32_t partition_bytes[NUM_DPU_RANK];

            int max_bytes = 0;
            int32_t elem_bytes[NUM_DPU_RANK];

            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                int64_t elem_num = return_arg[dpu].elem_num;
                // printf("elem_num:%ld\n", elem_num);
                if (ptr != NULL)
                {
                    ((int32_t *)ptr)[dpu] = return_arg[dpu].partition_num;
                }

                partition_bytes[dpu] = return_arg[dpu].partition_num * sizeof(int32_t);

                if (max_partition < partition_bytes[dpu])
                {
                    max_partition = partition_bytes[dpu];
                }

                elem_bytes[dpu] = elem_num * tuple_size;

                if (max_bytes < elem_bytes[dpu])
                {
                    max_bytes = elem_bytes[dpu];
                }
            }

            // printf("Local Partitioning %d %d/%d\n", rank_id, partitioned_result_node->block_byte/sizeof(int64_t), max_bytes/sizeof(int64_t));

            // this->UpdateStackNode(rank_id, generated_local_tid_node, elem_bytes, max_bytes);
            this->UpdateStackNode(rank_id, partitioned_result_node, elem_bytes, max_bytes);
            this->UpdateStackNode(rank_id, partition_info_node, partition_bytes, max_partition);
        }
        break;
        case DPU_FUNC_LOCAL_HASH_PARTITIONING:
        {
            hash_local_partitioning_return_arg *return_arg = (hash_local_partitioning_return_arg *)malloc(sizeof(hash_local_partitioning_return_arg) * NUM_DPU_RANK);

            int i = 0;
            struct dpu_set_t dpu_set;

            DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu_set)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu_set, (void *)(return_arg + i)));
                i++;
            }

            DPU_ASSERT(dpu_push_xfer(
                *(this->dpu_pairs[rank_id].first),
                DPU_XFER_FROM_DPU,
                param_return_var_map[DPU_FUNC_TYPE].c_str(),
                0,
                param_return_size_map[DPU_FUNC_TYPE],
                DPU_XFER_DEFAULT));

            /////////////////////////////////////////////////////////////////////

            hash_local_partitioning_arg *arg = (hash_local_partitioning_arg *)(params[0]);

            int tuple_size = arg->tuple_size;

            // StackNode* partitioned_result_node
            //     = this->FindNode(rank_id, arg->partitioned_result_start_byte);
            StackNode *partition_info_node = this->FindNode(rank_id, arg->result_partition_info_start_byte);

            int max_partition = 0;
            int32_t partition_bytes[NUM_DPU_RANK];

            int max_bytes = 0;
            int32_t elem_bytes[NUM_DPU_RANK];

            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                int64_t elem_num = return_arg[dpu].elem_num;
                // printf("elem_num:%ld\n", elem_num);
                if (ptr != NULL)
                {
                    ((int32_t *)ptr)[dpu] = return_arg[dpu].partition_num;
                }

                partition_bytes[dpu] = return_arg[dpu].partition_num * sizeof(int32_t);

                if (max_partition < partition_bytes[dpu])
                {
                    max_partition = partition_bytes[dpu];
                }

                elem_bytes[dpu] = elem_num * tuple_size;

                if (max_bytes < elem_bytes[dpu])
                {
                    max_bytes = elem_bytes[dpu];
                }
            }

            // printf("Local Partitioning %d %d/%d\n", rank_id, partitioned_result_node->block_byte/sizeof(int64_t), max_bytes/sizeof(int64_t));

            // this->UpdateStackNode(rank_id, generated_local_tid_node, elem_bytes, max_bytes);
            // this->UpdateStackNode(rank_id, partitioned_result_node, elem_bytes, max_bytes);
            this->UpdateStackNode(rank_id, partition_info_node, partition_bytes, max_partition);
        }
        break;
        case DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE:
        {
            hash_phj_probe_return_arg *return_arg = (hash_phj_probe_return_arg *)malloc(param_return_size_map[DPU_FUNC_TYPE] * NUM_DPU_RANK);

            int i = 0;
            struct dpu_set_t dpu_set;

            DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu_set)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu_set, (void *)(return_arg + i)));
                i++;
            }

            DPU_ASSERT(dpu_push_xfer(
                *(this->dpu_pairs[rank_id].first),
                DPU_XFER_FROM_DPU,
                param_return_var_map[DPU_FUNC_TYPE].c_str(),
                0,
                param_return_size_map[DPU_FUNC_TYPE],
                DPU_XFER_DEFAULT));

            /////////////////////////////////////////////////////////////////////

            hash_phj_probe_arg *arg = (hash_phj_probe_arg *)(params[0]);

            int tuple_size = sizeof(tuplePair_t);

            StackNode *result_s_node = this->FindNode(rank_id, arg->Result_offset);

            int max_partition = 0;
            int32_t partition_bytes[NUM_DPU_RANK];

            int max_bytes = 0;
            int32_t elem_bytes[NUM_DPU_RANK];

            int S = 0;
            int selected = 0;
#ifdef COLLECT_LOGS
            int miss_count = 0;
#endif

            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                int64_t result_size = return_arg[dpu].result_size;

                elem_bytes[dpu] = result_size * tuple_size;

                if (max_bytes < elem_bytes[dpu])
                {
                    max_bytes = elem_bytes[dpu];
                }

                selected += return_arg[dpu].result_size;
                S += arg[dpu].S_num;
#ifdef COLLECT_LOGS
                miss_count += return_arg[dpu].miss_count;
#endif
            }
#ifdef COLLECT_LOGS
            join_instance->CreateJoinLog(rank_id, S, selected, miss_count);
#endif

            this->UpdateStackNode(rank_id, result_s_node, elem_bytes, max_bytes);
        }
        break;
        case DPU_FUNC_MPSM_JOIN_PARTITIONING:
        {
            sort_merge_partitioning_return_arg *return_arg = (sort_merge_partitioning_return_arg *)malloc(param_return_size_map[DPU_FUNC_TYPE] * NUM_DPU_RANK);

            // Read Return Val
            int i = 0;
            struct dpu_set_t dpu_set;

            DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu_set)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu_set, (void *)(return_arg + i)));
                i++;
            }

            DPU_ASSERT(dpu_push_xfer(
                *(this->dpu_pairs[rank_id].first),
                DPU_XFER_FROM_DPU,
                param_return_var_map[DPU_FUNC_TYPE].c_str(),
                0,
                param_return_size_map[DPU_FUNC_TYPE],
                DPU_XFER_DEFAULT));

            // Update Stack Info

            // Find StackNode by Offset
            sort_merge_partitioning_arg *arg = (sort_merge_partitioning_arg *)(params[0]);

            StackNode *r_sorted_node = this->GetStackNode(rank_id, arg->r_sorted_start_byte);

            int max = 0;
            for (int dpu = 0; dpu < NUM_DPU_RANK; dpu++)
            {
                r_sorted_node->data_bytes[dpu] = return_arg[dpu].r_total_elem * sizeof(tuplePair_t);
                if (max < r_sorted_node->data_bytes[dpu])
                    max = r_sorted_node->data_bytes[dpu];
            }

            r_sorted_node->block_byte = max;
            free(return_arg);
        }
        break;
        default:
            break;
        }

        // free param buffer
        free((char *)(params[0]));

        // Error Checking
        switch (DPU_FUNC_TYPE)
        {
        case DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING:
        case DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING:
        case DPU_FUNC_PHJ_PROBE_HASH_TABLE:
        case DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE:
        case DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE:
        case DPU_FUNC_GLOBAL_HASH_PARTITIONING:
        case DPU_FUNC_GLB_PARTITION_PACKET:
        case DPU_FUNC_GLB_PARTITION_COUNT:
        case DPU_FUNC_LOCAL_HASH_PARTITIONING:
        case DPU_FUNC_NESTED_LOOP_JOIN:
        case DPU_FUNC_FINISH_JOIN:
        {
            dpu_results_t *return_arg = (dpu_results_t *)malloc(sizeof(dpu_results_t) * NUM_DPU_RANK);

            int i = 0;
            struct dpu_set_t dpu_set;

            DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu_set)
            {
                DPU_ASSERT(dpu_prepare_xfer(dpu_set, (void *)(return_arg + i)));
                i++;
            }

            DPU_ASSERT(dpu_push_xfer(
                *(this->dpu_pairs[rank_id].first),
                DPU_XFER_FROM_DPU,
                "dpu_results",
                0,
                sizeof(dpu_results_t),
                DPU_XFER_DEFAULT));

            bool error = false;

            // if (rank_id == 0)
            // {
            //     printf("%d: CYCLE: %lu TIME(ms): %f\n", DPU_FUNC_TYPE, return_arg[0].cycle_count, return_arg[0].cycle_count * 1.0 / 350000000.0 * 1000);
            // }

            for (int r = 0; r < NUM_DPU_RANK; r++)
            {
                if (return_arg[r].ERROR_TYPE_0 != 0)
                {
                    printf("%s: %d ERROR_TYPE_0:%d at R:%d D:%d\n",
                           KRED, DPU_FUNC_TYPE, return_arg[r].ERROR_TYPE_0, rank_id, r);
                    error = true;
                    break;
                }
                if (return_arg[r].ERROR_TYPE_1 != 0)
                {
                    printf("%s: %d ERROR_TYPE_1 %d at R:%d D:%d\n",
                           KRED, DPU_FUNC_TYPE, return_arg[r].ERROR_TYPE_1, rank_id, r);
                    error = true;
                    break;
                }
                if (return_arg[r].ERROR_TYPE_2 != 0)
                {
                    printf("%s: %d ERROR_TYPE_2 %d at R:%d D:%d\n",
                           KRED, DPU_FUNC_TYPE, return_arg[r].ERROR_TYPE_2, rank_id, r);
                    error = true;
                    break;
                }
                if (return_arg[r].ERROR_TYPE_3 != 0)
                {
                    printf("%s: %d ERROR_TYPE_3:%d at R:%d D:%d\n",
                           KRED, DPU_FUNC_TYPE, return_arg[r].ERROR_TYPE_3, rank_id, r);
                    error = true;
                    break;
                }
            }

            if (error)
            {
                dpu_pairs[rank_id].second->PrintCurrentMramStatus();
                // exit(-1);
            }

            free(return_arg);
        }
        break;

        default:
            break;
        }
    }

    // Retrieve data from ranks in rank_ids. Working thread # defined by UPMEM driver execute them - Comment
    int IDPHandler::RetrieveData(
        std::vector<int> &rank_ids,
        std::vector<std::vector<char *>> &retreive_buffers,
        std::vector<std::vector<int>> &data_bytes,
        const char *buffer_name)
    {
        if (rank_ids.size() == 1)
        {
            int read_size = RetrieveData(rank_ids[0], retreive_buffers[0], data_bytes[0], buffer_name);
            return 0;
        }
        /////////////////////////////////////////////////////////////////

        std::thread *thrs[rank_ids.size()];

        for (int rank_id : rank_ids)
        {
            thrs[rank_id] = new std::thread([](int rank_id_, std::vector<char *> &retreive_buffers_, std::vector<int> &data_bytes_, IDPHandler *idp_handler, const char *buffer_name_)
                                            { int read_size = idp_handler->RetrieveData(rank_id_, retreive_buffers_, data_bytes_, buffer_name_); },
                                            rank_id, std::ref(retreive_buffers[rank_id]), std::ref(data_bytes[rank_id]), this, buffer_name);
        }

        for (int rank_id : rank_ids)
        {
            thrs[rank_id]->join();
            delete thrs[rank_id];
        }

        return 0;
    }

    int IDPHandler::RetrieveDataManual(int rank_id, std::vector<char *> &retreive_buffers, const char *buffer_name, int byte_offset, int byte_size)
    {
        int buff_size;
        int offset;

        struct StackNode *node = this->FindNode(rank_id, buffer_name);

        buff_size = node->block_byte;
        offset = node->start_byte;

        struct dpu_set_t dpu;

        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(retreive_buffers[i])));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_FROM_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            offset + byte_offset,
            byte_size,
            DPU_XFER_DEFAULT));

        return buff_size;
    }

    int IDPHandler::RetrieveDataPreAlloced(int rank_id, std::vector<char *> &retreive_buffers, std::vector<int> &data_bytes, const char *buffer_name)
    {
        int buff_size;
        int offset;

        struct StackNode *node = this->FindNode(rank_id, buffer_name);

        buff_size = node->block_byte;
        offset = node->start_byte;

        data_bytes.clear();

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            data_bytes.push_back(node->data_bytes[i]);
        }

        struct dpu_set_t dpu;

        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(retreive_buffers[i])));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_FROM_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            offset,
            buff_size,
            DPU_XFER_DEFAULT));

        return buff_size;
    }

    int IDPHandler::SetRNSRank(int rank_id)
    {
        std::vector<char *> retreive_buffers;
        std::vector<int> data_bytes;

        retreive_buffers.resize(NUM_DPU_RANK);
        data_bytes.resize(NUM_DPU_RANK);

        int buff_size = 8;

        char *data = (char *)malloc(buff_size * NUM_DPU_RANK);

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            retreive_buffers[i] = data + i * buff_size;
        }

        struct dpu_set_t dpu;

        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(retreive_buffers[i])));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_FROM_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            0,
            buff_size,
            DPU_XFER_DEFAULT));

        free(data);

        return 0;
    }

    int IDPHandler::RetrieveData(
        int rank_id,
        std::vector<char *> &retreive_buffers,
        std::vector<int> &data_bytes,
        const char *buffer_name)
    {
        struct StackNode *node = this->FindNode(rank_id, buffer_name);

        int buff_size = node->block_byte;
        int offset = node->start_byte;
#ifdef DEBUG_DPULOG
        printf("[%s] Rank%d node->start_byte: %d buff_size: %d\n", __func__, rank_id, offset, buff_size);
#endif

        if (retreive_buffers[0] == NULL)
        {
            char *data = (char *)aligned_alloc(64, buff_size * NUM_DPU_RANK);

            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                retreive_buffers[i] = data + i * buff_size;
                data_bytes[i] = node->data_bytes[i];
            }
        }

        struct dpu_set_t dpu;

        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(retreive_buffers[i])));
            i++;
        }

        // printf("offset\t%d\t\n", offset);
        // printf("buff_size\t%d\t\n", buff_size);

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_FROM_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            offset,
            buff_size,
            DPU_XFER_DEFAULT));

        return buff_size;
    }

    int IDPHandler::StoreData(
        std::vector<int> &rank_ids,
        std::vector<std::vector<char *>> &buffers, // data to store to DPU in char list : buffers[Rank_id][DPU_id] - Comment
        std::vector<std::vector<int>> &data_bytes,
        std::vector<int> &block_size,
        const char *buffer_name)
    {
        std::thread *thrs[rank_ids.size()];

        for (int rank_id : rank_ids)
        {
#ifdef DEBUG_DPULOG
            printf("[%s] Rank%d buff_size: %d\n", __func__, rank_id, block_size[rank_id]);
#endif
            thrs[rank_id] = new std::thread(
                [buffer_name](
                    int rank_id_,
                    std::vector<char *> &buffers_,
                    std::vector<int> &data_bytes_,
                    int block_size,
                    IDPHandler *idp_handler)
                {
                    idp_handler->StoreData(
                        rank_id_,
                        buffers_,
                        data_bytes_,
                        block_size,
                        buffer_name);
                },
                rank_id, std::ref(buffers[rank_id]), std::ref(data_bytes[rank_id]), block_size[rank_id], this);
        }

        for (int rank_id : rank_ids)
        {
            thrs[rank_id]->join();
            delete thrs[rank_id];
        }

        return 0;
    }

    int IDPHandler::StoreDataAligned(
        int rank_id, std::vector<char *> &buffers,
        std::vector<int> &data_bytes, int block_size, const char *buffer_name, int align_byte_size)
    {
        int i = 0;
        struct dpu_set_t dpu;

        // Update Stack info
        StackNode *node = this->PushStackNodeAligned(rank_id, buffer_name, data_bytes.data(), block_size, align_byte_size);

        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(buffers[i])));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_TO_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            node->start_byte,
            block_size,
            DPU_XFER_DEFAULT));

        return 0;
    }

    int IDPHandler::StoreDataAligned(
        std::vector<int> &rank_ids, std::vector<std::vector<char *>> &buffers,
        std::vector<std::vector<int>> &data_bytes, std::vector<int> &block_size, const char *buffer_name, int align_byte_size)
    {
        std::thread *thrs[rank_ids.size()];

        for (int rank_id : rank_ids)
        {
            thrs[rank_id] = new std::thread(
                [buffer_name](
                    int rank_id_,
                    std::vector<char *> &buffers_,
                    std::vector<int> &data_bytes_,
                    int block_size_,
                    int align_byte_size_,
                    IDPHandler *idp_handler)
                {
                    idp_handler->StoreDataAligned(
                        rank_id_,
                        buffers_,
                        data_bytes_,
                        block_size_,
                        buffer_name, align_byte_size_);
                },
                rank_id, std::ref(buffers[rank_id]), std::ref(data_bytes[rank_id]), block_size[rank_id], align_byte_size, this);
        }

        for (int rank_id : rank_ids)
        {
            thrs[rank_id]->join();
            delete thrs[rank_id];
        }

        return 0;
    }

    int IDPHandler::RotNCopy(
        rotate_n_stream_job_t *job)
    {

        // // job->src_rank_dpu_set = &(this->dpu_pairs[job->src_rank].first);
        // // job->dst_rank_dpu_set = &(this->dpu_pairs[job->dst_rank].first);

        // pthread_t thread;
        // pthread_attr_t attr;

        return 0;
    }

    int IDPHandler::StoreData(
        int rank_id,
        std::vector<char *> &buffers,
        std::vector<int> &data_bytes,
        int block_size,
        const char *buffer_name)
    {
        int i = 0;
        struct dpu_set_t dpu;

        // Update Stack info
        StackNode *node = this->PushStackNode(rank_id, buffer_name, data_bytes.data(), block_size);

        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)(buffers[i])));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_TO_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            node->start_byte,
            block_size,
            DPU_XFER_DEFAULT));

        return 0;
    }

    void IDPHandler::ClearMRAM(int rank_id)
    {
        struct dpu_set_t dpu;
        char *memset_buffer = (char *)malloc(63 * 1024 * 1024);
        memset((void *)memset_buffer, 0, 63 * 1024 * 1024);

        int i = 0;
        DPU_FOREACH(*(this->dpu_pairs[rank_id].first), dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)memset_buffer));
            i++;
        }

        DPU_ASSERT(dpu_push_xfer(
            *(this->dpu_pairs[rank_id].first),
            DPU_XFER_TO_DPU,
            DPU_MRAM_HEAP_POINTER_NAME,
            0,
            63 * 1024 * 1024,
            DPU_XFER_DEFAULT));

        free((void *)memset_buffer);
    }

    void IDPHandler::ClearMRAM(std::vector<int> &rank_ids)
    {

        char *memset_buffer = (char *)malloc(63 * 1024 * 1024);
        memset((void *)memset_buffer, 0, 63 * 1024 * 1024);

        std::thread *thrs[rank_ids.size()];

        for (int rank_id : rank_ids)
        {

            thrs[rank_id] = new std::thread([](int rank_id_, char *memset_buffer, struct dpu_set_t dpu_set)
                                            {
                struct dpu_set_t dpu;
                int i = 0;

                DPU_FOREACH(dpu_set, dpu)
                {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)memset_buffer));
                    i++;
                }

                DPU_ASSERT(dpu_push_xfer(
                dpu_set,
                DPU_XFER_TO_DPU,
                DPU_MRAM_HEAP_POINTER_NAME,
                0,
                63 * 1024 * 1024,
                DPU_XFER_DEFAULT)); },
                                            rank_id, memset_buffer, *(this->dpu_pairs[rank_id].first));
        }

        for (int rank_id : rank_ids)
        {
            thrs[rank_id]->join();
            delete thrs[rank_id];
        }
        free((void *)memset_buffer);
    }

    StackNode *IDPHandler::GetStackNode(int rank_id, const char *stack_node_name)
    {
        return this->FindNode(rank_id, stack_node_name);
    }

    StackNode *IDPHandler::GetStackNode(int rank_id, uint32_t offset)
    {
        return this->FindNode(rank_id, offset);
    }

    void IDPHandler::RemoveStackNode(int rank_id, const char *stack_node_name)
    {
        this->dpu_pairs[rank_id].second->RemoveNode(stack_node_name);
    }

    void IDPHandler::GetStackNode(std::vector<StackNode *> &stack_nodes, std::vector<int> &rank_ids, const char *stack_node_name)
    {
        for (auto rank_id : rank_ids)
        {
            stack_nodes.push_back(this->FindNode(rank_id, stack_node_name));
        }
    }

    void IDPMemoryMgr::PrintCurrentMramStatus(void)
    {
        // #ifdef DEBUG_DPULOG
        int num_indexes = this->mram_free_space_indicates.size();

        for (int i = 0; i < num_indexes; i++)
        {
            if (this->mram_free_space_indicates[i].first == true)
            {
                printf("Free: %d ~ %d: Size: %lf(KB)\n",
                       this->mram_space_idxs[i], this->mram_space_idxs[i + 1], (this->mram_space_idxs[i + 1] - this->mram_space_idxs[i]) / 1024.0);
            }
            else
            {
                StackNode *node = this->mram_free_space_indicates[i].second;
                printf("Occupied: %d ~ %d: Size: %lf(KB) :%s\n",
                       this->mram_space_idxs[i], this->mram_space_idxs[i + 1], (this->mram_space_idxs[i + 1] - this->mram_space_idxs[i]) / 1024.0, node->name.c_str());

                if (node->start_byte != this->mram_space_idxs[i])
                {
                    printf("%sError! NodeName:%s node->start_byte:%d != this->mram_space_idxs[i]:%d\n",
                           KRED, node->name.c_str(), node->start_byte, this->mram_space_idxs[i]);
                    exit(-1);
                }

                if ((node->start_byte + node->block_byte) != this->mram_space_idxs[i + 1])
                {
                    printf("%sError! NodeName:%s node->start_byte + node->block_byte:%d != this->mram_space_idxs[i+1]:%d\n",
                           KRED, node->name.c_str(), node->start_byte + node->block_byte, this->mram_space_idxs[i + 1]);

                    exit(-1);
                }
            }
        }
        // #endif
    }

    // Maybe, update used memory status of each DPU MRAM - Comment
    void IDPMemoryMgr::UpdateStackNode(StackNode *node, int *data_bytes, int block_byte)
    {
        if (block_byte == 0)
            block_byte = 8;

        int start_byte = node->start_byte;
        int prev_block_byte = node->block_byte;

        int num_indexes = this->mram_space_idxs.size();

        int curr_idx = -1;

        for (int i = 0; i < (num_indexes - 1); i++)
        {
            int start_byte_ = this->mram_space_idxs[i];

            if (start_byte_ == start_byte)
            {
                curr_idx = i;
                break;
            }
        }

        if (curr_idx == -1)
        {
            printf("%sCannot Find Node for Update %d.\n", KRED, start_byte);
            exit(-1);
        }
        else
        {
            if (block_byte < prev_block_byte)
            {
                // 다음 space가 free한 공간일때,
                if (this->mram_free_space_indicates[curr_idx + 1].first == true)
                {
                    this->mram_space_idxs[curr_idx + 1] = start_byte + block_byte;
                }
                // 다음 space가 free한 공간이 아닐때,
                else
                {
                    this->mram_space_idxs[curr_idx + 1] = start_byte + block_byte;
                    this->mram_space_idxs.insert(mram_space_idxs.begin() + curr_idx + 2, start_byte + prev_block_byte);
                    this->mram_free_space_indicates.insert(mram_free_space_indicates.begin() + curr_idx + 1, std::make_pair(true, nullptr));
                }
            }
            else if (block_byte == prev_block_byte)
            {
                // do nothing
            }
            else
            {
                // 다음 space가 free한 공간일때,
                if (this->mram_free_space_indicates[curr_idx + 1].first == true)
                {
                    // 다음 space가 충분한지 확인한다.
                    if ((this->mram_space_idxs[curr_idx + 2] - this->mram_space_idxs[curr_idx]) > block_byte)
                    {
                        this->mram_space_idxs[curr_idx + 1] = start_byte + block_byte;
                    }
                    else
                    {
                        // 업데이트 불가
                        printf("%s1Cannot increase block_byte. %s :%d into %d\n", KRED, node->name.c_str(), prev_block_byte, block_byte);
                        PrintCurrentMramStatus();
                        exit(-1);
                    }
                }
                // 다음 space가 free한 공간이 아닐때,
                else
                {
                    // 업데이트 불가
                    printf("%s2Cannot increase block_byte of %s :%d into %d\n", KRED, node->name.c_str(), prev_block_byte, block_byte);
                    PrintCurrentMramStatus();
                    exit(-1);
                }
            }
        }

        if (data_bytes != NULL)
        {
            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                node->data_bytes[i] = data_bytes[i];
            }
        }
        else
        {
            for (int i = 0; i < NUM_DPU_RANK; i++)
            {
                node->data_bytes[i] = block_byte;
            }
        }
        node->block_byte = block_byte;
    }

    void IDPHandler::UpdateStackNode(int rank_id, StackNode *node, int *data_bytes, int block_byte)
    {
        auto &pair = this->dpu_pairs.at(rank_id);
        pair.second->UpdateStackNode(node, data_bytes, block_byte);
    }

    void IDPHandler::UpdateStackNode(int rank_id, StackNode *node, int block_byte)
    {
        auto &pair = this->dpu_pairs.at(rank_id);
        pair.second->UpdateStackNode(node, NULL, block_byte);
    }

    struct StackNode *IDPHandler::FindNode(int rank_id, const char *name)
    {
        return dpu_pairs[rank_id].second->FindNode(name);
    }

    struct StackNode *IDPHandler::FindNode(int rank_id, uint32_t offset)
    {
        return dpu_pairs[rank_id].second->FindNode(offset);
    }

    bool IDPHandler::CheckStackNameExists(int rank_id, const char *name)
    {
        for (StackNode *node : this->dpu_pairs[rank_id].second->stack_info.nodes)
        {
            if (node->name == std::string(name))
            {
                printf("%sError: Node Name: %s already exists.\n", KRED, name);
                return false;
            }
        }
        return true;
    }

    struct StackNode *IDPHandler::PushStackNodeAligned(int rank_id, const char *name, int mram_align_size)
    {
        if (!CheckStackNameExists(rank_id, name))
        {
            exit(-1);
        }

        printf("%sDeprecated!\n", KRED);
        exit(-1);

        return dpu_pairs[rank_id].second->PushStackNodeAligned(name, NULL, -1, 8);
    }

    struct StackNode *IDPHandler::PushStackNode(int rank_id, const char *name, int *data_bytes, int block_byte)
    {
        if (!CheckStackNameExists(rank_id, name))
        {
            exit(-1);
        }
        return dpu_pairs[rank_id].second->PushStackNodeAligned(name, data_bytes, block_byte, 8);
    }

    struct StackNode *IDPHandler::PushStackNodeAligned(int rank_id, const char *name, int *data_bytes, int block_byte, int mram_align_size)
    {
        if (!CheckStackNameExists(rank_id, name))
        {
            exit(-1);
        }
        return dpu_pairs[rank_id].second->PushStackNodeAligned(name, data_bytes, block_byte, mram_align_size);
    }

    struct StackNode *IDPHandler::PushStackNode(int rank_id, const char *name, int *data_bytes, int block_byte, DataType data_type, int data_type_size)
    {
        if (!CheckStackNameExists(rank_id, name))
        {
            exit(-1);
        }
        return dpu_pairs[rank_id].second->PushStackNodeAligned(name, data_bytes, block_byte, 8, data_type, data_type_size);
    }

    struct StackNode *IDPHandler::PushStackNodeAligned(int rank_id, const char *name, int *data_bytes, int block_byte, int mram_align_size, DataType data_type, int data_type_size)
    {
        if (!CheckStackNameExists(rank_id, name))
        {
            exit(-1);
        }
        return dpu_pairs[rank_id].second->PushStackNodeAligned(name, data_bytes, block_byte, mram_align_size, data_type, data_type_size);
    }

    uint8_t IDPHandler::GetChannelID(int rank_id)
    {
        // assert(this>dpu_pairs[rank_id].first->list.nr_ranks == 1);
        // struct dpu_rank_t *rank = this->dpu_pairs[rank_id].first->list.ranks[0];
        // rank
        return rnc_job_q->channel_ids[rank_id];
    }

    int IDPHandler::GetNumaNodeID(int rank_id)
    {
        return rnc_job_q->numa_node_ids[rank_id];
    }

    std::vector<int> IDPHandler::GetRankIDsPerChannel(uint8_t channel_id)
    {
        std::vector<int> rank_ids;
        for (int i = 0; i < NUM_MAX_RANKS; i++)
        {
            if (GetChannelID(i) == channel_id)
                rank_ids.push_back(i);
        }
        return rank_ids;
    }

    std::vector<int> IDPHandler::GetRankIDsPerNode(int node_id)
    {
        std::vector<int> rank_ids;
        for (int i = 0; i < NUM_MAX_RANKS; i++)
        {
            if (GetNumaNodeID(i) == node_id)
                rank_ids.push_back(i);
        }
        return rank_ids;
    }

    IDPHandler::IDPHandler()
    {
        pthread_mutex_init(&(this->print_log_lock), NULL);
        pthread_mutex_init(&(this->master_thr_mutex), NULL);
        pthread_mutex_init(&(this->encoding_table_mutex), NULL);
        pthread_cond_init(&(this->master_thr_condition), NULL);
        rnc_job_q = (RNS_Job_Queue_t *)malloc(sizeof(RNS_Job_Queue_t));
        rnc_job_q->worker_num = NUM_MAX_RANKS;
    }

    IDPHandler::IDPHandler(int rnc_worker_thread_num)
    {
        pthread_mutex_init(&(this->print_log_lock), NULL);
        pthread_mutex_init(&(this->master_thr_mutex), NULL);
        pthread_mutex_init(&(this->encoding_table_mutex), NULL);
        pthread_cond_init(&(this->master_thr_condition), NULL);
        rnc_job_q = (RNS_Job_Queue_t *)malloc(sizeof(RNS_Job_Queue_t));
        rnc_job_q->worker_num = rnc_worker_thread_num;

        std::string str;
        str = base_json;

        Json::Reader reader;
        Json::Value *root = new Json::Value();
        if (reader.parse(str, *root) == false)
        {
            std::cerr << "Failed to parse Json : " << reader.getFormattedErrorMessages() << std::endl;
            return;
        }

        this->curr_query_tree = &((*root)["query_tree"]);
    }

    IDPHandler::~IDPHandler()
    {
        for (auto &elem : encoding_table_map)
        {
            delete elem.second;
        }
        if (pthread_mutex_lock(&(rnc_job_q->mutex)) != 0)
        {
            printf("%sError pthread Lock Error. %s:%d\n", KRED, __FILE__, __LINE__);
            exit(-1);
        }

        rnc_job_q->job_done = 1;
        pthread_cond_broadcast(&(rnc_job_q->cond));
        if (pthread_mutex_unlock(&(rnc_job_q->mutex)) != 0)
        {
            printf("%sError pthread Lock Error. %s:%d\n", KRED, __FILE__, __LINE__);
            exit(-1);
        }

        for (int t = 0; t < rnc_job_q->worker_num; t++)
        {
            pthread_join(rnc_job_q->worker_threads_numa[t], NULL);
        }
        pthread_mutex_destroy(&(this->rnc_job_q->mutex));
        pthread_cond_destroy(&(this->rnc_job_q->cond));

        pthread_mutex_destroy(&this->master_thr_mutex);
        pthread_cond_destroy(&this->master_thr_condition);

        pthread_mutex_destroy(&this->encoding_table_mutex);

        pthread_mutex_destroy(&(this->print_log_lock));

        free(rnc_job_q);

        for (auto &dpu_pair : dpu_pairs)
        {
            dpu_free(*dpu_pair.first);
            delete dpu_pair.second;
        }
    }
}
