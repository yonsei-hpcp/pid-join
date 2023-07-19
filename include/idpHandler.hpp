#pragma once 


#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <list>
#include "argument.h"
#include "type_defs.hpp"
#include "json/json.h"
#include <queue>

#ifdef __cplusplus
extern "C"
{
#endif

#include <dpu.h>

#ifdef __cplusplus
}
#endif


namespace pidjoin
{
    class JoinInstance;
    class JoinOperator;

    extern std::unordered_map<int, std::string> param_name_map;
    extern std::unordered_map<int, int> param_size_map;
    extern std::unordered_map<std::string, int> operator_name_map;
    extern std::unordered_map<int, std::string> param_binary_name_map;
    extern std::unordered_map<int, std::string> param_return_var_map;
    extern std::unordered_map<int, int> param_return_size_map;

    class IDPArgumentHandler;
    class IDPHandler;

    /** 
     * Meta data for data stored in a rank - Comment
     * /////////////////// -> start_byte
     * // Data                          ^   ^
     * // Data                          |   |
     * // Data                          |   |
     * // Data -------------> data_bytes -  |
     * // 0000                              |
     * // 0000                              |
     * // 0000                              |
     * // 0000 ----------------> block_byte -
     */
    struct StackNode
    {
        int start_byte; // offset in physical address of MRAMs. All DPUs in a rank can be only accessed with the same offset - Comment
        int block_byte; // size of data block per DPU(MRAM) - Comment
        int data_bytes[64]; // size of useful data in a data block for each DPU - Comment
        std::string name;
        DataType data_type;
        int data_type_size;
    };

    class StackInfo
    {
    private:
    public:
        std::vector<StackNode *> nodes;
    };

    class IDPMemoryMgr
    {
        friend class IDPHandler;
        // friend class IDPArgumentHandler;

    private:
        std::vector<int> mram_space_idxs;
        std::vector<std::pair<bool, StackNode *>> mram_free_space_indicates;
        int dpu_id;
        StackInfo stack_info;
        pthread_mutex_t mem_mgr_lock;
        void RemoveNode(const char *name);
        struct StackNode *FindNode(const char *name);
        struct StackNode *FindNode(uint32_t offset);
        struct StackNode *PushStackNodeAligned(const char *name, int *data_bytes, int block_byte, int mram_align_size);
        struct StackNode *PushStackNodeAligned(const char *name, int *data_bytes, int block_byte, int mram_align_size, DataType dt, int dt_size);
        // Deprecated
        struct StackNode *PushStackNode(const char *name, int *data_bytes, int block_byte);
        int FindEmptySpace(StackNode *new_node, int mram_align_size, int block_byte);
        void UpdateStackNode(StackNode *node, int *data_bytes, int block_byte);

    public:
        void PrintCurrentMramStatus();
        IDPMemoryMgr()
        {
            pthread_mutex_init(&mem_mgr_lock, NULL);
            mram_space_idxs.push_back(0);
            mram_space_idxs.push_back(63 * 1024 * 1024);
            mram_free_space_indicates.push_back(std::make_pair(true, nullptr));
        }
        ~IDPMemoryMgr()
        {
            pthread_mutex_destroy(&mem_mgr_lock);
        }
    };

#define RANK_THREAD_SIGNAL_END 0
#define RANK_THREAD_SIGNAL_NEXT_OP 1

#define RANK_THREAD_STATUS_TERMINATED 0
#define RANK_THREAD_STATUS_JOB_DONE 1
#define RANK_THREAD_STATUS_RUNNING 2
#define RANK_THREAD_STATUS_PENDING 3

#define NUM_MAX_RANKS 40

    typedef struct
    {
        pthread_mutex_t wait_mutex;
        pthread_cond_t wait_condition;
    } rank_thread_control_t;

    typedef struct
    {
        rank_thread_control_t thr_control;
        IDPHandler* idp_handler;
        int run_signal;
        int curr_status;
        int rank_id;
        void *join_operator;
    } rank_thread_handle_t;

    #ifndef RNS_NEW_QUE
    struct rotate_n_stream_job_t_cmp
    {
        bool operator()(rotate_n_stream_job_t *a, rotate_n_stream_job_t *b)
        {
            return (a->job_priority >= b->job_priority); // rnc job prioroty queue outputs job with low priority first - Comment
        }
    };
    typedef std::priority_queue<rotate_n_stream_job_t *, std::vector<rotate_n_stream_job_t *>, rotate_n_stream_job_t_cmp> RNS_job_priority_queue_t;
    #else
    class RNS_job_priority_queue_t
    {
    private:
        std::list<rotate_n_stream_job_t *> _list;
        std::list<rotate_n_stream_job_t *>::iterator _top_elem;
        bool compare(rotate_n_stream_job_t *job_a, rotate_n_stream_job_t *job_b);
        std::pair<int, int> _prev_pop;
    public:
        RNS_job_priority_queue_t() {};
        ~RNS_job_priority_queue_t() {};
        void push(rotate_n_stream_job_t *job);
        rotate_n_stream_job_t *top();
        void pop();
        bool empty();

        std::string _type = "chBal";
    };
    #endif

    class IDPHandler
    {
        friend class IDPArgumentHandler;
        friend class TableSchemaManager;
        friend class JoinOperator;

    private:
        pthread_mutex_t print_log_lock;
        Json::Value *curr_query_tree = NULL;
        // Pipelined
        pthread_t rank_threads[NUM_MAX_RANKS];
        rank_thread_handle_t rank_thread_handle[NUM_MAX_RANKS];
        // Shared Objs
        pthread_mutex_t master_thr_mutex;
        pthread_cond_t master_thr_condition;
        // Basics
        pthread_mutex_t encoding_table_mutex;
        std::unordered_map<std::string, ENCODING_TABLE_t *> encoding_table_map;
        std::unordered_map<std::string, RankwiseMemoryBankBuffers_t *> immediate_buffer_map;
        struct StackNode *FindNode(int rank_id, uint32_t offset);

    public:
        std::vector<std::pair<struct dpu_set_t *, IDPMemoryMgr *>> dpu_pairs;
        RNS_Job_Queue_t *rnc_job_q;
        struct StackNode *FindNode(int rank_id, const char *name);
        IDPHandler();
        IDPHandler(int rnc_worker_thread_num);
        ~IDPHandler();

        struct StackNode *PushStackNodeAligned(int rank_id, const char *name, int mram_align_size);
        struct StackNode *PushStackNode(int rank_id, const char *name, int *data_bytes, int block_byte);
        struct StackNode *PushStackNodeAligned(int rank_id, const char *name, int *data_bytes, int block_byte, int mram_align_size);
        struct StackNode *PushStackNode(int rank_id, const char *name, int *data_bytes, int block_byte, DataType data_type, int data_type_size);
        struct StackNode *PushStackNodeAligned(int rank_id, const char *name, int *data_bytes, int block_byte, int mram_align_size, DataType data_type, int data_type_size);

        bool CheckStackNameExists(int rank_id, const char *name);
        void SendPipelineDoneSignal();
        void AllocateRank(std::vector<int> &rank_ids, int num);
        void AllocateRankPipeline(std::vector<int> &rank_ids, int num);

        int GetEncodedData(std::string &attr_name, std::string &key_name);

        void LoadProgram(int rank_id, int param);
        void LoadProgram(std::vector<int> &rank_id, int param);

        void PrintStackInfo(int rank_id);
        void PrintStackInfo(std::vector<int> &rank_ids);
        void PrintStackInfoBrief(int rank_id);

        void LoadParameter(int rank_id, DPUKernelParams_t &params_rank, int DPU_FUNC_TYPE);

        void RunKernel(int rank_id);
        void RunKernel(std::vector<int> &rank_ids);
        void RunKernelAsync(int rank_id);
        void RunKernelAsync(std::vector<int> &rank_ids);

        void WaitKernel(int rank_id);
        void WaitKernel(std::vector<int> &rank_ids);

        void ReadLog(int rank_id);
        void ReadLog(int rank_id, int dpu_id);

        int SetRNSRank(int rank_id);

        int RetrieveData(int rank_id, std::vector<char *> &retreive_buffers, std::vector<int> &data_bytes, const char *buffer_name);
        int RetrieveDataPreAlloced(int rank_id, std::vector<char *> &retreive_buffers, std::vector<int> &data_bytes, const char *buffer_name);
        int RetrieveDataManual(int rank_id, std::vector<char *> &retreive_buffers, const char *buffer_name, int byte_offset, int byte_size);
        int RetrieveData(std::vector<int> &rank_id, RankwiseMemoryBankBuffers_t& retreive_buffers, std::vector<std::vector<int>> &data_bytes, const char *buffer_name);

        int InitRNS();
        void SetJoinInstanceOnRNSJobQueue(JoinInstance *instance);
        int AddRNSJob(const rotate_n_stream_job_t *new_job, bool lock);
        int RotNCopy(
            rotate_n_stream_job_t *job);

        int StoreData(
            int rank_id, std::vector<char *> &buffers,
            std::vector<int> &data_bytes, int block_size, const char *buffer_name);
        int StoreData(
            std::vector<int> &rank_ids, RankwiseMemoryBankBuffers_t &buffers,
            std::vector<std::vector<int>> &data_bytes, std::vector<int> &block_size, const char *buffer_name);

        int StoreDataAligned(
            int rank_id, std::vector<char *> &buffers,
            std::vector<int> &data_bytes, int block_size, const char *buffer_name, int align_byte_size);
        int StoreDataAligned(
            std::vector<int> &rank_ids, RankwiseMemoryBankBuffers_t &buffers,
            std::vector<std::vector<int>> &data_bytes, std::vector<int> &block_size, const char *buffer_name, int align_byte_size);

        void HandleKernelResults(JoinInstance *join_instance, int rank_id, int DPU_FUNC_TYPE, DPUKernelParams_t &params);
        void HandleKernelResults(JoinInstance *join_instance, int rank_id, int DPU_FUNC_TYPE, DPUKernelParams_t &params, char *ptr);

        void ClearMRAM(int rank);
        void ClearMRAM(std::vector<int> &rank_ids);
        void PrintDPUInfo();

        void RemoveStackNode(int rank_id, const char *stack_node_name);
        StackNode *GetStackNode(int rank_id, const char *stack_node_name);
        StackNode *GetStackNode(int rank_id, uint32_t offset);
        void UpdateStackNode(int rank_id, StackNode *node, int *data_bytes, int block_byte);
        void UpdateStackNode(int rank_id, StackNode *node, int block_byte);
        void GetStackNode(std::vector<StackNode *> &stack_nodes, std::vector<int> &rank_ids, const char *stack_node_name);
        
        // Pipelining
        int ProceedQueryPlan(JoinOperator *join_operator, int rank_id, DPUKernelParams_t *param);

        uint8_t GetChannelID(int rank_id);
        int GetNumaNodeID(int rank_id);
        std::vector<int> GetRankIDsPerChannel(uint8_t channel_id);
        std::vector<int> GetRankIDsPerNode(int node_id);
        std::map<int, int> m_jobs_per_ch; // shared variable to control conccurent jobs per UPMEM channel doing RNS - Comment
    };

    class IDPArgumentHandler
    {
    private:
        // void ValidateArg
    public:
        ///////////////////////////////////////////////////////////////////////////////
        // Join
        ///////////////////////////////////////////////////////////////////////////////

        static void *ConfigureSortMergeJoinPartitioningArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *rnc_packets_joinkey_name,
            const char *sorted_joinkey_name,
            const char *histogram_addr_name);

        ////////////

        static void *ConfigureSortMergeJoinProbeArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *rnc_packets_joinkey_name,
            const char *partitioned_joinkey_name,
            const char *histogram_addr_name,
            const char *sorted_joinkey_name,
            const char *result_tids_name);

        ///////////////////////////////////////////////////////////////////////////////

        static void *ConfigureSortMergeJoinSortArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int packet_size,
            const char *rnc_packets_joinkey_name,
            const char *sorted_joinkey_name,
            bool data_skewness);

        ////////////

        static void *ConfigureSortMergeJoinProbeAllArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *r_sorted_joinkey_name,
            const char *s_sorted_joinkey_name,
            const char *result_tids_name);
        ///////////////////////////////////////////////////////////////////////////////

        static void *ConfigureGlobalPartitioningArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int partition_num, 
            const char *input_arr_name,
            const char *partitioned_input_arr_name,
            const char *partition_info_name, 
            const char *histogram_name);
        //////////////////////////////////////////////////////

        static void *ConfigureGlobalPartitioningPrevArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int partition_num,
            const char *tid_buffer_name,
            const char *payload_buffer_name,
            const char *partitioned_tid_name,
            const char *partitioned_payload_name,
            const char *histogram_name);

        static void *ConfigurePacketwiseGlobalPartitioningArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int partition_num,
            int packet_size,
            RankwiseMemoryBankBuffers_t *imm_hist_buff,
            const char *packet_histogram_name,
            const char *partition_info_name,
            const char *tid_buffer_name,
            const char *payload_buffer_name,
            const char *aligned_tid_name,
            const char *aligned_payload_name);

        static void *ConfigurePacketwiseLocalPartitioningArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int packet_size,
            int *partition_nums,
            int shift_val,
            int tuple_size,
            const char *rnc_packets_joinkey_name,
            const char *generated_tid_name,
            const char *partitioned_key_name,
            const char *partition_info_name);

        //////////////////////////////////////////////////////

        static void *ConfigureLocalPartitioningArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int *partition_nums,
            int shift_val,
            int tuple_size,
            const char *input_arr_name,
            const char *partitioned_arr_name,
            const char *partition_info_name);

        /////////////////////////////////////////////////////////////////////

        static void *ConfigurePHJHashTableBuildHorizontalArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *r_partitioned_payload_name,
            const char *r_partitioned_tid_name,
            const char *r_partitioned_info_name,
            const char *hash_table_name);

        /////////////////////////////////////////////////////////////////////

        static void *ConfigurePHJHashTableProbeHorizontalArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *hash_table_name,
            const char *partitioned_s_payload_name,
            const char *partitioned_s_tid_name,
            const char *s_partition_info_name,
            const char *result_tids_name);

        static void *ConfigurePHJHashTableProbeInnerHorizontalArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int join_type,
            const char *hash_table_name,
            const char *partitioned_s_arr_name,
            const char *s_partition_info_name,
            const char *result_tids_name);

        /////////////////////////////////////////////////////////////////////

        static void *ConfigureNPHJBuildHorizontalArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *hash_table_name,
            const char *R_name);

        /////////////////////////////////////////////////////////////////////

        static void *ConfigureNPHJProbeHorizontalArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char *hash_table_name,
            const char *S_name,
            const char *Result_name);

        //////////////////////////////////////////////////////////////////////////////
        // Global Partitioning for Join v2
        //////////////////////////////////////////////////////////////////////////////

        static void *ConfigureGlobalPartitioningPacketArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int packet_size,
            int partition_num,
            int partition_type,
            RankwiseMemoryBankBuffers_t *imm_hist_buffers,
            const char *input_buffer_name1, // tid
            const char *local_histogram_name,
            const char *packet_histogram_name,
            const char *input_buffer_name2, // Payload
            const char *result_buffer_name);

        static void *ConfigureGlobalPartitioningCountArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int packet_size,
            int partition_num,
            int partition_type,
            const char *input_buffer_name,
            const char *partition_info_name,
            const char *histogram_name);
        
        static void *ConfigureDebuggerArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int debugger_op_type,
            const char *target_addr1,
            const char *target_addr2);

        static void *ConfigureMicrobenchmarkArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int benchamark_type, int tasklet_num);
    
        // nested loop join
        static void *ConfigureNestedLoopJoinArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int packet_size,
            const char *R_packets_name,
            const char *S_packets_name,
            const char *result_node_name);

        static void *ConfigureFinishJoinArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            const char * join_result_name);

        
    };

}

