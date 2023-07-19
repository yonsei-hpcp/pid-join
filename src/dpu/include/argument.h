#ifndef _ARGUMENT_H_
#define _ARGUMENT_H_

#define NUM_MAX_RANKS 40

#define ERROR_TYPE_NULL 1
#define ERROR_TYPE_NOT_ACCEPTED 2

typedef int64_t Key64_t;
typedef uint64_t TupleID64_t;


#define JOIN_TYPE_EQUI 1

#define GLB_PART_ARR_L0 1

#define GLB_PART_TUPLE_ID_IDX0_2 2
#define GLB_PART_TUPLE_ID_IDX1_2 3

#define GLB_PART_TUPLE_ID_IDX0_4 4
#define GLB_PART_TUPLE_ID_IDX1_4 5
#define GLB_PART_TUPLE_ID_IDX2_4 6
#define GLB_PART_TUPLE_ID_IDX3_4 7

#define GLB_PART_JK_IDX0_2 12
#define GLB_PART_JK_IDX1_2 13

#define GLB_PART_JK_IDX0_4 14
#define GLB_PART_JK_IDX1_4 15
#define GLB_PART_JK_IDX2_4 16
#define GLB_PART_JK_IDX3_4 17

#define GLB_PART_JK_IDX0_8 18



#define PACKET_PROJ_TYPE_DOUBLE 1
#define PACKET_PROJ_TYPE_INT64 2
#define PACKET_PROJ_TYPE_INT8 3


typedef struct
{
    uint32_t lvalue;
    uint32_t rvalue;
} tuplePair_t;

typedef struct
{
    uint32_t bucket[2];
} tuplePairBucket_t;


#define DIVIDE_TID_KEY_FROM_TK_PAIR64(dst_tid, dst_key, tkpair64)		\
do{		\
	dst_tid = (tkpair64 >> 32);		\
	dst_key = (tkpair64 & 0xFFFFFFFF);		\
}while(0)

#define COMBINE_TID_KEY(tuple_id, join_key, result)		\
do{		\
	result = (((int64_t)tuple_id << 32) | (int64_t)join_key);		\
}while(0)


typedef struct
{
    int32_t ERROR_TYPE_0;
    int32_t ERROR_TYPE_1;
    int32_t ERROR_TYPE_2;
    int32_t ERROR_TYPE_3;
    uint64_t cycle_count;
    // hash join
    uint64_t miss_count;
    uint64_t total_count;
} dpu_results_t;


#define GET_RANK_DPU_FROM_TUPLE_ID(dpu_id, tuple_id)                        \
do {                                                                        \
    int32_t rank_id = ((uint32_t)tuple_id & (0xF8000000)) >> 27;            \
    int32_t local_dpu_id = (((uint32_t)tuple_id & (0x7e00000))) >> 21;      \
    dpu_id = ((rank_id << 6) + local_dpu_id);                               \
} while (0)

#define GET_LOCAL_ID_FROM_TUPLE_ID(local_id, tuple_id)     \
do{                                                        \
    local_id = (uint32_t)(tuple_id & (0x1fffff));          \
}while(0)

#define CREATE_TUPLE_ID(rank_id, dpu_id, tuple_idx, ret)      \
do {                                                          \
    ret = ((tuple_idx | (rank_id << 27)) | (dpu_id << 21));   \
} while(0)                                                            

typedef struct
{
    int32_t debugger_op;
    int32_t check_target_addr1;
    int32_t check_target_elem_num1;
    int32_t check_target_addr2;
    int32_t check_target_elem_num2;
    int32_t num_max_tuples;
    int32_t rank_id;
    int32_t dpu_id;
    int32_t total_rank;
} debugger_arg;

typedef struct
{
    int32_t benchmark_type; 
    int32_t stride_byte;
    int32_t tasklet_num;
} microbenchmark_arg;

typedef struct
{
    // common
    int64_t cycle_count;
    // mram bandwidth
    int64_t mram_read;
    int64_t mram_write;
    int32_t mram_granularity; 
    int32_t xfer_byte_size;
    // ops
    int64_t ops_count;
    // hash
    int64_t miss_count;
    int64_t hash_count;
    int64_t not_matched;
} microbenchmark_return_arg;



typedef struct
{
} debugger_return_arg;

typedef struct
{
    /* data */
    int32_t num_packets;
    int32_t packet_size;
    int32_t r_packet_start_byte;
    int32_t r_sorted_start_byte;
    int32_t histogram_addr_start_byte;
    int32_t data_skewness;
} sort_merge_partitioning_arg;

typedef struct
{
    /* data */
    int32_t num_packets;
    int32_t s_packet_start_byte;
    int32_t s_sorted_start_byte;
    int32_t r_partitioned_start_byte;
    int32_t r_total_bytes;
    int32_t histogram_addrs;
    int32_t result_probe_start_byte;
} sort_merge_probe_arg;

typedef struct
{
    /* data */
    int32_t r_sorted_start_byte;
    int32_t s_sorted_start_byte;
    int32_t r_total_bytes;
    int32_t s_total_bytes;
    int32_t result_probe_start_byte;
} sort_merge_probe_all_arg;

typedef struct
{
    /* data */
    int32_t r_total_elem;
} sort_merge_partitioning_return_arg;

typedef struct
{
    /* data */
} sort_merge_probe_return_arg;



typedef struct
{
    TupleID64_t tid_dpu_internal_offset;
    int32_t     rank_id;
    int32_t     dpu_id;
    int32_t     tids_offset; // or packet
    int32_t     tids_size;
    int32_t     payload_offset;
    int32_t     payload_num;
    int32_t     payload_size;
    int32_t     result_offset;
    // additional vars
    int32_t     packet_project_type; // 1B 2B 4B 8B
    int32_t     data_type; 
    int32_t     tuple_size; 
} projection_arg;

typedef struct
{
    int32_t tid1_start_byte;
    int32_t tid2_start_byte;
    int32_t tid1_elem_num;
    int32_t tid2_elem_num;
    int32_t tid_output_start_byte;
} OR_predicate_arg;

typedef struct
{
    int32_t table_r_start_byte;
    int32_t partitioned_table_r_start_byte;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
    int32_t partition_num;
    int32_t table_r_num_elem;
} hash_global_partitioning_arg;


typedef struct
{
    int32_t input_type; // 0: Compressed Array, 1: 2 Arrays. 
    int32_t partition_type; // 0: tid partitioning , 1: hash partitioning, 
    int32_t input_offset;
    int32_t elem_num;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
    int32_t partition_num;
} glb_partition_count_arg;

typedef struct
{
    int32_t join_result_start_byte;
    int32_t max_bytes;
    int32_t effective_bytes;
    int32_t dummy;
} finish_join_arg;

typedef struct
{
    int32_t dpu_id;
    int32_t rank_id;
    int32_t input_type; // 0: Compressed Array, 1: 2 Arrays. 
    int32_t partition_type; // 0: tid partitioning , 1: hash partitioning, 
    int32_t input_offset1;
    int32_t input_offset2;
    int32_t elem_num;
    int32_t histogram_start_byte;
    int32_t packet_histogram_start_byte;
    int32_t partition_num;
    int32_t result_offset;
    int32_t packet_size;
    // opt
    int32_t partition_start;
} glb_partition_packet_arg;

typedef struct
{
	int32_t tid_start_byte;
	int32_t table_r_start_byte;
	int32_t histogram_start_byte;
	int32_t partitioned_tid_start_byte;
	int32_t partitioned_r_start_byte;
	int32_t partition_num;
	int32_t table_r_num_elem;
} hash_global_partitioning_prev_arg;

typedef struct
{
    int32_t input_arr_start_byte;
    int32_t input_data_bytes;
    int32_t partitioned_input_arr_start_byte;
    int32_t result_partition_info_start_byte;
    // if 0, do calculate partitioning
    int32_t do_calculate_partition_num; 
    int32_t shift_val;
    int32_t total_rank;
    int32_t tuple_size;
} hash_local_partitioning_arg;

typedef struct
{
    int32_t input_arr_start_byte;
    int32_t input_data_bytes;
    int32_t partitioned_input_arr_start_byte;
    int32_t result_partition_info_start_byte;
    // if 0, do calculate partitioning
    int32_t do_calculate_partition_num; 
    int32_t shift_val;
    int32_t total_rank;
    int32_t tuple_size;
    int32_t level; 
} hash_local_partitioning_multilevel_arg;

typedef struct
{
    int32_t packet_start_byte;
    int32_t generated_local_tid_start_byte;
    int32_t partitioned_result_start_byte;
    int32_t result_partition_info_start_byte;
    // if 0, do calculate partitioning
    int32_t do_calculate_partition_num; 
    int32_t shift_val;
    int32_t total_rank;
    int32_t num_packets;
    int32_t tuple_size;
    int32_t packet_size;
} packetwise_hash_local_partitioning_arg;

typedef struct
{
    int64_t elem_num;
    int64_t partition_num;
} packetwise_hash_local_partitioning_return_arg;

typedef struct
{
    int64_t elem_num;
    int64_t partition_num;
} hash_local_partitioning_return_arg;

typedef struct
{
    int64_t elem_num;
    int64_t partition_num;
} hash_local_partitioning_multilevel_return_arg;

typedef struct
{
	int32_t dpu_id;
	int32_t rank_id;
	int32_t histogram_start_byte;
	int32_t partition_info_start_byte;
	int32_t tid_start_byte; 
	int32_t payload_start_byte;
	int32_t result_tid_start_byte; // NOT USED
	int32_t result_payload_start_byte;
	int32_t table_num_elem;
	int32_t partition_num;
} packetwise_hash_global_partitioning_arg;

typedef struct
{
    int32_t packet_size;
    int32_t packet_type;
    int32_t dpu_total_packet_num;
} packetwise_hash_global_partitioning_return_arg;

typedef struct
{
    int32_t data_type_size;
    int32_t packet_num;
    int32_t tid_packet_start_byte;
    int32_t payload_array_start_byte;
    int32_t payload_packet_start_byte;
} packetwise_attr_partitioning_arg;

typedef struct
{
    int32_t tid_src_start_byte;
    int32_t tid_dest_start_byte;
    int32_t partition_num;
    int32_t tid_elem_num;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
} tid_partitioning_arg;

typedef struct
{
    int64_t tid_squeezed_total_elem_num;
} tid_partitioning_return_arg;


#define TID_MASK 0x7FF
#define TID_SHIFT 23

typedef struct
{
    int32_t R_packet_start_byte;
    int32_t S_packet_start_byte;
    int32_t R_num_packets;
    int32_t S_num_packets;
    int32_t packet_size;
    int32_t result_start_byte;
} nested_loop_join_arg;

typedef struct
{
    int32_t R_packet_start_byte;
    int32_t num_packets;
    int32_t packet_size;
    int32_t hash_table_start_byte;
    int32_t total_rank;
    int32_t tuple_size;
} hash_nphj_build_arg;

typedef struct
{
    int32_t S_packet_start_byte;
    int32_t num_packets;
    int32_t packet_size;
    int32_t hash_table_start_byte;
    int32_t result_start_byte;
    int32_t hash_table_bucket_num;
    int32_t total_rank;
    int32_t tuple_size;
} hash_nphj_probe_arg;

typedef struct
{
    uint32_t parted_R_offset;
    uint32_t parted_Tid_R_offset;
    uint32_t HT_offset;
    uint32_t parted_R_info_offset;
    uint32_t partition_num;
    uint32_t R_num;
    uint32_t compressed_key;
} hash_phj_build_arg;

typedef struct
{
    uint32_t HT_offset;
    uint32_t input_offset;
    uint32_t input_byte_size;
    uint32_t input_type_size;
    uint32_t histogram_offset;
    uint32_t partition_num;
    uint32_t target_index;
} hash_unique_and_partition_arg;

typedef struct
{
    int32_t parted_S_offset;
    int32_t parted_Tid_S_offset;
    int32_t HT_offset;
    int32_t parted_S_info_offset;
    int32_t Result_offset;
    int32_t partition_num;
    int32_t S_num;
} packetwise_hash_phj_probe_arg;

typedef struct
{
    uint32_t parted_R_offset;
    uint32_t parted_Tid_R_offset;
    uint32_t HT_offset;
    uint32_t parted_R_info_offset;
    uint32_t partition_num;
    uint32_t R_num;
} packetwise_hash_phj_build_arg;

typedef struct
{
    int32_t parted_S_offset;
    int32_t parted_Tid_S_offset;
    int32_t HT_offset;
    int32_t parted_S_info_offset;
    int32_t Result_offset;
    int32_t partition_num;
    int32_t S_num;
    int32_t rank_id;
    int32_t dpu_id;
    int32_t key_table_type;
    int32_t probe_type;
} hash_phj_probe_arg;

typedef struct
{
    int32_t hash_table_offset;
    int32_t R_offset;
    int32_t R_byte_size;
} no_partitioned_join_build_hash_table_arg;

typedef struct
{
    int32_t hash_table_offset;
    int32_t hash_table_byte_size;
    int32_t S_offset;
    int32_t S_byte_size;
    int32_t result_offset;
} no_partitioned_join_probe_hash_table_arg;

typedef struct
{
	int64_t result_size;	// Size in Byte
    int32_t miss_count;
} hash_phj_probe_return_arg;


#define ADD_NODE_MASK       0xF0000000
#define SUB_NODE_MASK       0xE0000000
#define MUL_NODE_MASK       0xD0000000
#define DIV_NODE_MASK       0xC0000000

#define MRAM_REF_NODE_MASK  0xA0000000
#define CONSTANT_NODE_MASK  0x90000000
#define WRAM_REF_NODE_MASK  DPUWISE_TUPLEID_OFFSET00

#define STACK_END           0xFFFFFFFF

#define ARITH_TYPE_MASK     0x08000000
#define ARITH_OP_MASK       0xF0000000
#define ARITH_VALUE_MASK    0x03FFFFFF

#define STACK_NODE_DEPTH    10

#define ARITH_TYPE_MM 0x0
#define ARITH_TYPE_MW 0x1
#define ARITH_TYPE_MC 0x2
#define ARITH_TYPE_CM 0x3
#define ARITH_TYPE_CW 0x4
#define ARITH_TYPE_CC 0x5
#define ARITH_TYPE_WM 0x6
#define ARITH_TYPE_WW 0x7
#define ARITH_TYPE_WC 0x8

union ArithNode
{
    int32_t Node_Type;
    int32_t value;
    int32_t operand_offset;
};

typedef struct
{
    // TID
    int32_t tid_offset;
    int tids_size;
    int32_t output_offset;
    union ArithNode arith_node_stack[STACK_NODE_DEPTH];
} arithmetic_arg;

#define OP_TYPE_INT32 0x0
#define OP_TYPE_INT64 0x1
#define OP_TYPE_CHAR 0x2
#define OP_TYPE_DATE 0x3


typedef struct
{
    int32_t key_col_double_buff_offset;
    int32_t tid_col_double_buff_offset;
    int32_t key_col_src_offset;
    int32_t tid_col_src_offset;
    int32_t key_col_num;
    int32_t key_col_dest_offset;
    int32_t tid_col_dest_offset;
} int64_col_sort_w_tid_arg;

typedef struct
{
    int32_t order; // ascending / descending
    int32_t tids_offset;
    int32_t tids_num;
    int32_t data_offset;
    int32_t data_num;
} integer_sort_arg;

typedef struct
{
    int32_t order; // ascending / descending
    int32_t tids_offset;
    int32_t tids_num;
    int32_t data_offset;
    int32_t data_num;
} string_sort_arg;

typedef struct
{
    int32_t order; // ascending / descending
    int32_t tids_offset;
    int32_t tids_num;
    int32_t date_offset;
    int32_t date_num;
} date_sort_arg;




//////////////////////////////////////////////////
// Selection Arg
//////////////////////////////////////////////////

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int32_t pred_op;
	int32_t cmp_date;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} date_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data1_size;
	int32_t data1_offset;
	int32_t data2_size;
	int32_t data2_offset;
	int32_t result_offset;
} date_date_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	double cmp_double;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} double_selection_arg;


typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int64_t cmp_int64;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} integer64_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	uint8_t cmp_encode;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
    int32_t dtype_size;
} encoded_string_selection_arg;


typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	//
    int predicate_num;
    int32_t predicates[8];
	//
    int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} in_predicate_selection_arg;

typedef struct
{
	int32_t is_double;
	double cmp_double1;
	double cmp_double2;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} _8b_btw_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int64_t cmp_integer_1;
	int64_t cmp_integer_2;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} int64_btw_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int32_t cmp_date1;
	int32_t cmp_date2;
	int32_t pred_op;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} date_btw_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	int32_t cmp_str[30];
	int32_t char_num[256];
	int32_t max_length;
	int32_t cmp_length;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} string_mid_selection_arg;

typedef struct
{
	TupleID64_t tid_dpu_internal_offset;
	char cmp_str[16];
	int32_t max_length;
	int32_t tids_size;
	int32_t tids_offset;
	int32_t data_size;
	int32_t data_offset;
	int32_t result_offset;
} string_seq_selection_arg;

typedef struct
{
	int64_t result_size;	// Size in Byte
} selection_return_arg;


// 8 bit

/***
 * 1024B Key based Pages
 * */

// 64 bit
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define RNS_PAGE_SIZE_128B 128
#define RNS_PAGE_SIZE_64B 64


// #define SET_LOCAL_PAGE_TUPLE_ID(num, src_rank, dst_rank, packet_offset, elem_num)
// {
//     (src_rank << )
// }

// #define GET_LOCAL_PAGE_TUPLE_ID(num, src_rank, dst_rank, packet_offset, elem_num)
// {
//     src_rank
// }

#define PAGE_ELEM_NUM_1024B_8B 127

#define CALCULATE_LOCAL_PAGE_TUPLE_ID(my_packet_id, packet_idx, tuple_id) \
    do\
    {\
        tuple_id = (my_packet_id) * sizeof(data_packet_u64_128_t) + 8 + (packet_idx << 3);\
    } \
    while (0)


typedef struct
{
    int8_t num;
    int8_t src_rank;
    int8_t dst_rank;
    uint8_t packet_offset;
} packet_id_t;


/***
 * 128B Key based Pages
 * */
#define PAGE_ELEM_NUM_128B_8B 16

typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_i64_128_t;


typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_u64_128_t;

typedef struct
{
    tuplePair_t tup_elem[PAGE_ELEM_NUM_128B_8B];  
} data_packet_t64_128_t;


/***
 * 64B Key based Pages
 * */

#define PAGE_ELEM_NUM_64B_8B 8

typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_u64_64_t;

#endif