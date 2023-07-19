DPU_DIR := src
HOST_DIR := host
BUILDDIR ?= bin
NR_TASKLETS ?= 16
BL ?= 10
NR_DPUS ?= 1
ENERGY ?= 0

# Projection

TUPLE_ID_ORDER_PROJECTION := ${BUILDDIR}/tuple_id_order_projection
SRC_TUPLE_ID_ORDER_PROJECTION := ${DPU_DIR}/project/tuple_id_order_projection.c

PROJECTION_1B := ${BUILDDIR}/projection_1B
SRC_PROJECTION_1B := ${DPU_DIR}/project/projection_1B.c

PACKET_PROJECTION := ${BUILDDIR}/packet_projection
SRC_PACKET_PROJECTION := ${DPU_DIR}/project/packet_projection.c

PROJECTION_1B_UNSORTED := ${BUILDDIR}/projection_1B_unsorted
SRC_PROJECTION_1B_UNSORTED := ${DPU_DIR}/project/projection_1B_unsorted.c

PROJECTION_8B := ${BUILDDIR}/projection_8B
SRC_PROJECTION_8B := ${DPU_DIR}/project/projection_8B.c

GLB_PARTITION_COUNT := ${BUILDDIR}/glb_partition_count
SRC_GLB_PARTITION_COUNT := ${DPU_DIR}/join/glb_partition_count.c ${DPU_DIR}/hash.c

GLB_PARTITION_PACKET := ${BUILDDIR}/glb_partition_packet
SRC_GLB_PARTITION_PACKET := ${DPU_DIR}/join/glb_partition_packet.c ${DPU_DIR}/hash.c

#SRC_GLB_PARTITION_PACKET := ${DPU_DIR}/join/glb_partition_packet_opt.c ${DPU_DIR}/hash.c


DEBUGGER := ${BUILDDIR}/debugger
SRC_DEBUGGER := ${DPU_DIR}/debugger.c

# All2All KEY1 Rank Join

A2A_PACKETWISE_ATTR_PARTITION_PACKET_128B := ${BUILDDIR}/a2a_packetwise_attr_partition_packet_128B

SRC_A2A_PACKETWISE_ATTR_PARTITION_PACKET_128B := ${DPU_DIR}/project/a2a_packetwise_attr_partition_packet_128B.c

A2A_SRJ_GLOBAL_PARTITION := ${BUILDDIR}/a2a_srj_global_partition
A2A_SRJ_GLOBAL_PARTITION_PREV := ${BUILDDIR}/a2a_srj_global_partition_prev
A2A_SRJ_LOCAL_PARTITION := ${BUILDDIR}/a2a_srj_local_partition
A2A_PACKETWISE_SRJ_LOCAL_PARTITION := ${BUILDDIR}/a2a_packetwise_srj_local_partition
A2A_PACKETWISE_SRJ_GLOBAL_PARTITION := ${BUILDDIR}/a2a_packetwise_srj_global_partition

SRC_A2A_SRJ_GLOBAL_PARTITION = ${DPU_DIR}/join/a2a_srj_global_partition.c ${DPU_DIR}/hash.c
SRC_A2A_SRJ_GLOBAL_PARTITION_PREV := ${DPU_DIR}/join/a2a_srj_global_partition_prev.c
SRC_A2A_SRJ_LOCAL_PARTITION = ${DPU_DIR}/join/a2a_srj_local_partition.c ${DPU_DIR}/hash.c
SRC_A2A_PACKETWISE_SRJ_LOCAL_PARTITION = ${DPU_DIR}/join/a2a_packetwise_srj_local_partition.c ${DPU_DIR}/hash.c
SRC_A2A_PACKETWISE_SRJ_GLOBAL_PARTITION = ${DPU_DIR}/join/a2a_packetwise_srj_global_partition.c ${DPU_DIR}/hash.c

# Hash Table build Probe

A2A_SRJ_BUILD_LINEAR_PROBE := ${BUILDDIR}/a2a_srj_build_linear_probe
A2A_SRJ_PROBE_INNER_LINEAR_PROBE := ${BUILDDIR}/a2a_srj_probe_inner_mt_linear_probe
A2A_SRJ_PROBE := ${BUILDDIR}/a2a_srj_probe

SRC_A2A_SRJ_BUILD_LINEAR_PROBE = ${DPU_DIR}/join/a2a_srj_build.c ${DPU_DIR}/hash.c
SRC_A2A_SRJ_PROBE_INNER_LINEAR_PROBE = ${DPU_DIR}/join/a2a_srj_probe_inner_mt.c ${DPU_DIR}/hash.c
SRC_A2A_SRJ_PROBE = ${DPU_DIR}/join/a2a_srj_probe.c

# nested loop join
NESTED_LOOP_JOIN := ${BUILDDIR}/nested_loop_join
SRC_NESTED_LOOP_JOIN := ${DPU_DIR}/join/nested_loop_join.c



#NO_PARTITIONED_BUILD := ${BUILDDIR}/a2a_no_partitioned_build
#NO_PARTITIONED_PROBE_INNER := ${BUILDDIR}/a2a_no_partitioned_probe
SRC_NO_PARTITIONED_BUILD = ${DPU_DIR}/join/no_partitioned_hash_join_build.c ${DPU_DIR}/hash.c
SRC_NO_PARTITIONED_PROBE_INNER = ${DPU_DIR}/join/no_partitioned_hash_join_probe.c ${DPU_DIR}/hash.c

A2A_SRJ_NOTID_BUILD := ${BUILDDIR}/a2a_srj_notid_build
A2A_SRJ_NOTID_PROBE := ${BUILDDIR}/a2a_srj_notid_probe

SRC_A2A_SRJ_NOTID_BUILD = ${DPU_DIR}/join/a2a_srj_notid_build.c
SRC_A2A_SRJ_NOTID_PROBE = ${DPU_DIR}/join/a2a_srj_notid_probe.c

# Sort Merge Join
MPSM_JOIN_PARTITION := ${BUILDDIR}/mpsm_join_partition
MPSM_JOIN_PROBE := ${BUILDDIR}/mpsm_join_probe

SRC_MPSM_JOIN_PARTITION := ${DPU_DIR}/join/mpsm_join_partition.c
SRC_MPSM_JOIN_PROBE := ${DPU_DIR}/join/mpsm_join_probe.c


MPSM_JOIN_SORT := ${BUILDDIR}/mpsm_join_sort
MPSM_JOIN_RADIX_SORT := ${BUILDDIR}/mpsm_join_radix_sort
MPSM_JOIN_MERGE_SORT := ${BUILDDIR}/mpsm_join_merge_sort
MPSM_JOIN_QUICK_SORT := ${BUILDDIR}/mpsm_join_quick_sort
MPSM_JOIN_PROBE_ALL := ${BUILDDIR}/mpsm_join_probe_all

SRC_MPSM_JOIN_SORT := ${DPU_DIR}/join/mpsm_join_sort.c
SRC_MPSM_JOIN_RADIX_SORT := ${DPU_DIR}/join/mpsm_join_radix_sort.c
SRC_MPSM_JOIN_MERGE_SORT := ${DPU_DIR}/join/mpsm_join_merge_sort.c
SRC_MPSM_JOIN_QUICK_SORT := ${DPU_DIR}/join/mpsm_join_quick_sort.c
SRC_MPSM_JOIN_PROBE_ALL := ${DPU_DIR}/join/mpsm_join_probe_all.c

# GroupBy and Aggregations
MICROBENCHMARK_MRAM_BANDWIDTH := ${BUILDDIR}/microbenchmark_mram_bandwidth
MICROBENCHMARK_WRAM_BANDWIDTH := ${BUILDDIR}/microbenchmark_wram_bandwidth
MICROBENCHMARK_OPS := ${BUILDDIR}/microbenchmark_ops
MICROBENCHMARK_HASH := ${BUILDDIR}/microbenchmark_hash

SRC_MICROBENCHMARK_MRAM_BANDWIDTH = ${DPU_DIR}/test/mram_bandwidth.c
SRC_MICROBENCHMARK_WRAM_BANDWIDTH = ${DPU_DIR}/test/wram_bandwidth.c
SRC_MICROBENCHMARK_OPS := ${DPU_DIR}/test/ops.c # ${DPU_DIR}/hash_test.c
SRC_MICROBENCHMARK_HASH :=${DPU_DIR}/hash_test.c  #${DPU_DIR}/test/hash.c 

GROUPBY_1B := ${BUILDDIR}/groupby_uint8
GROUPBY_4B := ${BUILDDIR}/groupby_int32
GROUPBY_8B := ${BUILDDIR}/groupby_int64
AGGREGATION := ${BUILDDIR}/aggregation

GROUPBY_AGGREGATION := ${BUILDDIR}/groupby_aggregation

SRC_KEY1_COL_HASH_GROUPBY_AGGR = ${DPU_DIR}/groupby/key1_col_hash_groupby_aggr.c
SRC_KEY2_COL_HASH_GROUPBY_AGGR = ${DPU_DIR}/groupby/key2_col_hash_groupby_aggr.c
SRC_KEY3_COL_HASH_GROUPBY_AGGR = ${DPU_DIR}/groupby/key3_col_hash_groupby_aggr.c
SRC_NOKEY_GROUPBY_AGGR = ${DPU_DIR}/groupby/nokey_col_hash_groupby_aggr.c

SRC_GROUPBY = ${DPU_DIR}/groupby/groupby.c
SRC_AGGREGATION = ${DPU_DIR}/groupby/aggregation.c
#SRC_GROUPBY_AGGREGATION = ${DPU_DIR}/groupby/hash_groupby_aggr.c
SRC_GROUPBY_AGGREGATION = ${DPU_DIR}/groupby/hash_groupby_aggr_v2.c

# Sort
INT64_COL_SORT_W_TID := ${BUILDDIR}/int64_col_sort_w_tid
SRC_INT64_COL_SORT_W_TID = ${DPU_DIR}/sort/int64_col_sort_w_tid.c


# Selection
DATE_SELECTION := ${BUILDDIR}/date_selection
DATE_DATE_SELECTION := ${BUILDDIR}/date_date_selection
DOUBLE_SELECTION := ${BUILDDIR}/double_selection
INTEGER64_SELECTION := ${BUILDDIR}/integer64_selection
8B_BTW_SELECTION := ${BUILDDIR}/8b_btw_selection
DATE_BTW_SELECTION := ${BUILDDIR}/date_btw_selection
STRING_MID_SELECTION := ${BUILDDIR}/string_mid_selection
STRING_SEQ_SELECTION := ${BUILDDIR}/string_seq_selection
ENCODED_STRING_SELECTION := ${BUILDDIR}/encoded_string_selection
ENCODED_STRING_SELECTION_IN_PREDICATE := ${BUILDDIR}/encoded_string_selection_in_predicate
INTEGER64_SELECTION_IN_PREDICATE := ${BUILDDIR}/integer64_selection_in_predicate
FINISH_JOIN := ${BUILDDIR}/finish_join

SRC_DATE_SELECTION = ${DPU_DIR}/select/date_selection.c
SRC_DATE_DATE_SELECTION = ${DPU_DIR}/select/date_date_selection.c
SRC_DOUBLE_SELECTION = ${DPU_DIR}/select/double_selection.c
SRC_INTEGER64_SELECTION := ${DPU_DIR}/select/integer64_selection.c
SRC_8B_BTW_SELECTION = ${DPU_DIR}/select/8b_btw_selection.c
SRC_DATE_BTW_SELECTION = ${DPU_DIR}/select/date_btw_selection.c
SRC_STRING_MID_SELECTION = ${DPU_DIR}/select/string_mid_selection.c
SRC_STRING_SEQ_SELECTION = ${DPU_DIR}/select/string_seq_selection.c
SRC_ENCODED_STRING_SELECTION := ${DPU_DIR}/select/encoded_string_selection.c
SRC_ENCODED_STRING_SELECTION_IN_PREDICATE := ${DPU_DIR}/select/encoded_string_selection_in_predicate.c
SRC_INTEGER64_SELECTION_IN_PREDICATE := ${DPU_DIR}/select/integer64_selection_in_predicate.c
SRC_FINISH_JOIN := ${DPU_DIR}/join/finish_join.c
##############################

COMMON_INCLUDES := ./include

.PHONY: all clean

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
DPU_FLAGS := ${COMMON_FLAGS} -O3 -DDEBUG -DSTACK_SIZE_DEFAULT=256 -DCRC -DVALIDATION

all: \
$(A2A_SRJ_GLOBAL_PARTITION) \
$(A2A_SRJ_LOCAL_PARTITION)  \
$(A2A_SRJ_BUILD_LINEAR_PROBE)  \
$(A2A_SRJ_PROBE_INNER_LINEAR_PROBE) \
$(A2A_PACKETWISE_SRJ_LOCAL_PARTITION)  \
${FINISH_JOIN} \
$(GLB_PARTITION_COUNT) \
$(GLB_PARTITION_PACKET) \
$(MICROBENCHMARK_MRAM_BANDWIDTH) \
$(MICROBENCHMARK_WRAM_BANDWIDTH) \
$(MICROBENCHMARK_OPS) \
$(MPSM_JOIN_PARTITION) \
$(MPSM_JOIN_PROBE) \
$(MPSM_JOIN_SORT) \
$(MPSM_JOIN_RADIX_SORT) \
$(MPSM_JOIN_MERGE_SORT) \
$(MPSM_JOIN_QUICK_SORT) \
$(MPSM_JOIN_PROBE_ALL) \
$(NO_PARTITIONED_PROBE_INNER) \
$(NESTED_LOOP_JOIN) \
${MICROBENCHMARK_HASH} \


${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${MPSM_JOIN_PARTITION}: ${SRC_MPSM_JOIN_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_PARTITION}
${MPSM_JOIN_PROBE}: ${SRC_MPSM_JOIN_PROBE} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_PROBE}

${MPSM_JOIN_SORT}: ${SRC_MPSM_JOIN_SORT} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_SORT}
${MPSM_JOIN_RADIX_SORT}: ${SRC_MPSM_JOIN_RADIX_SORT} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_RADIX_SORT}
${MPSM_JOIN_MERGE_SORT}: ${SRC_MPSM_JOIN_MERGE_SORT} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_MERGE_SORT}
${MPSM_JOIN_QUICK_SORT}: ${SRC_MPSM_JOIN_QUICK_SORT} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_QUICK_SORT}
${MPSM_JOIN_PROBE_ALL}: ${SRC_MPSM_JOIN_PROBE_ALL} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MPSM_JOIN_PROBE_ALL}
${FINISH_JOIN}: ${SRC_FINISH_JOIN} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_FINISH_JOIN}
######################################################

${MICROBENCHMARK_MRAM_BANDWIDTH}: ${SRC_MICROBENCHMARK_MRAM_BANDWIDTH} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MICROBENCHMARK_MRAM_BANDWIDTH} 

${MICROBENCHMARK_WRAM_BANDWIDTH}: ${SRC_MICROBENCHMARK_WRAM_BANDWIDTH} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_MICROBENCHMARK_WRAM_BANDWIDTH} 

${MICROBENCHMARK_OPS}: ${SRC_MICROBENCHMARK_OPS} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=16 -o $@ ${SRC_MICROBENCHMARK_OPS} 

${MICROBENCHMARK_HASH}: ${SRC_MICROBENCHMARK_HASH} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=16 -o $@ ${SRC_MICROBENCHMARK_HASH} 

${DEBUGGER}: ${SRC_DEBUGGER} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_DEBUGGER} 

######################################################
${GLB_PARTITION_COUNT}: ${SRC_GLB_PARTITION_COUNT} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_GLB_PARTITION_COUNT} 

${GLB_PARTITION_PACKET}: ${SRC_GLB_PARTITION_PACKET} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_GLB_PARTITION_PACKET} 

${PACKET_PROJECTION}: ${SRC_PACKET_PROJECTION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_PACKET_PROJECTION} 

${TUPLE_ID_ORDER_PROJECTION}: ${SRC_TUPLE_ID_ORDER_PROJECTION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_TUPLE_ID_ORDER_PROJECTION} 

${A2A_PACKETWISE_ATTR_PARTITION_PACKET_128B}: ${SRC_A2A_PACKETWISE_ATTR_PARTITION_PACKET_128B} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_PACKETWISE_ATTR_PARTITION_PACKET_128B} 

${A2A_SRJ_GLOBAL_PARTITION}: ${SRC_A2A_SRJ_GLOBAL_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_SRJ_GLOBAL_PARTITION} 

${A2A_SRJ_GLOBAL_PARTITION_PREV}: ${SRC_A2A_SRJ_GLOBAL_PARTITION_PREV} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=${NR_TASKLETS} -o $@ ${SRC_A2A_SRJ_GLOBAL_PARTITION_PREV}

${A2A_SRJ_LOCAL_PARTITION}: ${SRC_A2A_SRJ_LOCAL_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_SRJ_LOCAL_PARTITION} 

${A2A_PACKETWISE_SRJ_LOCAL_PARTITION}: ${SRC_A2A_PACKETWISE_SRJ_LOCAL_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_PACKETWISE_SRJ_LOCAL_PARTITION} 

${A2A_PACKETWISE_SRJ_GLOBAL_PARTITION}: ${SRC_A2A_PACKETWISE_SRJ_GLOBAL_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_PACKETWISE_SRJ_GLOBAL_PARTITION} 

${A2A_SRJ_TID_PARTITION}: ${SRC_A2A_SRJ_TID_PARTITION} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=${NR_TASKLETS} -o $@ ${SRC_A2A_SRJ_TID_PARTITION} 

${NESTED_LOOP_JOIN}: ${SRC_NESTED_LOOP_JOIN} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_NESTED_LOOP_JOIN}

######################################################

${A2A_SRJ_BUILD_LINEAR_PROBE}: ${SRC_A2A_SRJ_BUILD_LINEAR_PROBE} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -DLINEAR_PROBE -o $@ ${SRC_A2A_SRJ_BUILD_LINEAR_PROBE} 

${A2A_SRJ_PROBE}: ${SRC_A2A_SRJ_PROBE} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_A2A_SRJ_PROBE}

${A2A_SRJ_PROBE_INNER_LINEAR_PROBE}: ${SRC_A2A_SRJ_PROBE_INNER_LINEAR_PROBE} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -DLINEAR_PROBE -o $@ ${SRC_A2A_SRJ_PROBE_INNER_LINEAR_PROBE}

${INT64_COL_SORT_W_TID}: ${SRC_INT64_COL_SORT_W_TID} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -DNR_TASKLETS=12 -o $@ ${SRC_INT64_COL_SORT_W_TID}
######################################################

clean:
	$(RM) -r $(BUILDDIR)

