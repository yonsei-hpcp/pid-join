
#define _GNU_SOURCE
#include <stdint.h>

#include "dpu_region_address_translation.h"
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <dpu.h>
#include <errno.h>
#include <limits.h>
#include <numa.h>
#include <numaif.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <sys/sysinfo.h>
#include <time.h>
#include "static_verbose.h"

#define BANK_START(dpu_id) (0x40000 * ((dpu_id) % 4) + ((dpu_id >= 4) ? 0x40 : 0))
#define BANK_START_OPT(dpu_id) (0x40000 * ((dpu_id)&3))
#define BANK_OFFSET_NEXT_DATA(i) (i * 16) // For each 64bit word, you must jump 16 * 64bit (2 cache lines)
#define BANK_CHUNK_SIZE 0x20000
#define BANK_NEXT_CHUNK_OFFSET 0x100000
#define MRAM_SIZE 0x4000000 // 64MB

#define CACHE_LINE 64
#define CACHE_LINE2 128

static uint32_t apply_address_translation_on_mram_offset(uint32_t byte_offset)
{
    /* We have observed that, within the 26 address bits of the MRAM address, we need to apply an address translation:
     *
     * virtual[13: 0] = physical[13: 0]
     * virtual[20:14] = physical[21:15]
     * virtual[   21] = physical[   14]
     * virtual[25:22] = physical[25:22]
     *
     * This function computes the "virtual" mram address based on the given "physical" mram address.
     */

    uint32_t mask_21_to_15 = ((1 << (21 - 15 + 1)) - 1) << 15;
    uint32_t mask_21_to_14 = ((1 << (21 - 14 + 1)) - 1) << 14;
    uint32_t bits_21_to_15 = (byte_offset & mask_21_to_15) >> 15;
    uint32_t bit_14 = (byte_offset >> 14) & 1;
    uint32_t unchanged_bits = byte_offset & ~mask_21_to_14;

    return unchanged_bits | (bits_21_to_15 << 14) | (bit_14 << 21);
}

static uint64_t address_offset_change(uint32_t byte_offset)
{
    /* We have observed that, within the 26 address bits of the MRAM address, we need to apply an address translation:
     *
     * virtual[13: 0] = physical[13: 0]
     * virtual[20:14] = physical[21:15]
     * virtual[   21] = physical[   14]
     * virtual[25:22] = physical[25:22]
     *
     * This function computes the "virtual" mram address based on the given "physical" mram address.
     */

    uint32_t mask_21_to_15 = ((1 << (21 - 15 + 1)) - 1) << 15;
    uint32_t mask_21_to_14 = ((1 << (21 - 14 + 1)) - 1) << 14;
    uint32_t bits_21_to_15 = (byte_offset & mask_21_to_15) >> 15;
    uint32_t bit_14 = (byte_offset >> 14) & 1;
    uint32_t unchanged_bits = byte_offset & ~mask_21_to_14;
    uint32_t mram_64_bit_word_offset = unchanged_bits | (bits_21_to_15 << 14) | (bit_14 << 21);
    uint64_t next_data = mram_64_bit_word_offset << 4;
    uint64_t host_side_offset = (next_data & (BANK_CHUNK_SIZE - 1)) + (next_data / BANK_CHUNK_SIZE) * BANK_NEXT_CHUNK_OFFSET;
    return host_side_offset;
}

void PrintJobInfo(rotate_n_stream_job_t *job)
{
    printf(
        "job->job_priority: %lf\n"
        "job->src_rank: %d\n"
        "job->dst_rank: %d\n"
        "job->mram_src_offset: %d\n"
        "job->mram_dst_offset: %d\n"
        "job->job_type: %d\n"
        "job->src_packet_num: %f\n"
        "job->bankchunk_8_offset: %d\n",
        job->job_priority,
        job->src_rank,
        job->dst_rank,
        job->mram_src_offset,
        job->mram_dst_offset,
        job->job_type,
        job->src_packet_num,
        job->bankchunk_8_offset);
}

/////////////////////////////////////////////////////////

#define RNS_SRC_BG 4
#define RNS_TAR_BG 8

// Where RNS really happens. (Read/Write 64B*2) * (packet_size / 8) = 2KB / (128 / packet_size) = 1 packet from 16 chips - Comment
#define RNS_COPY(rotate, rotate_bit, iter, src_rank_bgwise_addr, dst_rank_bgwise_addr)    \
    do                                                                                    \
    {                                                                                     \
        void *src_rank_clwise_addr = src_rank_bgwise_addr;                                \
        void *dst_rank_clwise_addr1 = dst_rank_bgwise_addr;                               \
        void *dst_rank_clwise_addr2 = dst_rank_bgwise_addr + (64 * 2 * iter);             \
                                                                                          \
        __m512i reg1;                                                                     \
        __m512i reg2;                                                                     \
        __m512i reg1_rot;                                                                 \
        __m512i reg2_rot;                                                                 \
                                                                                          \
        for (int cl = 0; cl < iter; cl++)                                                 \
        {                                                                                 \
            reg1 = _mm512_stream_load_si512((void *)(src_rank_clwise_addr));              \
            reg2 = _mm512_stream_load_si512((void *)(src_rank_clwise_addr + CACHE_LINE)); \
            reg1_rot = _mm512_rol_epi64(reg1, rotate_bit);                                \
            reg2_rot = _mm512_rol_epi64(reg2, rotate_bit);                                \
            _mm512_stream_si512((void *)(dst_rank_clwise_addr1), reg1_rot);               \
            _mm512_stream_si512((void *)(dst_rank_clwise_addr2), reg2_rot);               \
            src_rank_clwise_addr += CACHE_LINE2;                                          \
            dst_rank_clwise_addr1 += CACHE_LINE2;                                         \
            dst_rank_clwise_addr2 += CACHE_LINE2;                                         \
        }                                                                                 \
    } while (0)

#define RNS_FLUSH_DST(iter, dst_rank_bgwise_addr)                                  \
    do                                                                             \
    {                                                                              \
        void *dst_rank_clwise_addr1 = dst_rank_bgwise_addr;                        \
        void *dst_rank_clwise_addr2 = dst_rank_bgwise_addr + (CACHE_LINE2 * iter); \
                                                                                   \
        for (int cl = 0; cl < iter; cl++)                                          \
        {                                                                          \
            __builtin_ia32_clflushopt((uint8_t *)dst_rank_clwise_addr1);           \
            __builtin_ia32_clflushopt((uint8_t *)(dst_rank_clwise_addr2));         \
            dst_rank_clwise_addr1 += CACHE_LINE2;                                  \
            dst_rank_clwise_addr2 += CACHE_LINE2;                                  \
        }                                                                          \
    } while (0)

#define RNS_FLUSH_SRC(iter, src_rank_bgwise_addr)                                      \
    do                                                                                 \
    {                                                                                  \
        void *src_rank_clwise_addr = src_rank_bgwise_addr;                             \
                                                                                       \
        for (int cl = 0; cl < iter; cl++)                                              \
        {                                                                              \
            __builtin_ia32_clflushopt((uint8_t *)src_rank_clwise_addr);                \
            __builtin_ia32_clflushopt((uint8_t *)(src_rank_clwise_addr + CACHE_LINE)); \
            /*_mm_mfence(); */                                                         \
            src_rank_clwise_addr += CACHE_LINE2;                                       \
        }                                                                              \
    } while (0)

static void xeon_sp_do_rnc_thread(RNS_Job_Queue_t *queue, rotate_n_stream_job_t *job)
{
    int num_packets = job->src_packet_num;
    int packet_size = job->packet_size;
    int offset_gran = packet_size * 64;
    int iteration = packet_size / 8;
    int src_off_gran = CACHE_LINE2 * iteration;
    int dst_off_gran = src_off_gran * 2;

    int src_bg_offset = (packet_size * 1024) / (4);

    if ((job->mram_src_offset & (offset_gran - 1)) != 0) // 8KB offset will be inside 1MB address space - Comment
    {
        printf("%sError: mram_src_offset %d is not aligned.\n", "\x1B[36m", job->mram_src_offset);
        PrintJobInfo(job);
        exit(-1);
    }

    if ((job->mram_dst_offset & (offset_gran - 1)) != 0)
    {
        printf("%sError: mram_dst_offset %d is not aligned.\n", "\x1B[36m", job->mram_dst_offset);
        PrintJobInfo(job);
        exit(-1);
    }

    for (int rp = 0; rp < job->num_repeat; rp++)
    {
        int temp_src_mram_offset = job->mram_src_offset + (1024 * 1024); // + 1MB. for log only when xfer size does not exceed 63MB - Comment
        int temp_dst_mram_offset = job->mram_dst_offset + (1024 * 1024);

        if (temp_src_mram_offset >= MRAM_SIZE)
        {
            printf("%sError: mram_dst_offset %d >= 64MB.\n", "\x1B[36m", job->mram_dst_offset);
            PrintJobInfo(job);
            exit(-1);
        }
        if (temp_dst_mram_offset >= MRAM_SIZE)
        {
            printf("%sError: mram_dst_offset %d >= 64MB.\n", "\x1B[36m", job->mram_dst_offset);
            PrintJobInfo(job);
            exit(-1);
        }

        void *src_rank_base_addr = (void *)(queue->ptr_regions[job->src_rank]);
        void *dst_rank_base_addr = (void *)(queue->ptr_regions[job->dst_rank]);

#ifdef RNS_DEBUG
        int64_t xfer_byte = 0;
#endif
        for (int p = 0; p < num_packets; p++)
        {
            ////////////////////////////////////////////
            // Address calculation Start
            ////////////////////////////////////////////

            int64_t mram_src_offset_1mb_wise;
            int64_t mram_dst_offset_1mb_wise;

            uint32_t mram_64_bit_word_offset = apply_address_translation_on_mram_offset(temp_src_mram_offset);
            uint64_t next_data = BANK_OFFSET_NEXT_DATA(mram_64_bit_word_offset);
            mram_src_offset_1mb_wise = (next_data % BANK_CHUNK_SIZE) + (next_data / BANK_CHUNK_SIZE) * BANK_NEXT_CHUNK_OFFSET;

            mram_64_bit_word_offset = apply_address_translation_on_mram_offset(temp_dst_mram_offset);
            next_data = BANK_OFFSET_NEXT_DATA(mram_64_bit_word_offset);
            mram_dst_offset_1mb_wise = (next_data % BANK_CHUNK_SIZE) + (next_data / BANK_CHUNK_SIZE) * BANK_NEXT_CHUNK_OFFSET;

            ////////////////////////////////////////////
            // Address calculation Done
            ////////////////////////////////////////////

            void *src_rank_addr = src_rank_base_addr + mram_src_offset_1mb_wise;
            void *dst_rank_addr = dst_rank_base_addr + mram_dst_offset_1mb_wise;

            _mm_mfence();

            for (int src_bg = 0; src_bg < RNS_SRC_BG; src_bg++)
            {
                void *src_rank_addr__ = src_rank_addr + (src_bg * (256 * 1024));

                for (int target_bg = 0; target_bg < RNS_TAR_BG; target_bg++)
                {
// bg even
#ifndef RNS_DEBUG
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
#else
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
                    src_rank_addr__ += (src_off_gran);
#endif
                }
            }

            _mm_mfence();

            for (int src_bg = 0; src_bg < RNS_SRC_BG; src_bg++) // 4 * 128KB / (128 / packet_size) = 512KB / (128 / packet_size) : in one bank interleaved space (1MB) - Comment
            {
                void *src_rank_addr__ = src_rank_addr + (src_bg * (256 * 1024));
                // printf("temp_src_mram_offset: %d\n", temp_src_mram_offset);

                for (int target_bg = 0; target_bg < RNS_TAR_BG; target_bg++) // 8 * (8 * packet_size * 16) = 128KB / (128 / packet_size) : in one bank chunk space - Comment
                {
                    void *dst_rank_addr__ = dst_rank_addr + ((target_bg % 4) * (256 * 1024) + (target_bg / 4) * CACHE_LINE) + (src_bg * (src_bg_offset));

                    // bg even
                    RNS_COPY(0, 0, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(1, 8, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(2, 16, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(3, 24, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(4, 32, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(5, 40, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(6, 48, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_COPY(7, 56, iteration, src_rank_addr__, dst_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    dst_rank_addr__ += dst_off_gran;
                }
            }

            _mm_mfence();

            for (int src_bg = 0; src_bg < RNS_SRC_BG; src_bg++)
            {
                void *src_rank_addr__ = src_rank_addr + (src_bg * (256 * 1024));

                for (int target_bg = 0; target_bg < RNS_TAR_BG; target_bg++)
                {
                    // bg even
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                    RNS_FLUSH_SRC(iteration, src_rank_addr__);
                    src_rank_addr__ += (src_off_gran);
                }
            }

            for (int src_bg = 0; src_bg < RNS_SRC_BG; src_bg++)
            {
                for (int target_bg = 0; target_bg < RNS_TAR_BG; target_bg++)
                {
                    void *dst_rank_addr__ = dst_rank_addr + ((target_bg % 4) * (256 * 1024) + (target_bg / 4) * CACHE_LINE) + (src_bg * (src_bg_offset));
                    // bg even
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                    RNS_FLUSH_DST(iteration, dst_rank_addr__);
                    dst_rank_addr__ += dst_off_gran;
                }
            }

            temp_src_mram_offset += offset_gran;
            temp_dst_mram_offset += offset_gran;

            _mm_mfence(); // every packet_size * 64 * 64 = 512KB / (128 / packet_size) transfer - Comment
        }
    }
#ifdef RNS_DEBUG
// printf("[%s] xfer byte : %ld \n", __func__, xfer_byte);
#endif
}

static void xeon_sp_do_copy_from_pim_opt(RNS_Job_Queue_t *queue, rotate_n_stream_job_t *job)
{
    if ((job->mram_src_offset & (8192 - 1)) != 0)
    {
        printf("%sError: mram_src_offset %d is not aligned.\n", "\x1B[36m", job->mram_src_offset);
        PrintJobInfo(job);
        exit(-1);
    }
    int temp_src_mram_offset = job->mram_src_offset + (1024 * 1024);

    if (temp_src_mram_offset >= (64 * 1024 * 1024))
    {
        printf("%sError: mram_src_offset %d >= 64MB.\n", "\x1B[36m", job->mram_src_offset);
        PrintJobInfo(job);
        exit(-1);
    }

    uint8_t *src_rank_base_addr = (uint8_t *)(queue->ptr_regions[job->src_rank]);
    uint32_t size_transfer = job->xfer_bytes;
    uint64_t *buff = (uint64_t *)job->host_buffer;

    int32_t num_packets_per_dpu = ((size_transfer / 64) / 8192);
    int32_t leftover_bytes_per_dpu = ((size_transfer / 64) & (8192 - 1));

    register __m512i mask;
    mask = _mm512_set_epi64(
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL);

    register __m512i perm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    register __m512i perm_32bit = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    // // do flush
    // {
    //     temp_src_mram_offset = job->mram_src_offset + (1024 * 1024);

    //     for (int p = 0; p < num_packets_per_dpu; p++)
    //     {
    //         uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);
    //         uint8_t *src_rank_addr_bg = src_rank_base_addr + host_side_offset;

    //         for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
    //         {
    //             uint64_t *dpu_src_addr = (uint64_t *)(src_rank_addr_bg + BANK_START_OPT(src_dpu_group));

    //             // COpying 2 bank group

    //             for (uint32_t e = 0; e < ((8192 / (sizeof(int64_t) * 2))); e++)
    //             {
    //                 _mm_clflushopt((__m512i *)dpu_src_addr);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 1);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 2);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 3);

    //                 dpu_src_addr = dpu_src_addr + 32;
    //             }
    //         }

    //         temp_src_mram_offset += 8192;
    //     }

    //     if (leftover_bytes_per_dpu > 0)
    //     {
    //         uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);

    //         for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
    //         {
    //             uint64_t *dpu_src_addr = (uint64_t *)(src_rank_base_addr + host_side_offset + BANK_START_OPT(src_dpu_group));
    //             // COpying 2 bank group
    //             for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
    //             {
    //                 _mm_clflushopt((__m512i *)dpu_src_addr);
    //                 dpu_src_addr += 8;
    //             }
    //         }

    //         temp_src_mram_offset += 8192;
    //     }
    // }
    __builtin_ia32_mfence();
    // do copy
    {
        temp_src_mram_offset = job->mram_src_offset + (1024 * 1024);

        for (int p = 0; p < num_packets_per_dpu; p++)
        {
            uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);
            uint8_t *src_rank_addr_bg = src_rank_base_addr + host_side_offset;

            for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
            {
                uint64_t *dpu_src_addr = (uint64_t *)(src_rank_addr_bg + BANK_START_OPT(src_dpu_group));

                // COpying 2 bank group

                for (uint32_t e = 0; e < ((8192 / (sizeof(int64_t) * 2))); e++)
                {
                    __m512i load_before_permute_ur0 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr));
                    __m512i load_before_permute2_ur0 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 1));
                    __m512i load_before_permute_ur1 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 2));
                    __m512i load_before_permute2_ur1 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 3));

                    __m512i load_ur0 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute_ur0);
                    __m512i load2_ur0 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2_ur0);
                    __m512i load_ur1 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute_ur1);
                    __m512i load2_ur1 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2_ur1);

                    __m512i transpose_ur0 = _mm512_shuffle_epi8(load_ur0, mask);
                    __m512i transpose2_ur0 = _mm512_shuffle_epi8(load2_ur0, mask);
                    __m512i transpose_ur1 = _mm512_shuffle_epi8(load_ur1, mask);
                    __m512i transpose2_ur1 = _mm512_shuffle_epi8(load2_ur1, mask);

                    __m512i final_ur0 = _mm512_permutexvar_epi32(perm, transpose_ur0);
                    __m512i final2_ur0 = _mm512_permutexvar_epi32(perm, transpose2_ur0);
                    __m512i final_ur1 = _mm512_permutexvar_epi32(perm, transpose_ur1);
                    __m512i final2_ur1 = _mm512_permutexvar_epi32(perm, transpose2_ur1);

                    _mm512_store_epi64((void *)((__m512i *)buff), final_ur0);
                    _mm512_store_epi64((void *)((__m512i *)buff + 1), final2_ur0);
                    _mm512_store_epi64((void *)((__m512i *)buff + 2), final_ur1);
                    _mm512_store_epi64((void *)((__m512i *)buff + 3), final2_ur1);

                    dpu_src_addr = dpu_src_addr + 32;
                    buff = buff + 32;
                }
            }

            temp_src_mram_offset += 8192;
        }

        if (leftover_bytes_per_dpu > 0)
        {
            uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);

            for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
            {
                uint64_t *dpu_src_addr = (uint64_t *)(src_rank_base_addr + host_side_offset + BANK_START_OPT(src_dpu_group));
                // COpying 2 bank group
                for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
                {
                    __m512i load_before_permute = _mm512_stream_load_si512((void *)(dpu_src_addr));
                    dpu_src_addr += 8;
                    __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                    __m512i transpose = _mm512_shuffle_epi8(load, mask);
                    __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                    _mm512_store_epi64((void *)(buff), final);
                    buff += 8;
                }
            }

            temp_src_mram_offset += 8192;
        }
    }
    __builtin_ia32_mfence();
    // // do flush
    // {
    //     temp_src_mram_offset = job->mram_src_offset + (1024 * 1024);

    //     for (int p = 0; p < num_packets_per_dpu; p++)
    //     {
    //         uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);
    //         uint8_t *src_rank_addr_bg = src_rank_base_addr + host_side_offset;

    //         for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
    //         {
    //             uint64_t *dpu_src_addr = (uint64_t *)(src_rank_addr_bg + BANK_START_OPT(src_dpu_group));

    //             // COpying 2 bank group

    //             for (uint32_t e = 0; e < ((8192 / (sizeof(int64_t) * 2))); e++)
    //             {
    //                 _mm_clflushopt((__m512i *)dpu_src_addr);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 1);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 2);
    //                 _mm_clflushopt((__m512i *)dpu_src_addr + 3);

    //                 dpu_src_addr = dpu_src_addr + 32;
    //             }
    //         }

    //         temp_src_mram_offset += 8192;
    //     }

    //     if (leftover_bytes_per_dpu > 0)
    //     {
    //         uint64_t host_side_offset = address_offset_change(temp_src_mram_offset);

    //         for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
    //         {
    //             uint64_t *dpu_src_addr = (uint64_t *)(src_rank_base_addr + host_side_offset + BANK_START_OPT(src_dpu_group));
    //             // COpying 2 bank group
    //             for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
    //             {
    //                 _mm_clflushopt((__m512i *)dpu_src_addr);
    //                 dpu_src_addr += 8;
    //             }
    //         }

    //         temp_src_mram_offset += 8192;
    //     }
    // }

    __builtin_ia32_mfence();
}

static void xeon_sp_do_copy_to_pim_opt(RNS_Job_Queue_t *queue, rotate_n_stream_job_t *job)
{
    if ((job->mram_dst_offset & (8192 - 1)) != 0)
    {
        printf("%sError: mram_dst_offset %d is not aligned.\n", "\x1B[36m", job->mram_dst_offset);
        PrintJobInfo(job);
        exit(-1);
    }
    int temp_dst_mram_offset = job->mram_dst_offset + (1024 * 1024);

    if (temp_dst_mram_offset >= (64 * 1024 * 1024))
    {
        printf("%sError: mram_dst_offset %d >= 64MB.\n", "\x1B[36m", job->mram_dst_offset);
        PrintJobInfo(job);
        exit(-1);
    }

    uint8_t *dst_rank_base_addr = (uint8_t *)(queue->ptr_regions[job->dst_rank]);
    uint32_t size_transfer = job->xfer_bytes;
    uint64_t *buff = (uint64_t *)job->host_buffer;

    int32_t num_packets_per_dpu = ((size_transfer / 64) / 8192);
    int32_t leftover_bytes_per_dpu = ((size_transfer / 64) & (8192 - 1));

    __m512i mask;
    mask = _mm512_set_epi64(
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL,
        0x0f0b07030e0a0602ULL,
        0x0d0905010c080400ULL);

    __m512i perm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    __m512i perm_32bit = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    for (int p = 0; p < num_packets_per_dpu; p++)
    {
        uint64_t host_side_offset = address_offset_change(temp_dst_mram_offset);
        uint8_t *dst_rank_addr_bg = dst_rank_base_addr + host_side_offset;

        for (int dst_dpu_group = 0; dst_dpu_group < 4; dst_dpu_group++)
        {
            uint64_t *dpu_dst_addr = (uint64_t *)(dst_rank_addr_bg + BANK_START_OPT(dst_dpu_group));

            // Copying 2 bank group
            for (uint32_t e = 0; e < (8192 / sizeof(int64_t)); e++)
            {
                __m512i load_before_permute = _mm512_loadu_si512((void *)(buff));
                __m512i load_before_permute2 = _mm512_loadu_si512((void *)(buff + 8));
                buff += 16;

                __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                __m512i load2 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2);
                __m512i transpose = _mm512_shuffle_epi8(load, mask);
                __m512i transpose2 = _mm512_shuffle_epi8(load2, mask);
                __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                __m512i final2 = _mm512_permutexvar_epi32(perm, transpose2);

                _mm512_stream_si512((void *)(dpu_dst_addr), final);
                _mm512_stream_si512((void *)(dpu_dst_addr + 8), final2);
                dpu_dst_addr += 16;
            }
        }

        temp_dst_mram_offset += 8192;
    }

    // printf("leftover_bytes_per_dpu: %d\n", leftover_bytes_per_dpu);
    if (leftover_bytes_per_dpu > 0)
    {
        uint64_t host_side_offset = address_offset_change(temp_dst_mram_offset);

        for (int dst_dpu_group = 0; dst_dpu_group < 4; dst_dpu_group++)
        {
            uint64_t *dpu_dst_addr = (uint64_t *)(dst_rank_base_addr + host_side_offset + BANK_START_OPT(dst_dpu_group));
            // COpying 2 bank group
            for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
            {
                __m512i load_before_permute = _mm512_loadu_si512((void *)(buff));
                buff += 8;
                __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                __m512i transpose = _mm512_shuffle_epi8(load, mask);
                __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                _mm512_stream_si512((void *)(dpu_dst_addr), final);
                dpu_dst_addr += 8;
            }
        }

        temp_dst_mram_offset += 8192;
    }

    __builtin_ia32_mfence();
}

void *xeon_sp_do_rnc_worker( // rnc worker thread do jo - Comment
    void *queue)
{
    RNS_Job_Queue_t *job_queue = (RNS_Job_Queue_t *)queue;

    while (true)
    {
        rotate_n_stream_job_t *new_job = job_queue->rnc_queue_next_job(job_queue);
        // Terminate Condition
        if (new_job == NULL)
        {
            // printf("No More New Job...\n");
            break;
        }

        switch (new_job->job_type)
        {
        case DPU_TRANSFER_JOB_TYPE_UNORDERED_SCATTER:
        {
            xeon_sp_do_copy_to_pim_opt(job_queue, new_job);
            pthread_mutex_lock(new_job->mutex);
            *(new_job->running_jobs_per_rank) -= 1;
            pthread_cond_signal(new_job->cond);
            pthread_mutex_unlock(new_job->mutex);
            continue; // do not call return job
        };
        break;
        case DPU_TRANSFER_JOB_TYPE_UNORDERED_GATHER:
        {
            xeon_sp_do_copy_from_pim_opt(job_queue, new_job);
            pthread_mutex_lock(new_job->mutex);
            *(new_job->running_jobs_per_rank) -= 1;
            pthread_cond_signal(new_job->cond);
            pthread_mutex_unlock(new_job->mutex);
            continue; // do not call return job
        }
        break;
        case DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_32B:
        {
            new_job->packet_size = 32;
            xeon_sp_do_rnc_thread(job_queue, new_job);
        };
        break;
        case DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_16B:
        {
            new_job->packet_size = 16;
            xeon_sp_do_rnc_thread(job_queue, new_job);
        };
        break;
        case DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_8B:
        {
            new_job->packet_size = 8;
            xeon_sp_do_rnc_thread(job_queue, new_job);
        };
        break;
        case DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_128B: // DPU -> DPU - Comment
        {
            new_job->packet_size = 128;
            xeon_sp_do_rnc_thread(job_queue, new_job);
        }
        break;
        case DPU_TRANSFER_JOB_TYPE_ROTATE_AND_BYPASS_PACKET_SIZE_64B: // DPU -> DPU - Comment
        {
            new_job->packet_size = 64;
            xeon_sp_do_rnc_thread(job_queue, new_job);
        }
        break;
        default:
            printf("Error: new_job->job_type:%d addr %p\n", new_job->job_type, new_job);
            break;
        }

        job_queue->return_job(job_queue, new_job);
    }
    // printf("xeon_sp_do_rnc_worker Terminated.\n");
    return NULL;
}

void xeon_sp_do_rnc(RNS_Job_Queue_t *job_queue) // rnc worker thread creation - Comment
{
#ifdef VERBOSE
    printf("[%s] Start\n", __func__);
#endif
    for (int t = 0; t < job_queue->worker_num; t++)
    {
        pthread_create(&(job_queue->worker_threads_numa[t]), NULL, &xeon_sp_do_rnc_worker, job_queue);
    }
    return;
}

void xeon_sp_rot_n_stream(
    __attribute__((unused)) struct dpu_region_address_translation *tr_src,
    __attribute__((unused)) struct dpu_region_address_translation *tr_dst,
    __attribute__((unused)) void *base_region_addr_src,
    __attribute__((unused)) void *base_region_addr_dst,
    __attribute__((unused)) struct dpu_transfer_matrix *xfer_matrix_src,
    __attribute__((unused)) struct dpu_transfer_matrix *xfer_matrix_dst)
{
    // deprecated.
    return;
}


/**
 * @brief direction = true (scatter) false (gather)
 * 
 */
void xeon_sp_unordered_data_transfer_rankwise(
    __attribute__((unused)) struct dpu_region_address_translation *tr,
    __attribute__((unused)) void *base_region_addr,
    __attribute__((unused)) uint32_t mram_address,
    __attribute__((unused)) void *data,
    __attribute__((unused)) size_t length,
    __attribute__((unused)) bool direction,
    __attribute__((unused)) int num_threads)
{

    // unordered Scatter
    if (direction == true)
    {
        int temp_mram_address = mram_address + (1024 * 1024);

        if ((mram_address & (8192 - 1)) != 0)
        {
            printf("%sError: mram_address %d is not aligned.\n", "\x1B[36m", mram_address);
            exit(-1);
        }
        
        if (temp_mram_address >= (64 * 1024 * 1024))
        {
            printf("%sError: mram_address %d >= 64MB.\n", "\x1B[36m", mram_address);
            exit(-1);
        }

        uint8_t *dst_rank_base_addr = base_region_addr;
	uint32_t size_transfer = length;
        uint64_t *buff = (uint64_t*)data;

        int32_t num_packets_per_dpu = ((size_transfer / 64) / 8192);
        int32_t leftover_bytes_per_dpu = ((size_transfer / 64) & (8192 - 1));

        __m512i mask;
        mask = _mm512_set_epi64(
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL);

        __m512i perm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
        __m512i perm_32bit = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        for (int p = 0; p < num_packets_per_dpu; p++)
        {
            uint64_t host_side_offset = address_offset_change(temp_mram_address);
            uint8_t *dst_rank_addr_bg = dst_rank_base_addr + host_side_offset;

            for (int dst_dpu_group = 0; dst_dpu_group < 4; dst_dpu_group++)
            {
                uint64_t *dpu_dst_addr = (uint64_t *)(dst_rank_addr_bg + BANK_START_OPT(dst_dpu_group));

                // Copying 2 bank group
                for (uint32_t e = 0; e < (8192 / sizeof(int64_t)); e++)
                {
                    __m512i load_before_permute = _mm512_loadu_si512((void *)(buff));
                    __m512i load_before_permute2 = _mm512_loadu_si512((void *)(buff + 8));
                    buff += 16;

                    __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                    __m512i load2 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2);
                    __m512i transpose = _mm512_shuffle_epi8(load, mask);
                    __m512i transpose2 = _mm512_shuffle_epi8(load2, mask);
                    __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                    __m512i final2 = _mm512_permutexvar_epi32(perm, transpose2);

                    _mm512_stream_si512((void *)(dpu_dst_addr), final);
                    _mm512_stream_si512((void *)(dpu_dst_addr + 8), final2);
                    dpu_dst_addr += 16;
                }
            }

           temp_mram_address += 8192;
        }

        // printf("leftover_bytes_per_dpu: %d\n", leftover_bytes_per_dpu);
        if (leftover_bytes_per_dpu > 0)
        {
            uint64_t host_side_offset = address_offset_change(temp_mram_address);

            for (int dst_dpu_group = 0; dst_dpu_group < 4; dst_dpu_group++)
            {
                uint64_t *dpu_dst_addr = (uint64_t *)(dst_rank_base_addr + host_side_offset + BANK_START_OPT(dst_dpu_group));
                // COpying 2 bank group
                for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
                {
                    __m512i load_before_permute = _mm512_loadu_si512((void *)(buff));
                    buff += 8;
                    __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                    __m512i transpose = _mm512_shuffle_epi8(load, mask);
                    __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                    _mm512_stream_si512((void *)(dpu_dst_addr), final);
                    dpu_dst_addr += 8;
                }
            }

            temp_mram_address += 8192;
        }

        __builtin_ia32_mfence();
    }
    // unordered Gather
    else
    {
        int temp_mram_address = mram_address + (1024 * 1024);

        if ((mram_address & (8192 - 1)) != 0)
        {
            printf("%sError: mram_address %d is not aligned.\n", "\x1B[36m", mram_address);
            exit(-1);
        }

        if (temp_mram_address >= (64 * 1024 * 1024))
        {
            printf("%sError: mram_address %d >= 64MB.\n", "\x1B[36m", mram_address);
            exit(-1);
        }

        uint8_t *src_rank_base_addr = (uint8_t*)base_region_addr;
        uint32_t size_transfer = length;
        uint64_t *buff = (uint64_t *)data;

        int32_t num_packets_per_dpu = ((size_transfer / 64) / 8192);
        int32_t leftover_bytes_per_dpu = ((size_transfer / 64) & (8192 - 1));

        register __m512i mask;
        mask = _mm512_set_epi64(
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL,
            0x0f0b07030e0a0602ULL,
            0x0d0905010c080400ULL);

        register __m512i perm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
        register __m512i perm_32bit = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        __builtin_ia32_mfence();
        // do copy
        {
            temp_mram_address = mram_address + (1024 * 1024);

            for (int p = 0; p < num_packets_per_dpu; p++)
            {
                uint64_t host_side_offset = address_offset_change(temp_mram_address);
                uint8_t *src_rank_addr_bg = src_rank_base_addr + host_side_offset;

                for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
                {
                    uint64_t *dpu_src_addr = (uint64_t *)(src_rank_addr_bg + BANK_START_OPT(src_dpu_group));

                    // COpying 2 bank group

                    for (uint32_t e = 0; e < ((8192 / (sizeof(int64_t) * 2))); e++)
                    {
                        __m512i load_before_permute_ur0 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr));
                        __m512i load_before_permute2_ur0 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 1));
                        __m512i load_before_permute_ur1 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 2));
                        __m512i load_before_permute2_ur1 = _mm512_stream_load_si512((void *)((__m512i *)dpu_src_addr + 3));

                        __m512i load_ur0 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute_ur0);
                        __m512i load2_ur0 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2_ur0);
                        __m512i load_ur1 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute_ur1);
                        __m512i load2_ur1 = _mm512_permutexvar_epi32(perm_32bit, load_before_permute2_ur1);

                        __m512i transpose_ur0 = _mm512_shuffle_epi8(load_ur0, mask);
                        __m512i transpose2_ur0 = _mm512_shuffle_epi8(load2_ur0, mask);
                        __m512i transpose_ur1 = _mm512_shuffle_epi8(load_ur1, mask);
                        __m512i transpose2_ur1 = _mm512_shuffle_epi8(load2_ur1, mask);

                        __m512i final_ur0 = _mm512_permutexvar_epi32(perm, transpose_ur0);
                        __m512i final2_ur0 = _mm512_permutexvar_epi32(perm, transpose2_ur0);
                        __m512i final_ur1 = _mm512_permutexvar_epi32(perm, transpose_ur1);
                        __m512i final2_ur1 = _mm512_permutexvar_epi32(perm, transpose2_ur1);

                        _mm512_store_epi64((void *)((__m512i *)buff), final_ur0);
                        _mm512_store_epi64((void *)((__m512i *)buff + 1), final2_ur0);
                        _mm512_store_epi64((void *)((__m512i *)buff + 2), final_ur1);
                        _mm512_store_epi64((void *)((__m512i *)buff + 3), final2_ur1);

                        dpu_src_addr = dpu_src_addr + 32;
                        buff = buff + 32;
                    }
                }

                temp_mram_address += 8192;
            }

            if (leftover_bytes_per_dpu > 0)
            {
                uint64_t host_side_offset = address_offset_change(temp_mram_address);

                for (int src_dpu_group = 0; src_dpu_group < 4; src_dpu_group++)
                {
                    uint64_t *dpu_src_addr = (uint64_t *)(src_rank_base_addr + host_side_offset + BANK_START_OPT(src_dpu_group));
                    // COpying 2 bank group
                    for (int32_t e = 0; e < ((leftover_bytes_per_dpu >> 3) << 1); ++e)
                    {
                        __m512i load_before_permute = _mm512_stream_load_si512((void *)(dpu_src_addr));
                        dpu_src_addr += 8;
                        __m512i load = _mm512_permutexvar_epi32(perm_32bit, load_before_permute);
                        __m512i transpose = _mm512_shuffle_epi8(load, mask);
                        __m512i final = _mm512_permutexvar_epi32(perm, transpose);
                        _mm512_store_epi64((void *)(buff), final);
                        buff += 8;
                    }
                }

                temp_mram_address += 8192;
            }
        }
        
        __builtin_ia32_mfence();
    }

    return;
}
