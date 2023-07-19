#ifndef _HASH_H_
#define _HASH_H_

#include <stdio.h>
#include <defs.h>

uint32_t CRC_HASH(uint32_t key);

uint32_t One_at_a_time(uint32_t key);

uint32_t Tabulation(uint32_t key);

uint32_t MurmurHash3 (uint32_t key, int len, uint32_t seed);

uint32_t multiply_shift(uint32_t key);

uint32_t multiply_add_shift(uint32_t key);

uint32_t SuperFastHash(uint32_t Key);

uint32_t sdbm(uint32_t key);

uint32_t rand(int seed);

#endif