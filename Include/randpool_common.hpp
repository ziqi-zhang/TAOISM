// #ifndef RAND_POOL_H
// #define RAND_POOL_H

// #include "assert.h"
// #include "aes_stream_common.hpp"
// #include <unsupported/Eigen/CXX11/Tensor>

// typedef unsigned char SeedT[AES_STREAM_SEEDBYTES];

// #define STATE_LEN ((AES_STREAM_ROUNDS) + 1) * 16 + 16
// unsigned char init_key[STATE_LEN] = {0x00};	// TODO generate at random
// unsigned char init_seed[AES_STREAM_SEEDBYTES] = {0x00}; //TODO generate at random


// void init_PRG(aes_stream_state* state) {
// 	std::copy(init_key, init_key + STATE_LEN , state->opaque);
// 	aes_stream_init(state, init_seed);
// }

// void InitPrgWithSeed(aes_stream_state* state, const SeedT seed) {
// 	std::copy(init_key, init_key + STATE_LEN , state->opaque);
//     // state->opaque[0] = 0x1;

//     // _aes_stream_state *_st = (_aes_stream_state *) (void *) state;
//     // _aes_stream_state *_st = reinterpret_cast<_aes_stream_state *>(state);

//     // __m128i a;
//     // __m128i b;
//     // uint8_t array[16] = {};
//     // a = _mm_load_si128( (__m128i*) &array[0] );
//     // _st->round_keys[0] = _mm_load_si128( (__m128i*) &array[0] );
//     // _st->round_keys[0] = a;
//     // a = _st->round_keys[0];

// 	aes_stream_init(state, seed);
// 	// aes_stream_init(state, init_seed);
// }

// void get_PRG(aes_stream_state* state, unsigned char* out, size_t length_in_bytes) {
// 	aes_stream(state, out, length_in_bytes, true);
// }

// void get_r(aes_stream_state* state, unsigned char* out, size_t length_in_bytes, int shift) {
// 	get_PRG(state, out, length_in_bytes);
// }

// #endif



#ifndef RAND_POOL_H
#define RAND_POOL_H

#include "assert.h"
#include <immintrin.h>
#include "aes_stream_common.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

typedef unsigned char SeedT[AES_STREAM_SEEDBYTES];

#define STATE_LEN ((AES_STREAM_ROUNDS) + 1) * 16 + 16
extern unsigned char init_key[STATE_LEN];	// TODO generate at random
extern unsigned char init_seed[AES_STREAM_SEEDBYTES]; //TODO generate at random


void init_PRG(aes_stream_state* state);
void InitPrgWithSeed(aes_stream_state* state, const SeedT seed);

void get_PRG(aes_stream_state* state, unsigned char* out, size_t length_in_bytes);

void get_r(aes_stream_state* state, unsigned char* out, size_t length_in_bytes, int shift);

#endif
