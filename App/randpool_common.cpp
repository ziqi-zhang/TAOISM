#include "randpool_common.hpp"

unsigned char init_key[STATE_LEN] = {0x00};	// TODO generate at random
unsigned char init_seed[AES_STREAM_SEEDBYTES] = {0x00}; //TODO generate at random


void init_PRG(aes_stream_state* state) {
	std::copy(init_key, init_key + STATE_LEN , state->opaque);
	aes_stream_init(state, init_seed);
}

void InitPrgWithSeed(aes_stream_state* state, const SeedT seed) {
	std::copy(init_key, init_key + STATE_LEN , state->opaque);

	aes_stream_init(state, seed);
	// aes_stream_init(state, init_seed);
}

void get_PRG(aes_stream_state* state, unsigned char* out, size_t length_in_bytes) {
	aes_stream(state, out, length_in_bytes, true);
}

void get_r(aes_stream_state* state, unsigned char* out, size_t length_in_bytes, int shift) {
	get_PRG(state, out, length_in_bytes);
}
