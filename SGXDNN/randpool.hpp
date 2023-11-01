#pragma once

#include "aes-stream.hpp"
#include "randpool_common.hpp"

void fused_blind(aes_stream_state* state, float* out, float* blinded_input, float* blind, size_t image_size, size_t ch,
				 char* activation, integrityParams& params) {
	aes_stream_fused(state, out, blinded_input, blind, image_size, ch, activation, params);
} 
