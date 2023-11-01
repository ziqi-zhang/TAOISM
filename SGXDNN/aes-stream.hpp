#include "sgxdnn_main.hpp"
#include "aes_stream_common.hpp"

void aes_stream_fused(aes_stream_state *st, float *out, float* input, float* blind, size_t image_size, size_t ch,
					  char* activation, integrityParams& params);
