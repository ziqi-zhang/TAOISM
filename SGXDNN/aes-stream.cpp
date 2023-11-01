#include "aes-stream.hpp"
#include "sgxdnn_main.hpp"

#include "../App/aes_stream_common.cpp"

// #include <string>

#ifdef USE_SGX
#include "Enclave.h"
#endif

// #include "aes_stream_common.cpp"


/* reblinds the input and writes it to output outside of the enclave. For efficiency reasons, we perform a single
 * loop over the data in which we:
 * 	- compute activations
 *  - compute the AES PRG stream and blind the activations
 *  - perform the Freivalds checks for integrity *  - write the blinded data outside of the enclave
 */
//void aes_stream_fused(aes_stream_state *st, float* out, float* input, float* blind, size_t image_size, size_t ch,
//					  char* activation, integrityParams& params) {
//	std::string act(activation);
//	if (!params.integrity) {
//		if (act == "relu") {
//			AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//							 relu_avx, empty_verif_x, empty_verif_z, empty_verif_z_outer, params);
//		} else {
//			AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//							 relu6_avx, empty_verif_x, empty_verif_z, empty_verif_z_outer, params);
//		}
//		return;
//	}
//
//	assert(act == "relu");
//	if (params.pointwise_x && params.pointwise_z) {
//		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//						 relu_avx, preproc_verif_pointwise_X_inner, preproc_verif_pointwise_Z_inner, empty_verif_z_outer, params);
//	} else if (params.pointwise_x) {
//		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//						 relu_avx, preproc_verif_pointwise_X_inner, preproc_verif_Z_inner, preproc_verif_Z_outer, params);
//	} else if (params.pointwise_z) {
//		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//						 relu_avx, preproc_verif_X_inner, preproc_verif_pointwise_Z_inner, empty_verif_z_outer, params);
//	} else {
//		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
//						 relu_avx, preproc_verif_X_inner, preproc_verif_Z_inner, preproc_verif_Z_outer, params);
//	}
//}
