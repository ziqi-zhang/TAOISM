#ifndef SGXDNN_MAIN_H
#define SGXDNN_MAIN_H

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <memory>

#include "sgxdnn_common.hpp"
#include "common_with_enclaves.h"
#include "secret_tensor.hpp"

extern int p_int;
extern float p;
extern float mid;

extern int p_verif;
extern double inv_p_verif;


extern "C" {
        void SecretInitTensor(uint64_t TenId, void* voidDims);
        void SecretSetTen(uint64_t TenId, void* voidArr); 
        void SecretGetTen(uint64_t TenId, void* voidArr); 
        void SecretSetSeed(uint64_t TenId, uint64_t RawSeed);
        void SecretGetRandom(uint64_t TenId, void* voidArr, uint64_t RawSeed); 
        void SecretAddFromCpu(void* inputArr, uint64_t dstId);
        void SecretSgdUpdate(uint64_t paramId, uint64_t gradId, uint64_t momentumId,
                             float lr, float momentum, float weight_decay,
                             float dampening, bool nesterov, bool first_momentum);

        void SecretStochasticQuantize(uint64_t src_id, uint64_t dst_id, uint64_t q_tag);

        void newrelu(uint64_t TenIdin, uint64_t TenIdout, uint64_t size);
        void quantrelu(uint64_t TenIdin, uint64_t TenIdout, uint64_t size, float scale, float v_min, uint8_t zero);
        void newreluback(uint64_t TenIdout, uint64_t TenIddout, uint64_t TenIddin, uint64_t size);
        //void initmaxpool(uint64_t maxsizein, uint64_t maxsizeout);
        void initmaxpool(uint64_t FunId, uint64_t TenIdin_trans, uint64_t TenIdout_trans);
        void newmaxpool(uint64_t FunId, uint64_t TenIdin, uint64_t TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height,uint32_t filter_width,uint32_t row_stride,uint32_t col_stride, uint32_t row_pad, uint32_t col_pad);
        void newmaxpoolback(uint64_t FunId, uint64_t TenIddout, uint64_t TenIddin, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride);

        void SecretInitBatchnorm(
                uint64_t FunId,
                uint64_t input, uint64_t output, uint64_t gamma, uint64_t beta,
                // uint64_t der_input, uint64_t der_output, uint64_t der_gamma, uint64_t der_beta,
                uint64_t run_mean, uint64_t run_var, uint64_t cur_mean, uint64_t cur_var,
                uint64_t mu,
                uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t wuint64_th_,
                int affine_, int is_cumulative_, float momentum_, float epsilon_);
        void SecretBatchnormForward(uint64_t FunId, int Training);
        void SecretBatchnormBackward(uint64_t FunId);

        void SecretInitSGXLinear(
                uint64_t FunId,
                uint64_t input, uint64_t output, uint64_t weight, uint64_t bias,
                // uint64_t der_input, uint64_t der_output, uint64_t der_weight, uint64_t der_bias,
                uint32_t batch_, uint32_t input_size_, uint32_t output_size_);
        void SecretSGXLinearForward(uint64_t FunId);

        void SecretInitSGXConv(
                uint64_t FunId,
                uint64_t input, uint64_t output, uint64_t weight, uint64_t bias,
                // uint64_t der_input, uint64_t der_output, uint64_t der_weight, uint64_t der_bias,
                uint32_t batch_, uint32_t input_h, uint32_t input_w, uint32_t input_c, 
                uint32_t output_h, uint32_t output_w, uint32_t output_c,
                uint32_t kernel, uint32_t padding, uint32_t stride);
        void SecretSGXConvForward(uint64_t FunId);
}

#define REPS 2

// slow modulo operations
inline double mod(double x, int N){
	return fmod(x, static_cast<double>(N));
}

inline double mod_pos(double x, int N){
	return mod(mod(x, N) + N, N);
}

// Macros for fast modulos
#define REDUCE_MOD(lv_x) \
	{lv_x -= floor(lv_x * inv_p_verif) * p_verif;}

#define REDUCE_MOD_TENSOR(lv_tensor) \
	{lv_tensor = lv_tensor - (lv_tensor * inv_p_verif).floor() * static_cast<double>(p_verif);}

// vectorized activations
__m256 inline relu_avx(__m256 z) {
	return _mm256_round_ps(_mm256_mul_ps(_mm256_max_ps(z, zero8f), inv_shift8f), _MM_FROUND_CUR_DIRECTION);
}

__m256 inline relu6_avx(__m256 z) {
	return _mm256_round_ps(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(z, zero8f), six8f), inv_shift8f), _MM_FROUND_CUR_DIRECTION);
}

__m256 inline id_avx(__m256 z) {
	return z;
}

// parameters for the fused AES + integrity check
typedef struct integrityParams {
	bool integrity;
	bool pointwise_x;
	bool pointwise_z;
	double* res_x;
	double* res_z;
	float* kernel_r_data;
	double* r_left_data;
	double* r_right_data;
	__m256d temp_x[REPS];
	__m256d temp_z[REPS];
} integrityParams;

#endif
