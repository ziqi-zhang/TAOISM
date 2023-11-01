#define USE_EIGEN_TENSOR

#include "sgxdnn_main.hpp"

#include "Enclave.h"
#include "Enclave_t.h"

#include "Crypto.h"

void ecall_init_tensor(uint64_t TenId, void* voidDims) {
    SecretInitTensor(TenId, voidDims);
}

void ecall_set_ten(uint64_t TenId, void* voidArr) {
    SecretSetTen(TenId, voidArr);
}
void ecall_get_ten(uint64_t TenId, void* voidArr) {
    SecretGetTen(TenId, voidArr);
}

void ecall_set_seed(uint64_t TenId, uint64_t RawSeed) {
    SecretSetSeed(TenId, RawSeed);
}

void ecall_get_random(uint64_t TenId, void* voidArr, uint64_t RawSeed) {
    SecretGetRandom(TenId, voidArr, RawSeed);
}


void ecall_relu(uint64_t TenIdin, uint64_t TenIdout, uint64_t size){
    newrelu(TenIdin, TenIdout, size);
}

void ecall_quant_relu(uint64_t TenIdin, uint64_t TenIdout, uint64_t size, float scale, float v_min, uint8_t zero){
    quantrelu(TenIdin, TenIdout, size, scale, v_min, zero);
}

void ecall_reluback(uint64_t TenIdout, uint64_t TenIddout, uint64_t TenIddin, uint64_t size){
    newreluback(TenIdout, TenIddout, TenIddin, size);
}

void ecall_initmaxpool(uint64_t FunId, uint64_t TenIdin_trans, uint64_t TenIdout_trans){
    initmaxpool(FunId, TenIdin_trans, TenIdout_trans);
}

void ecall_maxpool(uint64_t FunId, uint64_t TenIdin, uint64_t TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride,uint32_t col_stride, uint32_t row_pad, uint32_t col_pad) {
    newmaxpool(FunId, TenIdin, TenIdout, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride, row_pad, col_pad);
}

void ecall_maxpoolback(uint64_t FunId, uint64_t TenIddout, uint64_t TenIddin, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride){
    newmaxpoolback(FunId, TenIddout, TenIddin, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride);
}

void ecall_add_from_cpu(void* inputArr, uint64_t dstId) {
    SecretAddFromCpu(inputArr, dstId);
}

void ecall_sgd_update(uint64_t paramId, uint64_t gradId, uint64_t momentumId,
                     float lr, float momentum, float weight_decay,
                     float dampening, int nesterov, int first_momentum) {
    SecretSgdUpdate(paramId, gradId, momentumId, lr, momentum, weight_decay, dampening, nesterov, first_momentum);
}

void ecall_stochastic_quantize(uint64_t src_id, uint64_t dst_id, uint64_t q_tag) {
    SecretStochasticQuantize(src_id, dst_id, q_tag);
}


void ecall_init_batchnorm(
        uint64_t FunId,
        uint64_t input, uint64_t output, uint64_t gamma, uint64_t beta,
        // uint64_t der_input, uint64_t der_output, uint64_t der_gamma, uint64_t der_beta,
        uint64_t run_mean, uint64_t run_var, uint64_t cur_mean, uint64_t cur_var,
        uint64_t mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    SecretInitBatchnorm(
            FunId,
            input, output, gamma, beta,
            // der_input, der_output, der_gamma, der_beta,
            run_mean, run_var, cur_mean, cur_var,
            mu,
            batch_, channel_, height_, width_,
            affine_, is_cumulative_, momentum_, epsilon_);
}

void ecall_batchnorm_forward(uint64_t FunId, int Training) {
    SecretBatchnormForward(FunId, Training);
}

void ecall_batchnorm_backward(uint64_t FunId) {
    SecretBatchnormBackward(FunId);
}

void ecall_init_sgx_linear(
        uint64_t FunId,
        uint64_t input, uint64_t output, uint64_t weight, uint64_t bias,
        // uint64_t der_input, uint64_t der_output, uint64_t der_weight, uint64_t der_bias,
        uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

    SecretInitSGXLinear(
            FunId,
            input, output, weight, bias,
            // der_input, der_output, der_weight, der_bias,
            batch_, input_size_, output_size_);
}

void ecall_sgx_linear_forward(uint64_t FunId) {
    SecretSGXLinearForward(FunId);
}

void ecall_init_sgx_conv(
        uint64_t FunId,
        uint64_t input, uint64_t output, uint64_t weight, uint64_t bias, 
        // uint64_t der_input, uint64_t der_output, uint64_t der_weight, uint64_t der_bias, 
        uint32_t batch_, uint32_t input_h, uint32_t input_w, uint32_t input_c, 
        uint32_t output_h, uint32_t output_w, uint32_t output_c,
        uint32_t kernel, uint32_t padding, uint32_t stride) {

    SecretInitSGXConv(
            FunId,
            input, output, weight, bias,
            // der_input, der_output, der_weight, der_bias,
            batch_, input_h, input_w, input_c, 
            output_h, output_w, output_c,
            kernel, padding, stride);
}

void ecall_sgx_conv_forward(uint64_t FunId) {
    SecretSGXConvForward(FunId);
}