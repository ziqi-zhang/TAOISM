#ifndef MAXPOOL_H
#define MAXPOOL_H

#ifdef USE_SGX
#include "Enclave.h"
#endif


#include <cstdint>
#include <memory>
#include <unordered_map>

#include "common_with_enclaves.h"
#include "secret_tensor.hpp"
#include "sgxdnn_common.hpp"

using namespace std;
using std::shared_ptr;


class MaxpoolBuffer {
public:
    MaxpoolBuffer() {}
    MaxpoolBuffer(IdT FunId_, IdT TenIdin_trans_, IdT TenIdout_trans_);

    ~MaxpoolBuffer() = default;

	IdT get_TenIdin_trans(){
		return TenIdin_trans;
	}

	IdT get_TenIdout_trans(){
		return TenIdout_trans;
	}
    //if NCHW->WHCN N=CN M=HW
    void transpose(const DtypeForCpuOp *src, DtypeForCpuOp *dst, const size_t N, const size_t M);

inline void transpose4x4_SSE(const float *A, float *B, const uint32_t lda, const uint32_t ldb) {
        __m128 row1 = _mm_load_ps(&A[0*lda]);
        __m128 row2 = _mm_load_ps(&A[1*lda]);
        __m128 row3 = _mm_load_ps(&A[2*lda]);
        __m128 row4 = _mm_load_ps(&A[3*lda]);
         _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
         _mm_store_ps(&B[0*ldb], row1);
         _mm_store_ps(&B[1*ldb], row2);
         _mm_store_ps(&B[2*ldb], row3);
         _mm_store_ps(&B[3*ldb], row4);
    }

    inline void transpose_block_SSE4x4(const float *A, float *B, const uint32_t lda, const uint32_t ldb ,const int block_size) {
        #pragma omp parallel for
        for(uint32_t i=0; i<ldb; i+=block_size) {
            for(uint32_t j=0; j<lda; j+=block_size) {
                uint32_t max_i2 = i+block_size < ldb ? i + block_size : ldb;
                uint32_t max_j2 = j+block_size < lda ? j + block_size : lda;
                for(uint32_t i2=i; i2<max_i2; i2+=4) {
                    for(uint32_t j2=j; j2<max_j2; j2+=4) {
                        transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                    }
                }
            }
         }
    }
    
    inline void MaxpoolAVX(const uint32_t num_img, float* input, float* output){
        // uint32_t base = num_img/8;
        // base *= 8;
        // printf("Base %d\n", base);

        // AVX requires to align address to 32 bytes, both input and output should align
        // printf("mod 8 %d\n", input)
        // #pragma omp parallel for        
        // for(size_t i=0; i<base; i+=8){
        //     printf("%d, addr %p\n", i, input);
        //     const __m256 inp8f = _mm256_load_ps(&input[i]);
        //     const __m256 out8f = _mm256_load_ps(&output[i]);
        //     const __m256 if_lq = _mm256_cmp_ps(out8f, inp8f, 0x01);
        //     const __m256 res8f = _mm256_blendv_ps(out8f, inp8f, if_lq);
        //     _mm256_stream_ps(&output[i], res8f);
        // }
        // printf("Middle\n");

        for (size_t i=0; i<num_img; i++){
            if (input[i] > output[i])
                output[i] = input[i];
        }
        // printf("Finish\n");
    }

    inline void PlainMaxpool(const uint32_t num_img, float* input, float* output){
        #pragma omp parallel for        
        for (size_t i=0; i<num_img; i++){
            if (input[i] > output[i])
                output[i] = input[i];
        }
    }

    inline void MaxpoolbackAVX(const uint32_t num_img, float* input, float* output, float* dinput, float* doutput){
        #pragma omp parallel for
        for(size_t i=0; i<num_img; i+=8){
            const __m256 inp8f = _mm256_load_ps(&input[i]);
            const __m256 out8f = _mm256_load_ps(&output[i]);
            const __m256 din8f = _mm256_load_ps(&dinput[i]);
            const __m256 dout8f = _mm256_load_ps(&doutput[i]);
            const __m256 if_eq = _mm256_cmp_ps(out8f, inp8f, 0x00);
            const __m256 sum8f = _mm256_add_ps(din8f, dout8f);
            const __m256 res8f = _mm256_blendv_ps(din8f, sum8f, if_eq); // define dinput
            const __m256 res28f = _mm256_blendv_ps(dout8f, zero8f, if_eq); // redefine doutput
            _mm256_store_ps(&dinput[i], res8f);
            _mm256_stream_ps(&doutput[i], res28f);
        }
    }

    void forward(
           shared_ptr<SecretTen> ten_in, shared_ptr<SecretTen> ten_out,
           shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
           uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
           uint32_t output_height, uint32_t output_width, uint32_t filter_height,
           uint32_t filter_width, uint32_t row_stride, uint32_t col_stride);

    void backward(
            shared_ptr<SecretTen> ten_din, shared_ptr<SecretTen> ten_dout,
            shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
            uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
            uint32_t output_height, uint32_t output_width,
            uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride);

    IdT FunId;
	IdT TenIdin_trans;
   	IdT TenIdout_trans;
};



#endif