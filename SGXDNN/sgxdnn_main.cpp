#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#include <malloc.h>
#else
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#endif

#include "sgxdnn_main.hpp"
#include "randpool.hpp"
#include "utils.hpp"
#include "chunk_manager.hpp"
#include "secret_tensor.hpp"
#include "stochastic.hpp"
#include "xoshiro.hpp"

#include "layers/batchnorm.hpp"
#include "layers/linear.hpp"
#include "layers/conv.hpp"
#include "layers/maxpool.hpp"

#include "common_with_enclaves.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <cstring>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <cstdlib>
#include <mutex>
#include <stack>
#include <time.h>
#include "Crypto.h"
#include <omp.h>
#include "../App/common_utils.cpp"


using namespace std;

using std::shared_ptr;
using std::make_shared;
using std::unordered_map;
using std::string;
using defer = shared_ptr<void>;


//using namespace SGXDNN;

int p_int = PrimeLimit;
float p = (float) p_int;
float mid = (float) (p_int / 2);

// some vectorized constants
__m256 p8f = _mm256_set1_ps(p);
__m256 p28f = _mm256_set1_ps(p * 2);
__m256 mid8f = _mm256_set1_ps(mid);
__m256 pmid8f = _mm256_set1_ps(p + mid);
__m256 negmid8f = _mm256_set1_ps(-mid - 1);
__m256 zero8f = _mm256_set1_ps((float)(0));
__m256 inv_shift8f = _mm256_set1_ps((float)(1.0/256));
__m256 six8f = _mm256_set1_ps((float) 6 * 256 * 256);

inline void MoveDown(float* input, float* out, int num_elements) {
	for(size_t i = 0; i < num_elements; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &input[i] );             // blinded input

			const __m256 if_geq  = _mm256_cmp_ps(inp8f, mid8f, 0x0d);    // unblinded >= mid
			// const __m256 if_lt   = _mm256_cmp_ps(inp8f, negmid8f, 0x01);  // unblinded < -mid
			const __m256 then8f  = _mm256_sub_ps(inp8f, p8f);            // unblinded - p
			// const __m256 elif8f  = _mm256_add_ps(inp8f, p8f);            // unblinded + p
			const __m256 res8f = _mm256_blendv_ps(
                                        inp8f,
										then8f,
										if_geq);

			_mm256_stream_ps(&out[i], res8f);
    }
}


void ModP(MapMatRowMajor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<DtypeForCpuOp>(1) / PrimeLimit;
    m.array() = m.array() - (m * invPLimit).array() * PLimit;
}

void ModP(EigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

void ModP(MapEigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

// #define PRINT_CHUNK_INFO 
// #define PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
// #define PRINT_CONV_INPUT_LOAD_CHUNK_INFO
// #define PRINT_CONV_IM2COL_CONSTRUCT_INFO
// #define PRINT_CONV_INIT_INFO
// #define PRINT_RUN_TIME_INFO



extern "C" {

void SecretInitTensor(IdT TenId, void *voidDims) {
    DimsT dims = *(DimsT*)voidDims;
    #ifdef PRINT_CHUNK_INFO
        printf("SecretInitTensor id %ld, size (%d,%d,%d,%d), ", TenId, dims.dim0, dims.dim1, dims.dim2, dims.dim3);
    #endif
    DimsT *Dims = (DimsT *) voidDims;
    SecretTenHolder[TenId] = make_shared<SecretTen>(TenId, Dims);
}

void SecretSetTen(IdT TenId, void *voidArr) {
    DtypeForCpuOp* cpu_p = (DtypeForCpuOp*) voidArr;
    // printf("TenId %ld, %f %f %f\n", TenId, cpu_p[0], cpu_p[1], cpu_p[2]);
    GetTenById(TenId)->SetTen((DtypeForCpuOp *) voidArr);
}

void SecretGetTen(IdT TenId, void *voidArr) {
    GetTenById(TenId)->GetTen((DtypeForCpuOp *) voidArr);
}

void SecretSetSeed(IdT TenId, uint64_t RawSeed) {
    GetTenById(TenId)->SetSeed(RawSeed);
}

void SecretGetRandom(IdT TenId, void *voidArr, uint64_t RawSeed) {
    GetTenById(TenId)->GetRandom((DtypeForCpuOp *) voidArr, RawSeed);
}

void SecretAddFromCpu(void* inputArr, IdT dstId) {
    shared_ptr<SecretTen > StoreTensor = GetTenById(dstId);
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    const int total_num_elem = StoreTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start_chunk, int num_elem_in_op) {
            DtypeForCpuOp* output_arr = store_chunk + start_chunk;
            DtypeForCpuOp* input_arr = ((DtypeForCpuOp*) inputArr) + start_store_chunk + start_chunk;
            for(size_t j = 0; j < num_elem_in_op; j++) {
                output_arr[j] += input_arr[j];
                output_arr[j] -= floor(output_arr[j] * invPLimit) * PLimit;
                output_arr[j] = (output_arr[j] >= mid) ? (output_arr[j] - p) : output_arr[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);

        chunk_manager.StoreChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

void newrelu(IdT TenIdin, IdT TenIdout, uint64_t size){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);
	shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_in,* chunk_tmp;
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
    // printf("Newrelu\n");
    //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
            const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
            const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
            _mm256_stream_ps(&chunk_tmp[i], res8f);
        } 
		chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
}
// void newrelu(IdT TenIdin, IdT TenIdout, uint64_t size){
//     shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);
// 	shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
// 	auto& chunk_manager = TrustedChunkManager::getInstance();
//     DtypeForCpuOp* chunk_in,* chunk_tmp;
//     ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
//     printf("Newrelu\n");
//     //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
//     auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
//         chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
// 		for(uint64_t i=0;i<num_elem_in_store_chunk;i++){
//             chunk_tmp[i] = chunk_tmp[i]>0 ? chunk_tmp[i] : 0;
//         }
//         // for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
//         //     const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
//         //     const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
//         //     const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
//         //     _mm256_stream_ps(&chunk_tmp[i], res8f);
//         // } 
// 		chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
//     };
//     run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
// }

void quantrelu(IdT TenIdin, IdT TenIdout, uint64_t size, float scale, float v_min, uint8_t zero){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);
	shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_in,* chunk_tmp, *float_chunk_tmp;
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
    ChunkGuard<DtypeForCpuOp> guard_float_tmp(StoreChunkPool::GetChunkPool(), float_chunk_tmp);
    uint8_t* chunk_tmp_uint8 = (uint8_t*) chunk_tmp;
    float zero_f = zero;
    // printf("In quant relu, scale is %f, v_min is %f, zero is %f\n", scale, v_min, zero_f);
    //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        
        for(uint64_t i=0;i<num_elem_in_store_chunk*4;i++){
            float a = chunk_tmp_uint8[i];
            // printf("original value %f, ", a);
            chunk_tmp_uint8[i] = chunk_tmp_uint8[i]>128 ? chunk_tmp_uint8[i] : 128;
            a = chunk_tmp_uint8[i];
            // printf("after value %f\n", a);
            
            // a = a / scale;
            // // printf("rescale %f, ", a);
            // chunk_tmp_uint8[i] = a>0 ? chunk_tmp_uint8[i] : zero;
        }
		// for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
        //     const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
        //     const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
        //     const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
        //     _mm256_stream_ps(&chunk_tmp[i], res8f);
        // } 
		chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
}

void newreluback(IdT TenIdout, IdT TenIddout,IdT TenIddin, uint64_t size){
    shared_ptr<SecretTen > ten_din = GetTenById(TenIddin);
    shared_ptr<SecretTen > ten_dout = GetTenById(TenIddout);
    shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_dtmp,* chunk_out;
    //ChunkGuard<DtypeForCpuOp> guard_din(StoreChunkPool::GetChunkPool(), chunk_din);
    ChunkGuard<DtypeForCpuOp> guard_dtmp(StoreChunkPool::GetChunkPool(), chunk_dtmp);
    ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_dout->GetChunkId(start_store_chunk),chunk_dtmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp)); 
        chunk_manager.GetChunk(ten_out->GetChunkId(start_store_chunk),chunk_out, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&chunk_out[i]);
            const __m256 if_eq = _mm256_cmp_ps(inp8f, zero8f, 0x00);
            const __m256 gra8f = _mm256_load_ps(&chunk_dtmp[i]);
            const __m256 res8f = _mm256_blendv_ps(gra8f, zero8f, if_eq);
            _mm256_stream_ps(&chunk_dtmp[i], res8f);
        }
        chunk_manager.StoreChunk(ten_din->GetChunkId(start_store_chunk), chunk_dtmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);	
}

unordered_map<IdT, shared_ptr<MaxpoolBuffer>> MaxpoolHolder;


shared_ptr<MaxpoolBuffer> GetBufferByIdM(IdT FunId) {
    return MaxpoolHolder[FunId];
}

void initmaxpool(IdT FunId, IdT TenIdin_trans, IdT TenIdout_trans){	
    // printf("Initmaxpool TenIdin_trans %ld, TenIdout_trans %ld\n", TenIdin_trans, TenIdout_trans);
    MaxpoolHolder[FunId] = make_shared<MaxpoolBuffer>(FunId, TenIdin_trans, TenIdout_trans);
}

void newmaxpool(IdT FunId, IdT TenIdin, IdT TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride, uint32_t row_pad, uint32_t col_pad){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);                      
    shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	IdT TenIdin_trans = GetBufferByIdM(FunId)->get_TenIdin_trans();
	shared_ptr<SecretTen> ten_in_trans = GetTenById(TenIdin_trans);
	IdT TenIdout_trans = GetBufferByIdM(FunId)->get_TenIdout_trans();
    shared_ptr<SecretTen> ten_out_trans = GetTenById(TenIdout_trans);  
    // printf("newmaxpool TenIdin_trans %ld, TenIdout_trans %ld\n", TenIdin_trans, TenIdout_trans);
	GetBufferByIdM(FunId)->forward(ten_in, ten_out,ten_in_trans, ten_out_trans, batch, channel,input_height,input_width,output_height,output_width,filter_height,filter_width,row_stride,col_stride);
}

void newmaxpoolback(IdT FunId, IdT TenIddout,IdT TenIddin, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride){
    shared_ptr<SecretTen > ten_din = GetTenById(TenIddin);                                                                                                
    shared_ptr<SecretTen > ten_dout = GetTenById(TenIddout);
    IdT TenIdin_trans = GetBufferByIdM(FunId)->get_TenIdin_trans();
    shared_ptr<SecretTen> ten_in_trans = GetTenById(TenIdin_trans);
    IdT TenIdout_trans = GetBufferByIdM(FunId)->get_TenIdout_trans();
    shared_ptr<SecretTen> ten_out_trans = GetTenById(TenIdout_trans);                                                                                               
    //shared_ptr<SecretTen > ten_in_trans = GetTenById(0);
    //uint64_t tensor_size=(batch*channel*input_height*input_width+STORE_CHUNK_ELEM/2)/STORE_CHUNK_ELEM*STORE_CHUNK_ELEM;
    //shared_ptr<SecretTen > ten_out_trans = GetTenById(tensor_size*sizeof(float));
    GetBufferByIdM(FunId)->backward(ten_din, ten_dout, ten_in_trans, ten_out_trans, batch, channel,input_height,input_width,output_height,output_width,filter_height,filter_width,row_stride,col_stride);
}

unordered_map<IdT, shared_ptr<BatchnormBuffer>> BatchnormHolder;
shared_ptr<BatchnormBuffer> GetBufferByIdB(IdT FunId) {
    return BatchnormHolder[FunId];
}
    
void SecretInitBatchnorm(
        IdT FunId,
        IdT input, IdT output, IdT gamma, IdT beta,
        // IdT der_input, IdT der_output, IdT der_gamma, IdT der_beta,
        IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
        IdT mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    auto bn_buffer = make_shared<BatchnormBuffer>(FunId);
    BatchnormHolder[FunId] = bn_buffer;

    bn_buffer->init(
            input, output, gamma, beta,
            // der_input, der_output, der_gamma, der_beta,
            run_mean, run_var, cur_mean, cur_var,
            mu,
            batch_, channel_, height_, width_,
            affine_, is_cumulative_, momentum_, epsilon_);
}

void SecretBatchnormForward(IdT FunId, int Training) {
    GetBufferByIdB(FunId)->forward(Training);
}

void SecretBatchnormBackward(IdT FunId) {
    GetBufferByIdB(FunId)->backward();
}

unordered_map<IdT, shared_ptr<SGXLinearBuffer>> SGXLinearHolder;
shared_ptr<SGXLinearBuffer> GetSGXLinearBufferByIdB(IdT FunId) {
    return SGXLinearHolder[FunId];
}
void SecretInitSGXLinear(
        IdT FunId,
        IdT input, IdT output, IdT weight, IdT bias,
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

    auto sgx_linear_buffer = make_shared<SGXLinearBuffer>(FunId);
    SGXLinearHolder[FunId] = sgx_linear_buffer;

    sgx_linear_buffer->init(
            input, output, weight, bias,
            // der_input, der_output, der_weight, der_bias,
            batch_, input_size_, output_size_);
}

void SecretSGXLinearForward(IdT FunId) {
    GetSGXLinearBufferByIdB(FunId)->forward();
}

unordered_map<IdT, shared_ptr<SGXConvBuffer>> SGXConvHolder;
shared_ptr<SGXConvBuffer> GetSGXConvBufferByIdB(IdT FunId) {
    return SGXConvHolder[FunId];
}
void SecretInitSGXConv(
        IdT FunId,
        IdT input, IdT output, IdT weight, IdT bias, 
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_h, uint32_t input_w, uint32_t input_c, 
        uint32_t output_h, uint32_t output_w, uint32_t output_c,
        uint32_t kernel, uint32_t padding, uint32_t stride) {

    auto sgx_conv_buffer = make_shared<SGXConvBuffer>(FunId);
    SGXConvHolder[FunId] = sgx_conv_buffer;
    sgx_conv_buffer->init(
            input, output, weight, bias, 
            // der_input, der_output, der_weight, der_bias,
            batch_, input_h, input_w, input_c, 
            output_h, output_w, output_c,
            kernel, padding, stride);
}

void SecretSGXConvForward(IdT FunId) {
    GetSGXConvBufferByIdB(FunId)->forward();
}


// Assume momentum > 0
void SecretSgdUpdate(IdT paramId, IdT gradId, IdT momentumId,
        DtypeForCpuOp lr, DtypeForCpuOp momentum, DtypeForCpuOp weight_decay,
        DtypeForCpuOp dampening, bool nesterov, bool first_momentum) {

    shared_ptr<SecretTen> ParamTensor = GetTenById(paramId);
    shared_ptr<SecretTen> GradTensor = GetTenById(gradId);
    shared_ptr<SecretTen> MomentumTensor = (momentumId != 0) ? GetTenById(momentumId) : nullptr;

    const int total_num_elem = ParamTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *param_store_chunk, *grad_store_chunk, *momentum_store_chunk;
    ChunkGuard<DtypeForCpuOp> param_guard(StoreChunkPool::GetChunkPool(), param_store_chunk);
    ChunkGuard<DtypeForCpuOp> grad_guard(StoreChunkPool::GetChunkPool(), grad_store_chunk);
    ChunkGuard<DtypeForCpuOp> momentum_guard(StoreChunkPool::GetChunkPool(), momentum_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ParamTensor->GetChunkId(start_store_chunk), param_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(GradTensor->GetChunkId(start_store_chunk), grad_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(MomentumTensor->GetChunkId(start_store_chunk), momentum_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            DtypeForCpuOp* param_arr = param_store_chunk + start;
            DtypeForCpuOp* grad_arr = grad_store_chunk + start;
            DtypeForCpuOp* momentum_arr = momentum_store_chunk + start;
            if (first_momentum) {
                for(size_t j = 0; j < num_elem_in_op; j++) {
                    grad_arr[j] += weight_decay * param_arr[j];
                    momentum_arr[j] = grad_arr[j];
                    param_arr[j] -= lr * momentum_arr[j];
                }
            } else {
                for(size_t j = 0; j < num_elem_in_op; j++) {
                    grad_arr[j] += weight_decay * param_arr[j];
                    momentum_arr[j] = momentum_arr[j] * momentum + (1 - dampening) * grad_arr[j];
                    param_arr[j] -= lr * momentum_arr[j];
                }
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
        chunk_manager.StoreChunk(ParamTensor->GetChunkId(start_store_chunk), param_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.StoreChunk(MomentumTensor->GetChunkId(start_store_chunk), momentum_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

void SecretStochasticQuantize(IdT src_id, IdT dst_id, uint64_t q_tag) {
    quantize_stochastic(GetTenById(src_id), GetTenById(dst_id), q_tag);
}

} // End of extern C
