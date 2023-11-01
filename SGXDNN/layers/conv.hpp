#ifndef CONV_H
#define CONV_H

#ifdef USE_SGX
#include "Enclave.h"
#endif


#include <cstdint>
#include <memory>
#include <unordered_map>

#include "common_with_enclaves.h"
#include "secret_tensor.hpp"
#include "utils.hpp"

using namespace std;
using std::shared_ptr;


template <typename Func>
void run_all_chunks_conv(Func chunk_op, int num_elem_in_chunk, int num_elem);


class SGXConvBuffer {
public:
    SGXConvBuffer(){}
    SGXConvBuffer(IdT FunId_);

    ~SGXConvBuffer() = default;

    void init(
            IdT input, IdT output, IdT weight, IdT bias,
            // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
            uint32_t batch_, uint32_t input_h_, uint32_t input_w_, uint32_t input_c_,
            uint32_t output_h_, uint32_t output_w_, uint32_t output_c_,
            uint32_t kernel_, uint32_t padding_, uint32_t stride_);

    // int get_num_batches_per_chunk(int num_elem_in_chunk) {
    //     return num_elem_in_chunk / input_size;
    // }

    void forward();

    void backward() {}
    
    IdT FunId;
    int batch, input_h, input_w, input_c;
    int output_h, output_w, output_c, kernel, padding, stride;

    int input_row_size, one_batch_input_size, input_elem_fetch_per_chunk, 
        patch_size, max_im2col_patches_per_chunk, max_output_patches_per_chunk, 
        max_weight_rows_per_chunk, max_weight_elem_per_chunk,
        max_matrix_mul_rows, im2col_num_elem_in_chunk, output_num_elem_in_chunk,
        im2col_patches_per_chunk, output_patches_per_chunk;

    int num_rows;
    int num_rows_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_col_per_chunk;
    int features_per_chunk, classes_per_chunk;

    int NumBatchesTrackedArr = 0;

    #ifdef PRINT_RUN_TIME_INFO
        sgx_time_t load_input_time=0, load_weight_time=0, im2col_construction_time=0, 
        save_output_time=0, matrix_mul_time=0, forward_prepare_time=0, total_time=0, test_time=0;
    #endif

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    // shared_ptr<SecretTen> der_input_tensor;
    // shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> weight_tensor;
    shared_ptr<SecretTen> bias_tensor;
    // shared_ptr<SecretTen> der_weight_tensor;
    // shared_ptr<SecretTen> der_bias_tensor;
};



#endif