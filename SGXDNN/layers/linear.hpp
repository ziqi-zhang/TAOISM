#ifndef LINEAR_H
#define LINEAR_H

#ifdef USE_SGX
#include "Enclave.h"
#endif

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "common_with_enclaves.h"
#include "secret_tensor.hpp"

using namespace std;
using std::shared_ptr;



class SGXLinearBuffer {
public:
    SGXLinearBuffer(){}
    SGXLinearBuffer(IdT FunId_);

    ~SGXLinearBuffer() = default;

    void init(
            IdT input, IdT output, IdT weight, IdT bias,
            uint32_t batch_, uint32_t input_size_, uint32_t output_size_
    );

    int get_num_batches_per_chunk(int num_elem_in_chunk);

    void forward();

    void backward() {}
    
    IdT FunId;
    int batch;
    int input_size;
    int output_size;

    int num_rows;
    int num_rows_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_col_per_chunk;
    int features_per_chunk, classes_per_chunk;
    int NumBatchesTrackedArr = 0;

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> der_input_tensor;
    shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> weight_tensor;
    shared_ptr<SecretTen> bias_tensor;
    shared_ptr<SecretTen> der_weight_tensor;
    shared_ptr<SecretTen> der_bias_tensor;
};

#endif