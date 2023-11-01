#ifndef BATCHNORM_H
#define BATCHNORM_H

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

class BatchnormBuffer {
public:
    BatchnormBuffer(){}
    BatchnormBuffer(IdT FunId_);

    ~BatchnormBuffer() = default;

    void init(
            IdT input, IdT output, IdT gamma, IdT beta,
            IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
            IdT mu,
            uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
            int affine_, int is_cumulative_, float momentum_, float epsilon_);

    DtypeForCpuOp get_fraction_bag(int num_elem_in_chunk);

    int get_num_batches_per_chunk(int num_elem_in_chunk);

    void forward(int training);

    void backward();
    
    IdT FunId;
    int batch;
    int channel;
    int height;
    int width;
    DtypeForCpuOp momentum;
    DtypeForCpuOp epsilon;

    bool is_cumulative;
    bool BackwardState;
    bool Affine;
    bool Training;

    int num_rows;
    int num_elem_per_sample, num_elem_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_rows_per_chunk;

    int NumBatchesTrackedArr = 0;

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> der_input_tensor;
    shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> mu_tensor;
    shared_ptr<SecretTen> gamma_tensor;
    shared_ptr<SecretTen> beta_tensor;
    shared_ptr<SecretTen> der_gamma_tensor;
    shared_ptr<SecretTen> der_beta_tensor;
    shared_ptr<SecretTen> run_mean_tensor;
    shared_ptr<SecretTen> run_var_tensor;
    shared_ptr<SecretTen> cur_mean_tensor;
    shared_ptr<SecretTen> cur_var_tensor;
};


#endif