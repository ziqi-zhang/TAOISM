#ifndef STOCHASTIC_H
#define STOCHASTIC_H


#include "secret_tensor.hpp"
#include "xoshiro.hpp"

void quantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten, uint64_t quantize_tag);

void dequantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten,
        uint64_t x_tag, uint64_t y_tag);


extern unordered_map<uint64_t, DtypeForCpuOp> quantize_exp;


#endif