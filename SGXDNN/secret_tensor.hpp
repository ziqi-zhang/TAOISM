#ifndef SECRET_TENSOR_H
#define SECRET_TENSOR_H


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
#include <omp.h>


#include "common_with_enclaves.h"

using namespace std;

class SecretTen {
public:
    SecretTen() {}
    SecretTen(IdT TenId_, DimsT* Dims_);
    ~SecretTen();

    int GetNumElem() { return Dims.dim0 * Dims.dim1 * Dims.dim2 * Dims.dim3; }
    int GetSizeInByte() { return GetNumElem() * sizeof(DtypeForCpuOp); }

    void Init();

    int GetChunkId(int start);

    void GetStoreChunk(int start, DtypeForCpuOp* store_chunk, int num_byte);

    void SetTen(DtypeForCpuOp* Arr);

    void GetTen(DtypeForCpuOp* Arr);

    void SetSeed(uint64_t RawSeed);

    void GetRandom(DtypeForCpuOp* DstArr, uint64_t RawSeed);


    IdT TenId;
    DimsT Dims;
    vector<int> ChunkIds;
    unordered_map<uint64_t, aes_stream_state*> PrgStateHolder;
};

extern unordered_map<IdT, shared_ptr<SecretTen>> SecretTenHolder;
extern unordered_map<IdT, shared_ptr<EigenTensor>> TensorHolder;

shared_ptr<SecretTen> GetTenById(IdT TenId);

#endif