#ifndef CHUNK_MANAGER_H
#define CHUNK_MANAGER_H


#include "sgxdnn_main.hpp"
// #include "randpool.hpp"
// #include "utils.hpp"

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
// #include "../App/common_utils.cpp"

using namespace std;

using std::shared_ptr;

class ChunkPool {
public:
    ChunkPool(int size_pool_, int num_byte_chunk_);

    int get_chunk_id();

    void return_chunk_id(int id);

    std::vector<void*> chunks;

private:
    int size_pool;
    int num_byte_chunk;
    std::mutex stack_mutex;
    std::stack<int> chunk_ids;
};


class StoreChunkPool {
public:
    static shared_ptr<ChunkPool> GetChunkPool() {
        static StoreChunkPool instance;
        return instance.chunk_pool;
    }
    StoreChunkPool(StoreChunkPool const&) = delete;
    void operator=(StoreChunkPool const&) = delete;

private:
    StoreChunkPool() {
        chunk_pool = make_shared<ChunkPool>(THREAD_POOL_SIZE * 2, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
    }
    shared_ptr<ChunkPool> chunk_pool;
};

template<typename T>
class ChunkGuard {
public:
    ChunkGuard<T>(shared_ptr<ChunkPool> chunk_pool_, T*& pointer) :
        chunk_pool(chunk_pool_)
    {
        id = chunk_pool->get_chunk_id();
        pointer = (T*) chunk_pool->chunks[id];
    }
    ~ChunkGuard<T>() {
        chunk_pool->return_chunk_id(id);
    }
private:
    int id;
    shared_ptr<ChunkPool> chunk_pool;
};


class TrustedChunkManager {
public:
    static TrustedChunkManager& getInstance();
    TrustedChunkManager(TrustedChunkManager const&) = delete;
    void operator=(TrustedChunkManager const&) = delete;

    IdT GetNewId();

    const int start_idx = 1000;

    void StoreChunk(IdT id, void* src_chunk, int num_byte);
    void GetChunk(IdT id, void* dst_chunk, int num_byte);

private:
    TrustedChunkManager();

    void* get_untrusted_mem(IdT id, int num_byte);

    const int size_chunk_pool = THREAD_POOL_SIZE;
    int max_num_byte_plain_chunk;
    int max_num_byte_enc_chunk;
    std::atomic<int> id_counter;
    std::mutex address_mutex;
    std::shared_ptr<ChunkPool> blind_chunks;
    std::unordered_map<int, std::pair<void*, int>> untrusted_mem_holder;
};

template <typename Func>
void run_all_chunks(Func chunk_op, int num_elem_in_chunk, int num_elem) {
    int start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk);
    }
    if (start_chunk < num_elem) chunk_op(start_chunk, num_elem - start_chunk);
}

template <typename Func>
void run_all_chunks_for_maxpool(Func chunk_op, size_t num_elem_in_chunk, size_t num_elem, size_t num_elem_out, size_t inputhw, size_t outputhw) {
    size_t start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk, num_elem_out);
    }
    
    size_t remain_size = num_elem - start_chunk;
    if (start_chunk < num_elem) chunk_op(start_chunk, remain_size, (remain_size/inputhw)*outputhw);
}


DtypeForCpuOp* get_small_chunk(
        shared_ptr<SecretTen> tensor,
        vector<std::pair<shared_ptr<SecretTen>, 
        DtypeForCpuOp*>>& small_chunks
);

void store_small_chunks(
        vector<std::pair<shared_ptr<SecretTen>, 
        DtypeForCpuOp*>>& small_chunks
);


#endif