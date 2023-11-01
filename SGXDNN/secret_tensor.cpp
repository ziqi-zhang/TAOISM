
#ifdef USE_SGX
#include "Enclave.h"
#endif


#include "secret_tensor.hpp"
#include "chunk_manager.hpp"
#include "common_with_enclaves.h"

using namespace std;

unordered_map<IdT, shared_ptr<SecretTen>> SecretTenHolder;
unordered_map<IdT, shared_ptr<EigenTensor>> TensorHolder;

SecretTen::SecretTen(IdT TenId_, DimsT* Dims_) : TenId(TenId_), Dims(*Dims_) { Init(); }

SecretTen::~SecretTen() { 
    for (auto& it: PrgStateHolder) free(it.second);
}

void SecretTen::Init() {
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    auto& chunk_manager = TrustedChunkManager::getInstance();

    auto chunk_op = [&](int start, int num_elem_in_op) {
        int chunk_id = chunk_manager.GetNewId();
        ChunkIds.push_back(chunk_id);
        // printf("num_elem_in_op %d, ", num_elem_in_op);
        chunk_manager.StoreChunk(chunk_id, store_chunk, num_elem_in_op * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
}

int SecretTen::GetChunkId(int start) {
    if (start >= GetNumElem()) {
        printf("The start exceed the size of the tensor.\n");
        throw std::invalid_argument("The start exceed the size of the tensor.");
    }
    // printf("SecretTen.GetChunkId ChunkIds (");
    // for (int i=0; i<ChunkIds.size(); i++){
    //     printf("%d, ", ChunkIds[i]);
    // }
    // printf(")\n");
    return ChunkIds[start / STORE_CHUNK_ELEM];
}

void SecretTen::GetStoreChunk(int start, DtypeForCpuOp* store_chunk, int num_byte) {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    int chunk_id = GetChunkId(start);
    chunk_manager.StoreChunk(chunk_id, store_chunk, num_byte * sizeof(DtypeForCpuOp));
}

void SecretTen::SetTen(DtypeForCpuOp* Arr) {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    auto chunk_op = [&](int start, int num_elem_in_op) {
        int chunk_id = GetChunkId(start);
        DtypeForCpuOp* src_arr = Arr + start;
        chunk_manager.StoreChunk(chunk_id, src_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
}

void SecretTen::GetTen(DtypeForCpuOp* Arr) {
    auto& chunk_manager = TrustedChunkManager::getInstance();
    auto chunk_op = [&](int start, int num_elem_in_op) {
        int chunk_id = GetChunkId(start);
        DtypeForCpuOp* dst_arr = Arr + start;
        chunk_manager.GetChunk(chunk_id, dst_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
}


void SecretTen::SetSeed(uint64_t RawSeed) {
    SeedT seed;
    memset(seed, 0, sizeof(SeedT));
    auto TmpRawSeed = RawSeed;
    for (int i = 0; TmpRawSeed > 0; i++) {
        seed[i] = (uint8_t) (TmpRawSeed & ((1 << 9) - 1));
        TmpRawSeed >>= 8;
    }
    PrgStateHolder[RawSeed] = (aes_stream_state*)memalign(16, sizeof(aes_stream_state));
    InitPrgWithSeed(PrgStateHolder[RawSeed], seed);
}

void SecretTen::GetRandom(DtypeForCpuOp* DstArr, uint64_t RawSeed) {
    auto PrgState = PrgStateHolder[RawSeed];
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    auto chunk_op = [&](int start, int num_elem_in_op) {
        float* input = DstArr + start;
        get_r(PrgState, (uint8_t*) input, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
        for(size_t j = 0; j < num_elem_in_op; j++) {
            input[j] -= floor(input[j] * invPLimit) * PLimit;
            input[j] = (input[j] >= mid) ? (input[j] - p) : input[j];
        }
    };
    run_all_chunks(chunk_op, WORK_CHUNK_ELEM, GetNumElem());
}

shared_ptr<SecretTen> GetTenById(IdT TenId) {
    return SecretTenHolder[TenId];
}
