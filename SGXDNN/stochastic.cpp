
#include "secret_tensor.hpp"
#include "chunk_manager.hpp"
#include "stochastic.hpp"

unordered_map<uint64_t, DtypeForCpuOp> quantize_exp;

void quantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten, uint64_t quantize_tag) {
    const int bits = 8;
    const int ebit = 8;
    const DtypeForCpuOp lower_limit = -pow(2, (bits - 1));
    const DtypeForCpuOp upper_limit = pow(2, (bits - 1)) - 1;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);
    //DtypeForCpuOp max_entry = 0;
    
	const __m256 neg8f = _mm256_set1_ps(-0.0f);
    __m256 tmp8f = _mm256_set1_ps(0.0f);

    auto get_max_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        int chunk_id = src_ten->GetChunkId(start_store_chunk);
        chunk_manager.GetChunk(chunk_id, store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&store_chunk[i]);
            const __m256 abs8f = _mm256_andnot_ps(neg8f, inp8f);
            const __m256 if_eq = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
            tmp8f = _mm256_blendv_ps(tmp8f, inp8f, if_eq);
        }
        //MapEigenVector src_vecmap(store_chunk, num_elem_in_store_chunk);
        //max_entry = std::max(max_entry, src_vecmap.cwiseAbs().maxCoeff());
    };
    run_all_chunks(get_max_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
    _mm256_stream_ps(dst_store_chunk, tmp8f);
    for(int i=4;i>0;i=i>>1){
        copy(dst_store_chunk+i,dst_store_chunk+2*i,dst_store_chunk+8);
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        const __m256 inp8f2 = _mm256_load_ps(&dst_store_chunk[8]);
        const __m256 if_eq = _mm256_cmp_ps(inp8f, inp8f2, 0x0e);
        const __m256 res8f = _mm256_blendv_ps(inp8f2, inp8f, if_eq);
        _mm256_stream_ps(dst_store_chunk, res8f);
    }

    if(1){
        dst_store_chunk[0] = (dst_store_chunk[0] == 0) ? 0: floor(log2(dst_store_chunk[0]));
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        //tmp8f = _mm256_set1_ps(pow(-2, (ebit - 1)));
        //__m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        //__m256 res8f = _mm256_blendv_ps(tmp8f, inp8f, if_gt); 
        tmp8f = _mm256_set1_ps(pow(2, (ebit - 1)) - 1);
        __m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        tmp8f = _mm256_blendv_ps(inp8f, tmp8f, if_gt);
        _mm256_stream_ps(dst_store_chunk, tmp8f);
    }
    DtypeForCpuOp exp = dst_store_chunk[0];

  //  DtypeForCpuOp exp = (max_entry == 0) ? 0 : floor(log2(max_entry));
  //  exp = std::min(std::max(exp, (DtypeForCpuOp) pow(-2, (ebit - 1))), (DtypeForCpuOp) pow(2, (ebit - 1) - 1));
    quantize_exp[quantize_tag] = exp;
    DtypeForCpuOp enlarge_factor = pow(2, -exp + (bits - 2));

    auto& xor_rnd = *get_fast_rng(quantize_tag);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float *input = store_chunk + start;
            float *output = dst_store_chunk + start;
            xor_rnd.rand_like(output, num_elem_in_op);
            for(uint64_t i=0;i<num_elem_in_op;i+=8){
				tmp8f = _mm256_set1_ps(enlarge_factor); 
                const __m256 inp8f = _mm256_load_ps(&input[i]);
                const __m256 out8f = _mm256_load_ps(&output[i]);
                const __m256 mul8f  = _mm256_mul_ps(inp8f, tmp8f);  
                const __m256 add8f = _mm256_add_ps(mul8f, out8f);  
                const __m256 flo8f = _mm256_floor_ps(add8f);
                tmp8f = _mm256_set1_ps(lower_limit);
                __m256 if_gt = _mm256_cmp_ps(flo8f, tmp8f, 0x0e);
                __m256 res8f = _mm256_blendv_ps(tmp8f, flo8f, if_gt);
                tmp8f = _mm256_set1_ps(upper_limit);
                if_gt = _mm256_cmp_ps(res8f, tmp8f, 0x0e);
                res8f = _mm256_blendv_ps(res8f, tmp8f, if_gt);
                _mm256_stream_ps(&output[i], res8f);
            }
            //MapEigenTensor in_map = MapEigenTensor(input, 1, 1, 1, num_elem_in_op);
            //MapEigenTensor out_map = MapEigenTensor(output, 1, 1, 1, num_elem_in_op);
            //out_map = (in_map * enlarge_factor + out_map).floor().cwiseMax(lower_limit).cwiseMin(upper_limit);
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
		//add
		chunk_manager.StoreChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		//add
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}

void dequantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten,
        uint64_t x_tag, uint64_t y_tag) {
    const int bits = 8;
    DtypeForCpuOp x_exp = quantize_exp[x_tag];
    DtypeForCpuOp y_exp = quantize_exp[y_tag];

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        MapEigenTensor src_map = MapEigenTensor(store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        MapEigenTensor dst_map = MapEigenTensor(dst_store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        DtypeForCpuOp shrink_factor = pow(2, x_exp - (bits - 2) + y_exp - (bits - 2));

        dst_map = src_map * shrink_factor;
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}