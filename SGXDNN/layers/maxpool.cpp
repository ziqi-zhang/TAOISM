#include "maxpool.hpp"
#include "common_with_enclaves.h"
#include "secret_tensor.hpp"
#include "chunk_manager.hpp"



MaxpoolBuffer::MaxpoolBuffer(IdT FunId_, IdT TenIdin_trans_, IdT TenIdout_trans_) : FunId(FunId_), TenIdin_trans(TenIdin_trans_), TenIdout_trans(TenIdout_trans_)  { }

void MaxpoolBuffer::transpose(const DtypeForCpuOp *src, DtypeForCpuOp *dst, const size_t N, const size_t M) {
#pragma omp parallel for
    for(size_t n = 0; n<N*M; n++) {
        size_t i = n/N;
        size_t j = n%N;
        dst[n] = src[M*j + i]; 
    }   
}

void MaxpoolBuffer::forward(
        shared_ptr<SecretTen> ten_in, shared_ptr<SecretTen> ten_out,
        shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
        uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
        uint32_t output_height, uint32_t output_width, uint32_t filter_height,
        uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {

    const uint32_t inputhw = input_height*input_width;
    uint32_t num_img_in_storechunk = STORE_CHUNK_ELEM/inputhw;

    if(STORE_CHUNK_ELEM % inputhw != 0){
        printf("!!!!!!!!!!!!!!!!!!! STORE_CHUNK_ELEM %% inputhw != 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return;
    }
    if (channel % 8 != 0){
        printf("Channel (%d) % 8 should be 0, but channel is not!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("Please change the form of AVX functions\n");
        return;
    }
    //if (num_img_in_storechunk % 8 != 0){
    //	printf("STORE_CHUNK_ELEM/inputhw is not divisible by 8!\n");
    //	return;
    //}

    const uint32_t radius_height = filter_height/2, radius_width = filter_width/2;
    const uint32_t outputhw = output_height * output_width;
    uint32_t outputsize_in_storechunk = num_img_in_storechunk * outputhw;
    const uint32_t total_size = batch * channel * inputhw;
    size_t idx_out=0;
    size_t idx_tmp=0;
    size_t size_of_store_chunk = STORE_CHUNK_ELEM * sizeof(float);      
    bool if_use_SSE_out =(outputhw%4==0);

    float* chunk_in, *chunk_out, *chunk_in_trans, *chunk_out_trans, *chunk_tmp;
    auto& chunk_manager = TrustedChunkManager::getInstance();

    ChunkGuard<DtypeForCpuOp> guard_in(StoreChunkPool::GetChunkPool(), chunk_in);
    ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    ChunkGuard<DtypeForCpuOp> guard_int(StoreChunkPool::GetChunkPool(), chunk_in_trans);
    ChunkGuard<DtypeForCpuOp> guard_outt(StoreChunkPool::GetChunkPool(), chunk_out_trans);
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp); // chunk_tmp is used to store output temporarily

    auto chunk_op = [&](size_t start_chunk, size_t num_elem_in, size_t num_elem_out) {
        num_img_in_storechunk = num_elem_in / inputhw;
        size_of_store_chunk = num_elem_in * sizeof(float);
        // printf("maxpooling forward in enclave. start_chunk: %d, num_elem_in %d, num_elem_out %d\n", start_chunk, num_elem_in, num_elem_out);
        chunk_manager.GetChunk(ten_in->GetChunkId(start_chunk), chunk_in, num_elem_in * sizeof(DtypeForCpuOp));
        // printf("Input: h=%d, w=%d\n", input_height, input_width);
        // for (auto ii=0; ii<input_height; ii++){
        //     for (auto jj=0; jj<input_width; jj++){
        //         printf("%f ", *(chunk_in+ii*input_width+jj));
        //     }
        //     printf("\n");
        // }
        // printf("transpose inputhw %d, num_img_in_storechunk %d \n", inputhw, num_img_in_storechunk);
        transpose_block_SSE4x4(chunk_in, chunk_in_trans, inputhw, num_img_in_storechunk, 8);
        // transpose(chunk_in, chunk_in_trans, num_img_in_storechunk, inputhw);
        // Save transpose chunk have problem when STORE_CHUNK_ELEM=1204224, channel=1024, imghw=8
        // chunk_manager.StoreChunk(ten_in_trans->GetChunkId(start_chunk), chunk_in_trans, size_of_store_chunk);
        // printf("Transpose input: h=%d, w=%d\n", input_height, input_width);
        // for (auto ii=0; ii<input_height; ii++){
        //     for (auto jj=0; jj<input_width; jj++){
        //         printf("%f ", *(chunk_in_trans+ii*input_width+jj));
        //     }
        //     printf("\n");
        // }
        fill(chunk_out_trans, chunk_out_trans + outputsize_in_storechunk, std::numeric_limits<DtypeForCpuOp>::lowest());
        for(uint32_t h = 0; h < input_height; ++h) {
            for(uint32_t w = 0; w < input_width; ++w) {
                // (h_start, h_end) * (w_start, w_end) is the range that the input
                // vector projects to.
                // const uint32_t h_start = (h < filter_height)
                //                         ? 0
                //                         : (h - filter_height) / row_stride + 1;
                // const uint32_t h_end = std::min(h / row_stride + 1, output_height);
                // const uint32_t w_start = (w < filter_width)
                //                         ? 0
                //                         : (w - filter_width) / col_stride + 1;
                // const uint32_t w_end = std::min(w / col_stride + 1, output_width);


                // const uint32_t h_start = h / row_stride;
                // const uint32_t h_end = std::min((h+filter_height)/row_stride, output_height);
                // const uint32_t w_start = w / col_stride;
                // const uint32_t w_end = std::min((w+filter_width)/col_stride , output_width);

                uint32_t h_start = (h < radius_height)
                                    ? 0
                                    : (h-radius_height + row_stride-1)/row_stride;
                uint32_t h_end = (h+radius_height)/row_stride+1;
                h_end = std::min<uint32_t>(h_end, output_height);

                uint32_t w_start = (w < radius_width)
                                    ? 0
                                    : (w-radius_width + col_stride-1)/col_stride;
                uint32_t w_end = (w+radius_width)/col_stride+1;
                w_end = std::min<uint32_t>(w_end, output_width);

                // if (h==0 && w==0)
                // printf(
                //     "(%d, %d): h[%d, %d], w[%d, %d]\n",
                //     h, w, h_start, h_end, w_start, w_end
                // );
                // compute elementwise max
                const uint32_t in_offset = (h * input_width + w)*num_img_in_storechunk;
                for (uint32_t ph = h_start; ph < h_end; ++ph) {
                    const uint32_t out_offset_base = ph * output_width;
                    for (uint32_t pw = w_start; pw < w_end; ++pw) {
                        const uint32_t out_offset = (out_offset_base + pw) * num_img_in_storechunk;
                        // printf(
                        //     "ph %d, pw %d, in_offset %d, out_offset_base %d, out_offset %d\n",
                        //     ph, pw, in_offset, out_offset_base, out_offset
                        // );
                        
                        // MaxpoolAVX(num_img_in_storechunk, chunk_in_trans+in_offset, chunk_out_trans + out_offset);
                        PlainMaxpool(num_img_in_storechunk, chunk_in_trans+in_offset, chunk_out_trans + out_offset);
                    }
                }
            }
        }
        // chunk_manager.StoreChunk(ten_out_trans->GetChunkId(start_chunk), chunk_out_trans, size_of_store_chunk);
        // printf("Save transposed output\n");
        // printf("Transpose output: h=%d, w=%d\n", output_height, output_width);
        // for (auto ii=0; ii<output_height; ii++){
        //     for (auto jj=0; jj<output_width; jj++){
        //         printf("%f ", *(chunk_out_trans+ii*output_width+jj));
        //     }
        //     printf("\n");
        // }
        // printf("use SSE %d\n", if_use_SSE_out);
        //transpose
        if(if_use_SSE_out){
            transpose_block_SSE4x4(chunk_out_trans, chunk_tmp, num_img_in_storechunk, outputhw, 8);
        }
        else{
            transpose(chunk_out_trans, chunk_tmp, outputhw, num_img_in_storechunk);
        }
        // transpose(chunk_out_trans, chunk_tmp, outputhw, num_img_in_storechunk);
        if(idx_tmp+num_elem_out<STORE_CHUNK_ELEM){
            copy(chunk_tmp, chunk_tmp+num_elem_out, chunk_out + idx_tmp);
            idx_tmp+=num_elem_out;
        }
        else{
            size_t idx_add = STORE_CHUNK_ELEM-idx_tmp;
            copy(chunk_tmp,chunk_tmp+idx_add,chunk_out+idx_tmp);
            chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, size_of_store_chunk);
            idx_out += STORE_CHUNK_ELEM;
            copy(chunk_tmp + idx_add,chunk_tmp + num_elem_out,chunk_out + idx_tmp+idx_add);
            idx_tmp += num_elem_out;
            idx_tmp -= STORE_CHUNK_ELEM; 
        }
    };//end of chunk_op
    run_all_chunks_for_maxpool(chunk_op, STORE_CHUNK_ELEM, batch * channel * inputhw, outputsize_in_storechunk, inputhw, outputhw);      

    if (idx_tmp!=0) {
        chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, idx_tmp * sizeof(DtypeForCpuOp)); 
    }
}//end maxpooling

void MaxpoolBuffer::backward(
        shared_ptr<SecretTen> ten_din, shared_ptr<SecretTen> ten_dout,
        shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
        uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
        uint32_t output_height, uint32_t output_width,
        uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {

    const uint32_t num_img = batch*channel;
    const uint32_t inputhw = input_height * input_width;
    const uint32_t num_img_in_storechunk = STORE_CHUNK_ELEM / inputhw;
    const uint32_t outputhw = output_height*output_width;
    uint32_t outputsize_in_storechunk = num_img_in_storechunk * outputhw;
    const uint32_t total_size = num_img * inputhw;
    const uint32_t total_size_out = num_img * outputhw;

    size_t idx_dout=0;
    size_t idx_tmp=0;
    bool if_use_SSE_out = (outputhw%4==0);
    float* chunk_din, *chunk_dout, *chunk_in_trans, *chunk_out_trans, *chunk_din_trans, *chunk_dout_trans, *chunk_tmp;
    auto& chunk_manager = TrustedChunkManager::getInstance();

    ChunkGuard<DtypeForCpuOp> guard_din(StoreChunkPool::GetChunkPool(), chunk_din);
    ChunkGuard<DtypeForCpuOp> guard_dout(StoreChunkPool::GetChunkPool(), chunk_dout);
    ChunkGuard<DtypeForCpuOp> guard_int(StoreChunkPool::GetChunkPool(), chunk_in_trans);
    ChunkGuard<DtypeForCpuOp> guard_outt(StoreChunkPool::GetChunkPool(), chunk_out_trans);
    ChunkGuard<DtypeForCpuOp> guard_dint(StoreChunkPool::GetChunkPool(), chunk_din_trans);
    ChunkGuard<DtypeForCpuOp> guard_doutt(StoreChunkPool::GetChunkPool(), chunk_dout_trans);
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);

    size_t start_chunk_out=0;
    if(total_size>=STORE_CHUNK_ELEM){
        size_t getsize_out;
        if(STORE_CHUNK_ELEM>total_size_out){
            getsize_out = total_size_out;
        }
        else{
            getsize_out = STORE_CHUNK_ELEM;
        }
        chunk_manager.GetChunk(ten_dout->GetChunkId(0), chunk_tmp, getsize_out * sizeof(DtypeForCpuOp));
        start_chunk_out += getsize_out; 
    }
    else{
        chunk_manager.GetChunk(ten_dout->GetChunkId(0), chunk_tmp, total_size_out * sizeof(float));
    }
    auto chunk_op = [&](size_t start_chunk, size_t num_elem_in, size_t num_elem_out) {
        chunk_manager.GetChunk(ten_in_trans->GetChunkId(start_chunk), chunk_in_trans, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(ten_out_trans->GetChunkId(start_chunk), chunk_out_trans, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
        if(num_elem_in == STORE_CHUNK_ELEM){    
            if(idx_tmp + outputsize_in_storechunk > STORE_CHUNK_ELEM){
                copy(chunk_tmp+idx_tmp,chunk_tmp+STORE_CHUNK_ELEM,chunk_dout);
                idx_dout = STORE_CHUNK_ELEM-idx_tmp;
                chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_tmp, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
                start_chunk_out += STORE_CHUNK_ELEM;
                idx_tmp = outputsize_in_storechunk-idx_dout;
                copy(chunk_tmp, chunk_tmp+idx_tmp, chunk_dout+idx_dout);
            }
            else{
                copy(chunk_tmp+idx_tmp,chunk_tmp+idx_tmp+outputsize_in_storechunk,chunk_dout);
                idx_tmp += outputsize_in_storechunk;
            }
        }
        else{
            if(idx_tmp==STORE_CHUNK_ELEM||idx_tmp==0){
                chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_dout, (total_size_out-start_chunk_out) * sizeof(DtypeForCpuOp));
            }
            else{
                copy(chunk_tmp+idx_tmp,chunk_tmp+STORE_CHUNK_ELEM,chunk_dout);
                idx_dout = STORE_CHUNK_ELEM-idx_tmp;
                if(total_size_out!=start_chunk_out)
                    chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_tmp, (total_size_out-start_chunk_out) * sizeof(DtypeForCpuOp));
                    //assume total_size_out-start_chunk_out+idx_dout<=STORE_CHUNK_ELEM
                idx_tmp = total_size_out - start_chunk_out;
                copy(chunk_tmp, chunk_tmp+idx_tmp, chunk_dout+idx_dout);
                    //idx_dout
            }
            
        }

        if(if_use_SSE_out){
            transpose_block_SSE4x4(chunk_dout, chunk_dout_trans, outputhw, num_img_in_storechunk, 4);
        }
        else{
            transpose(chunk_dout, chunk_dout_trans, num_img_in_storechunk, outputhw);
        }
        fill(chunk_din_trans, chunk_din_trans + STORE_CHUNK_ELEM,0);
        for(uint32_t h = 0; h < input_height; ++h) {
            for(uint32_t w = 0; w < input_width; ++w) {
                // (h_start, h_end) * (w_start, w_end) is the range that the input
                // vector projects to.
                const uint32_t h_start = (h < filter_height)
                                        ? 0
                                        : (h - filter_height) / row_stride + 1;
                const uint32_t h_end = std::min(h / row_stride + 1, output_height);
                const uint32_t w_start = (w < filter_width)
                                        ? 0
                                        : (w - filter_width) / col_stride + 1;
                const uint32_t w_end = std::min(w / col_stride + 1, output_width);
                // compute elementwise max
                const uint32_t in_offset = (h * input_width + w)*num_img_in_storechunk;
                for (uint32_t ph = h_start; ph < h_end; ++ph) {
                    const uint32_t out_offset_base = ph * output_width;
                    for (uint32_t pw = w_start; pw < w_end; ++pw) {
                        const uint32_t out_offset = (out_offset_base + pw) * num_img_in_storechunk;
                        MaxpoolbackAVX(num_img_in_storechunk, chunk_in_trans + in_offset, chunk_out_trans + out_offset, chunk_din_trans + in_offset, chunk_dout_trans + out_offset);
                    }
                }
            }
        }
        //transpose
        transpose_block_SSE4x4(chunk_din_trans, chunk_din, num_img_in_storechunk ,inputhw, 8);
        chunk_manager.StoreChunk(ten_din->GetChunkId(start_chunk), chunk_din, num_elem_in * sizeof(float));
    };//end of chunk_op
    run_all_chunks_for_maxpool(chunk_op, STORE_CHUNK_ELEM, total_size, outputsize_in_storechunk, inputhw, outputhw);
}//end maxpoolbackward
