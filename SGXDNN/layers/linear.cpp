#include "linear.hpp"
#include "common_with_enclaves.h"
#include "secret_tensor.hpp"
#include "chunk_manager.hpp"

SGXLinearBuffer::SGXLinearBuffer(IdT FunId_) : FunId(FunId_) {}

void SGXLinearBuffer::init(
        IdT input, IdT output, IdT weight, IdT bias,
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    // der_input_tensor = GetTenById(der_input);
    // der_output_tensor = GetTenById(der_output);

    // size = num_channel * sizeof(byte)
    weight_tensor = GetTenById(weight);
    bias_tensor = GetTenById(bias);
    // der_weight_tensor = GetTenById(der_weight);
    // der_bias_tensor = GetTenById(der_bias);

    batch = batch_;
    input_size = input_size_;
    output_size = output_size_;

    printf("SGXLinearBuffer initialized: Batch %d, input %d, output %d\n", batch, input_size, output_size);

    printf(
        "SGXLinearBuffer chunk_size %d, total_input %d, output_size %d, weight_size %d\n", 
        STORE_CHUNK_ELEM, input_size * batch, batch*output_size, input_size*output_size);
    printf("features per chunk %d\n", STORE_CHUNK_ELEM/input_size);

    default_num_batches_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / input_size;
    default_num_col_per_chunk = std::min(STORE_CHUNK_ELEM, weight_tensor->GetNumElem()) / input_size;
    
    if (STORE_CHUNK_ELEM % input_size != 0)  {
        float ratio = STORE_CHUNK_ELEM / input_size;
        printf("SGXLinearBuffer STORE_CHUNK_ELEM num_rows != 0\n");
        printf("Chunk_size %d / inpu_size %d = %f \n", STORE_CHUNK_ELEM, input_size, ratio);
        return;
    }
}

int SGXLinearBuffer::get_num_batches_per_chunk(int num_elem_in_chunk) {
    return num_elem_in_chunk / input_size;
}

void SGXLinearBuffer::forward() {
    printf("SGXDNN_Main SGXLinearBuffer forward\n");
    
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *data_chunk, *weight_chunk, *output_chunk, *bias_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
    ChunkGuard<DtypeForCpuOp> weight_guard(StoreChunkPool::GetChunkPool(), weight_chunk);
    ChunkGuard<DtypeForCpuOp> output_guard(StoreChunkPool::GetChunkPool(), output_chunk);
    ChunkGuard<DtypeForCpuOp> bias_guard(StoreChunkPool::GetChunkPool(), bias_chunk);

    // Default eigen matrix is ColMajor
    
    
    MapMatRowMajor bias_mat(bias_chunk, 1, output_size);
    MapMatRowMajor output_mat(output_chunk, batch, output_size);

    chunk_manager.GetChunk(output_tensor->GetChunkId(0), output_chunk, batch*output_size*sizeof(DtypeForCpuOp));
    chunk_manager.GetChunk(bias_tensor->GetChunkId(0), bias_chunk, output_size*sizeof(DtypeForCpuOp));
    
    run_all_chunks([&](int data_start_store_chunk, int data_num_elem_in_store_chunk) {
        int data_chunk_size_in_byte = data_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
        int num_features_in_chunk = data_num_elem_in_store_chunk / input_size;
        MapMatRowMajor data_mat(data_chunk, num_features_in_chunk, input_size);
        int data_start_idx = data_start_store_chunk / input_size;
        chunk_manager.GetChunk(input_tensor->GetChunkId(data_start_store_chunk), data_chunk, data_chunk_size_in_byte);
        // printf("SGXDNN_Main Input array: ");
        // for (auto i=0; i<10; i++){
        //     printf("%f ", data_chunk[i]);
        // }
        // printf("\n");
        // printf("SGXDNN_Main Input mat: ");
        // for (auto i=0; i<10; i++){
        //     printf("%f ", data_mat(0,i));
        // }
        // printf("\n");
        run_all_chunks([&](int weight_start_store_chunk, int weight_num_elem_in_store_chunk) {
            
            int weight_chunk_size_in_byte = weight_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            int num_classes_in_chunk = weight_num_elem_in_store_chunk / input_size;
            int class_start_idx = weight_start_store_chunk / input_size;
            // printf("data start %d, end %d. weight start %d, end %d\n",
            //     data_start_idx, data_start_idx+num_features_in_chunk,
            //     class_start_idx, class_start_idx+num_classes_in_chunk);
            MapMatRowMajor weight_mat(weight_chunk, num_classes_in_chunk, input_size);
            // printf("Rows %d, cols %d\n", num_features_in_chunk, num_classes_in_chunk);
            chunk_manager.GetChunk(weight_tensor->GetChunkId(weight_start_store_chunk), weight_chunk, weight_chunk_size_in_byte);

            // printf("SGXDNN_Main Weight array: ");
            // for (auto i=0; i<10; i++){
            //     printf("%f ", weight_chunk[i]);
            // }
            // printf("\n");
            // printf("SGXDNN_Main Weight mat: ");
            // for (auto i=0; i<10; i++){
            //     printf("%f ", weight_mat(0,i));
            // }
            // printf("\n");

            // printf("new rows %d, cols %d, size %d\n", output_mat.rows(), output_mat.cols(), output_mat.size());
            // printf("%d %d, %d %d\n", batch, output_size, num_features_in_chunk, num_classes_in_chunk);
            auto output_block = output_mat.block(data_start_idx, class_start_idx, num_features_in_chunk, num_classes_in_chunk);
            // printf("data_mat: rows %d, cols %d, size %d\n", data_mat.rows(), data_mat.cols(), data_mat.size());
            // printf("weight_mat: rows %d, cols %d, size %d\n", weight_mat.rows(), weight_mat.cols(), weight_mat.size());
            // printf("weight_mat.transpose(): rows %d, cols %d, size %d\n", weight_mat.transpose().rows(), weight_mat.transpose().cols(), weight_mat.transpose().size());
            // printf("output_mat: rows %d, cols %d, size %d\n", output_block.rows(), output_block.cols(), output_block.size());
            output_block.array() = data_mat * weight_mat.transpose();
            
        }, STORE_CHUNK_ELEM, weight_tensor->GetNumElem());
    }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

    // printf("SGXDNN_Main Nobias Output mat: ");
    // for (auto i=0; i<10; i++){
    //     printf("%f ", output_mat(0,i));
    // }
    // printf("\n");
    // printf("SGXDNN_Main Bias mat: ");
    // for (auto i=0; i<10; i++){
    //     printf("%f ", bias_mat(0,i));
    // }
    // printf("\n");

    // auto output_block = output_mat.block(0, 0, 1, output_size);
    // cannot declare output_block outside the loop
    for (auto i=0; i<batch; i++){
        auto output_block = output_mat.block(i, 0, 1, output_size);
        // if ( i==0 ){
        //     printf("Debug bias: ");
        //     printf("output %f + bias %f", output_block(0,0), bias_mat(0,0));
        // }
        output_block.array() = output_block + bias_mat;
        // if (i==0){
        //     printf(" = %f, raw mat %f", output_block(0,0), output_mat(0,0));
        //     printf("\n");
        // }
        // printf("Debug raw mat %f\n", output_mat(0,0));
    }
    
    // printf("SGXDNN_Main Output mat: ");
    // for (auto i=0; i<10; i++){
    //     printf("%f ", output_mat(1,i));
    // }
    // printf("\n");
    chunk_manager.StoreChunk(output_tensor->GetChunkId(0), output_chunk, batch*output_size*sizeof(DtypeForCpuOp));

}


