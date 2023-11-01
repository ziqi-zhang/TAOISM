#ifndef USE_SGX
#define EIGEN_USE_THREADS
#include <malloc.h>
#else
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#endif

#include <time.h>

#include "conv.hpp"
#include "common_with_enclaves.h"
#include "secret_tensor.hpp"
#include "chunk_manager.hpp"
// #include "utils.hpp"


template <typename Func>
void run_all_chunks_conv(Func chunk_op, int num_elem_in_chunk, int num_elem) {
    int start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk ) {
        chunk_op(start_chunk, num_elem_in_chunk, start_chunk + num_elem_in_chunk == num_elem);
    }
    if (start_chunk < num_elem) chunk_op(start_chunk, num_elem - start_chunk, true);
}

SGXConvBuffer::SGXConvBuffer(IdT FunId_) : FunId(FunId_) {}

void SGXConvBuffer::init(
        IdT input, IdT output, IdT weight, IdT bias,
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_h_, uint32_t input_w_, uint32_t input_c_,
        uint32_t output_h_, uint32_t output_w_, uint32_t output_c_,
        uint32_t kernel_, uint32_t padding_, uint32_t stride_) {
    #ifdef PRINT_CONV_INIT_INFO
        printf("SGX Conv Buffer init\n", input);
    #endif

    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    // der_input_tensor = GetTenById(der_input);
    // der_output_tensor = GetTenById(der_output);

    // size = num_channel * sizeof(byte)
    weight_tensor = GetTenById(weight);
    // der_weight_tensor = GetTenById(der_weight);
    bias_tensor = GetTenById(bias);
    // der_bias_tensor = GetTenById(der_bias);

    batch = batch_;
    input_h = input_h_; input_w = input_w_; input_c = input_c_;
    output_h = output_h_; output_w = output_w_; output_c = output_c_;
    kernel = kernel_; padding = padding_; stride = stride_;

    input_row_size = input_w * input_c; one_batch_input_size = input_h * input_row_size;
    if (STORE_CHUNK_ELEM % (input_row_size*stride) != 0){
        printf("STORE_CHUNK_ELEM %d cannot divide input_row_size*stride %d*%d, STORE_CHUNK_ELEM %% input_row_size=%d\n", 
        STORE_CHUNK_ELEM, input_row_size, stride, STORE_CHUNK_ELEM % (input_row_size*stride));
    }
    assert (STORE_CHUNK_ELEM % input_row_size == 0);
    int input_rows_per_chunk = STORE_CHUNK_ELEM / input_row_size;
    input_elem_fetch_per_chunk = input_rows_per_chunk * input_row_size;
    int num_input_per_chunk = input_rows_per_chunk / input_h;
    int ramain_rows_per_chunk = input_rows_per_chunk % input_h;
    #ifdef PRINT_CONV_INIT_INFO
        printf(
            "ChunkElem %d, row_size %d, rows %d, remain %d, fetch_elem %d\n", 
            STORE_CHUNK_ELEM, input_row_size, input_rows_per_chunk, STORE_CHUNK_ELEM % input_row_size, input_elem_fetch_per_chunk);
    #endif

    // pytorch weight shape is [output_c, input_c, kernel, kernel]
    #ifdef PRINT_CONV_INIT_INFO
        printf(
            "SGXConvBuffer initialized: Batch %d, input [%d,%d,%d], output [%d,%d,%d], weight [%d,%d,%d,%d], ", 
            batch, input_c, input_h, input_w, output_c, output_h, output_w,
            output_c, input_c, kernel, kernel);
        printf("kernel %d, padding %d, stride %d\n", kernel, padding, stride);
        printf("chunk_size %d, single batch size %d\n", STORE_CHUNK_ELEM, input_c*input_h*input_w);
    #endif

    patch_size = kernel * kernel * input_c;
    // Compute max im2col patches
    assert (STORE_CHUNK_ELEM > patch_size);
    max_im2col_patches_per_chunk = STORE_CHUNK_ELEM / patch_size;
    int total_im2col_patches = batch * output_h * output_w;
    max_im2col_patches_per_chunk = std::min(max_im2col_patches_per_chunk, total_im2col_patches);
    // Compute max output patches
    if (STORE_CHUNK_ELEM < output_c)
        printf("output channel (%d) is larger than STORE_CHUNK_ELEM\n", output_c);
    assert (STORE_CHUNK_ELEM >= output_c);
    if (STORE_CHUNK_ELEM % output_c != 0){
        printf("STORE_CHUNK_ELEM %d cannot divide output_channel %d, STORE_CHUNK_ELEM %% output_c=%d\n", 
            STORE_CHUNK_ELEM, output_c, STORE_CHUNK_ELEM % output_c);
    }
    assert (STORE_CHUNK_ELEM % output_c == 0);
    max_output_patches_per_chunk = STORE_CHUNK_ELEM / output_c;
    int total_output_patches = batch * output_h * output_w;
    max_output_patches_per_chunk = std::min(max_output_patches_per_chunk, total_output_patches);
    // Compute matrix mul rows
    max_matrix_mul_rows = std::min(max_im2col_patches_per_chunk, max_output_patches_per_chunk);
    // max_matrix_mul_rows = max_im2col_patches_per_chunk;
    // max_matrix_mul_rows = max_output_patches_per_chunk;
    
    im2col_patches_per_chunk = max_matrix_mul_rows * (max_im2col_patches_per_chunk / max_matrix_mul_rows);
    // output_patches_per_chunk = max_matrix_mul_rows * (max_output_patches_per_chunk / max_matrix_mul_rows);
    output_patches_per_chunk = max_output_patches_per_chunk;
    im2col_num_elem_in_chunk = im2col_patches_per_chunk * patch_size;
    output_num_elem_in_chunk = output_patches_per_chunk * output_c;

    max_weight_rows_per_chunk = max_im2col_patches_per_chunk;
    max_weight_elem_per_chunk = max_weight_rows_per_chunk * patch_size;

    #ifdef PRINT_CONV_INIT_INFO
        printf(
            "input  row_size %d, input_rows_per_chunk %d, inpu_chunk_size %d\n", 
            input_row_size, input_rows_per_chunk, input_row_size*input_rows_per_chunk);
        printf(
            "im2col row size %d, max_im2col_patches_per_chunk %d, max_matrix_mul_rows %d, im2col_patches_per_chunk %d, im2col_chunk_size %d\n", 
            patch_size, max_im2col_patches_per_chunk, max_matrix_mul_rows, im2col_patches_per_chunk, im2col_num_elem_in_chunk);
        printf(
            "output row size %d, max_output_patches_per_chunk %d, max_matrix_mul_rows %d, output_patches_per_chunk %d, output_chunk_size %d\n", 
            output_c, max_output_patches_per_chunk, max_matrix_mul_rows, output_patches_per_chunk, output_num_elem_in_chunk);
        printf(
            "weight row size %d, max_weight_rows_per_chunk %d, weight_chunk_size %d\n", 
            patch_size, max_weight_rows_per_chunk, max_weight_elem_per_chunk);
        printf("SGX Conv Buffer finish init\n");
    #endif

}


void SGXConvBuffer::forward() {
    // printf(
    //     "Secret Conv Forward, input (%d,%d,%d), output (%d,%d,%d), kernel %d, stride %d, padding %d\n",
    //     input_h, input_w, input_c,  output_h, output_w, output_c, kernel, stride,  padding
    // );
    #ifdef PRINT_RUN_TIME_INFO
        sgx_time_t total_start = get_time();
    #endif
    int output_width, output_height, stride_cols, stride_rows, filter_height, filter_width;
    int input_batches, input_height, input_width, input_depth;
    output_width = output_w; output_height = output_h;
    stride_cols = stride; stride_rows = stride;
    filter_height = kernel; filter_width = kernel;
    input_batches = batch; input_height = input_h; input_width = input_w; input_depth = input_c;

    sgx_time_t load_input_start_time, load_weight_start_time, save_output_start_time, im2col_construction_start_time, matrix_mul_start_time, forward_prepare_start_time;
    #ifdef PRINT_RUN_TIME_INFO
        forward_prepare_start_time = get_time();
    #endif
    const int filter_value_count = filter_width * filter_height * input_depth;
    if ((filter_value_count * sizeof(DtypeForCpuOp)) > STORE_CHUNK_ELEM){
        printf(
            "filter_value_count size is larger than STORE_CHUNK_ELEM, filter_value_count(%d) * DtypeForCpuOp(4) = %d > STORE_CHUNK_ELEM(%d)\n",
            filter_value_count, filter_value_count * sizeof(DtypeForCpuOp), STORE_CHUNK_ELEM
        );
    }
    assert((filter_value_count * sizeof(DtypeForCpuOp)) <= STORE_CHUNK_ELEM);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *data_chunk, *weight_chunk, *output_chunk, *im2col_chunk, *bias_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
    ChunkGuard<DtypeForCpuOp> weight_guard(StoreChunkPool::GetChunkPool(), weight_chunk);
    ChunkGuard<DtypeForCpuOp> output_guard(StoreChunkPool::GetChunkPool(), output_chunk);
    ChunkGuard<DtypeForCpuOp> im2col_guard(StoreChunkPool::GetChunkPool(), im2col_chunk);
    ChunkGuard<DtypeForCpuOp> bias_guard(StoreChunkPool::GetChunkPool(), bias_chunk);
    // DtypeForCpuOp tmp_output_chunk[max_matrix_mul_rows * output_c];
    // DtypeForCpuOp* operate_weight_chunk = (DtypeForCpuOp*)malloc((max_weight_rows_per_chunk+3) * output_c * sizeof(DtypeForCpuOp));
    // DtypeForCpuOp operate_weight_chunk[(max_weight_rows_per_chunk+3) * output_c];
    // printf("Begin operate weight %p\n", operate_weight_chunk);
    // DtypeForCpuOp* redundent_weight_chunk = (DtypeForCpuOp*)malloc(3 * output_c * sizeof(DtypeForCpuOp));
    // DtypeForCpuOp redundent_weight_chunk[3 * output_c];
    // printf("Begin redundent weight %p\n", redundent_weight_chunk);
    DtypeForCpuOp* operate_weight_chunk, *redundent_weight_chunk;
    int redundant_weight_cnt = 0;
    // DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + max_matrix_mul_rows * output_c;
    // MapMatRowMajor tmp_output_mat(tmp_output_chunk, max_matrix_mul_rows, output_c);
    MapMatRowMajor bias_mat(bias_chunk, 1, output_c);
    chunk_manager.GetChunk(bias_tensor->GetChunkId(0), bias_chunk, output_c*sizeof(DtypeForCpuOp));
    chunk_manager.GetChunk(output_tensor->GetChunkId(0), output_chunk, output_num_elem_in_chunk*sizeof(DtypeForCpuOp));
    #ifdef PRINT_RUN_TIME_INFO
        sgx_time_t end = get_time();
        forward_prepare_time = get_elapsed_time(forward_prepare_start_time, end);
    #endif

    // Operate input chunks
    DtypeForCpuOp *operate_input_chunk, *redundant_input_chunk;
    int redundant_input_rows = 0, input_row_idx_total = 0;
    int filter_radius_height = filter_height / 2, filter_radius_width = filter_width / 2;
    // int reuse_num_elem = filter_radius_height * input_width * input_depth;
    int im2col_row_idx_in_chunk = 0, output_row_idx_in_chunk = 0, im2col_row_idx_total = 0, output_row_idx_total = 0;
    run_all_chunks_conv([&](int data_start_store_chunk, int data_num_elem_in_store_chunk, bool last_chunk) {
        
        int data_chunk_size_in_byte = data_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
        #ifdef PRINT_RUN_TIME_INFO
            load_input_start_time = get_time();
        #endif
        chunk_manager.GetChunk(input_tensor->GetChunkId(data_start_store_chunk), data_chunk, data_chunk_size_in_byte);
        
        
        
        #ifdef PRINT_CONV_INPUT_LOAD_CHUNK_INFO
            int data_chunk_start_row = data_start_store_chunk / input_row_size;
            int data_chunk_start_batch = data_chunk_start_row / input_height;
            int data_chunk_start_y_in_batch = data_chunk_start_row % input_height;
            int data_chunk_end_row = (data_start_store_chunk + data_num_elem_in_store_chunk) / input_row_size;
            int data_chunk_end_batch = data_chunk_end_row / input_height;
            int data_chunk_end_y_in_batch = data_chunk_end_row % input_height;
            printf(
                "Load elem %d-%d, batch %d row %d to batch %d row %d\n",
                data_start_store_chunk, data_start_store_chunk + data_num_elem_in_store_chunk,
                data_chunk_start_batch, data_chunk_start_y_in_batch,
                data_chunk_end_batch, data_chunk_end_y_in_batch
            );
        #endif

        int operate_input_rows = data_num_elem_in_store_chunk / input_row_size + redundant_input_rows;
        // printf("Before malloc operate_input_chunk\n");
        operate_input_chunk = (DtypeForCpuOp*)malloc(operate_input_rows*input_row_size*sizeof(DtypeForCpuOp));
        // printf("After malloc operate_input_chunk\n");
        if (redundant_input_rows > 0){
            // printf("Pre-saved redundant rows %d\n", redundant_input_rows);
            memcpy(operate_input_chunk, redundant_input_chunk, redundant_input_rows*input_row_size*sizeof(DtypeForCpuOp));
            free(redundant_input_chunk);
        }
        DtypeForCpuOp* operate_input_chunk_copy_start = operate_input_chunk + redundant_input_rows*input_row_size;
        memcpy(operate_input_chunk_copy_start, data_chunk, data_num_elem_in_store_chunk*sizeof(DtypeForCpuOp));
        MapMatRowMajor operate_data_mat(operate_input_chunk, operate_input_rows, input_row_size);
        // printf("Operate input chunk shape (%d, %d)\n", operate_input_rows, input_row_size);
        
        if (!last_chunk){
            int new_redundant_input_rows = std::max(0, filter_height - stride);
            // printf("new_redundant_input_rows %d\n", new_redundant_input_rows);
            int new_redundant_input_elem_num = new_redundant_input_rows*input_row_size;
            // printf("Before malloc redundant_input_chunk, size %d\n", new_redundant_input_elem_num);
            redundant_input_chunk = (DtypeForCpuOp*)malloc(new_redundant_input_elem_num*sizeof(DtypeForCpuOp));
            // printf("After malloc redundant_input_chunk\n");
            DtypeForCpuOp* new_redundant_input_start = data_chunk + data_num_elem_in_store_chunk - new_redundant_input_elem_num;
            // printf("Before copy, size %d\n", new_redundant_input_elem_num);
            memcpy(redundant_input_chunk, new_redundant_input_start, new_redundant_input_elem_num*sizeof(DtypeForCpuOp));
            // printf("After copy\n");
            redundant_input_rows = new_redundant_input_rows;
        }
        // load_input_time += clock() - load_input_start_time;

        // print input
        // printf("Input: \n");
        // printf("%f %f %f\n", data_chunk[0], data_chunk[1], data_chunk[2]);
        // for (int r=0; r<input_height; r++){
        //     for (int c=0; c<input_width; c++){
        //         printf("[");
        //         for (int d=0; d<input_depth; d++){
        //             int bias = r * input_width * input_depth +
        //                 c * input_depth + d;
        //             DtypeForCpuOp* p = data_chunk + bias;
        //             printf("%f  ", *p);
        //         }
        //         printf("],  ");
        //     }
        //     printf("\n");
        // }
        #ifdef PRINT_RUN_TIME_INFO
            load_input_time += get_elapsed_time(load_input_start_time, get_time());
        #endif

        // row_idx_in_chunk, col_idx is the centor pixel in the input feature
        // printf("Im2col: \n");
        int end_row_idx_in_operate_chunk, start_row_idx_in_operate_chunk;
        if (!last_chunk)
            end_row_idx_in_operate_chunk = operate_input_rows-filter_radius_height;
        else
            end_row_idx_in_operate_chunk = operate_input_rows;
        if (data_start_store_chunk == 0)
            start_row_idx_in_operate_chunk = 0;
        else
            start_row_idx_in_operate_chunk = filter_radius_height;
        // for (int relevant_up_row_idx_in_chunk=0; relevant_up_row_idx_in_chunk < end_row_idx_in_chunk; relevant_up_row_idx_in_chunk+=stride_rows){
        // for (int row_idx_in_chunk=start_row_idx_in_operate_chunk; row_idx_in_chunk < end_row_idx_in_operate_chunk; row_idx_in_chunk+=stride_rows){
        int row_idx_in_chunk = start_row_idx_in_operate_chunk;
        #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
            printf("New input chunk, start row %d, end row %d\n", start_row_idx_in_operate_chunk, end_row_idx_in_operate_chunk);
        #endif
        // printf("Tag2\n");
        while (row_idx_in_chunk < end_row_idx_in_operate_chunk){
            int batch_idx = input_row_idx_total / input_height;
            int row_idx_in_batch = input_row_idx_total % input_height;
            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
            if (batch_idx == 3 && row_idx_in_batch < 10)
                printf("batch %d, row %d, input_row_total %d, input_row_in_batch %d\n", batch_idx, row_idx_in_batch, input_row_idx_total, row_idx_in_batch);
            #endif
            for (int col_idx=0; col_idx < input_width; col_idx+=stride_cols){
                // sgx_time_t test_start = get_time();
                #ifdef PRINT_RUN_TIME_INFO
                    im2col_construction_start_time = get_time();
                #endif
                const int in_y_origin = row_idx_in_batch - filter_radius_height;
                const int in_x_origin = col_idx - filter_radius_width;
                const int out_y = row_idx_in_batch / stride_rows, out_x = col_idx / stride_cols;
                DtypeForCpuOp* im2col_row_start = im2col_chunk + im2col_row_idx_in_chunk * patch_size;
                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                if (batch_idx == 3 && out_y==3 and (out_x==2 || out_x==1)){
                    printf(
                        "im2col_row_idx %d:  ", im2col_row_idx_total+im2col_row_idx_in_chunk
                    );
                    printf(
                        "batch %d input_center [%d,%d], origin [%d,%d], out [%d,%d]. ",
                        batch_idx, row_idx_in_batch, col_idx, in_y_origin, in_x_origin, out_y, out_x
                    );
                    printf("(");
                }
                #endif
                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                    int in_y = in_y_origin + filter_y;
                    #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                    if (batch_idx == 3 && out_y==3 and (out_x==2 || out_x==1)){
                        printf("in_y-%d", in_y);
                    }
                    #endif
                    if ((in_y < 0) || (in_y >= input_height)) {
                        if (in_y < 0){
                            int start = filter_y*filter_width*input_depth;
                            int end = start + (filter_width * input_depth);
                            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                            if (batch_idx == 3 && out_y==3 and out_x==2){
                                printf("[%d<0,%d-%d]", in_y, start, end);
                            }
                            #endif
                            DtypeForCpuOp* im2col_fill_row_start = im2col_row_start + start;
                            DtypeForCpuOp* im2col_fill_row_end = im2col_row_start + end;
                                // im2col_fill_row_start + (filter_width * input_depth);
                            std::fill(im2col_fill_row_start, im2col_fill_row_end, DtypeForCpuOp(0));
                        }
                        if (in_y >= input_height){
                            int end = patch_size - (filter_height-1-filter_y)*filter_width*input_depth;
                            int start = end - (filter_width * input_depth);
                            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                            if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                                printf("[%d>=%d,%d-%d]", in_y, input_height, start, end);
                            }
                            #endif
                            DtypeForCpuOp* im2col_fill_row_end =
                                im2col_row_start + end;
                            DtypeForCpuOp* im2col_fill_row_start = 
                                im2col_row_start + start;
                            // DtypeForCpuOp* im2col_fill_row_start = 
                            //     im2col_fill_row_end - (filter_width * input_depth);
                            std::fill(im2col_fill_row_start, im2col_fill_row_end, DtypeForCpuOp(0));
                        }
                    } else{
                        const int in_x_end = in_x_origin + filter_width;
                        const int left_zero_count = std::max(0, 0 - in_x_origin);
                        const int right_zero_count = std::max(0, in_x_end - input_width);
                        const int center_copy_count = filter_width - (left_zero_count + right_zero_count);
                        #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                        if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                            printf("[%d-%d-%d]\n", left_zero_count, center_copy_count, right_zero_count);
                        }
                        #endif
                        if (left_zero_count > 0) {
                            DtypeForCpuOp* im2col_left_start = im2col_row_start + filter_y * filter_width * input_depth;
                            DtypeForCpuOp* im2col_left_end =
                                im2col_left_start + (left_zero_count * input_depth);
                            std::fill(im2col_left_start, im2col_left_end, DtypeForCpuOp(0));
                            
                        }
                        if (center_copy_count > 0) {
                            int row_bias = filter_y - filter_radius_height;
                            const DtypeForCpuOp* input_row_start =
                                operate_input_chunk + ((row_idx_in_chunk + row_bias) * input_width * input_depth) +
                                (std::max(0, in_x_origin) * input_depth);
                            const DtypeForCpuOp* input_row_end =
                                input_row_start + (center_copy_count * input_depth);
                            DtypeForCpuOp* im2col_center_start =
                                im2col_row_start + filter_y * filter_width * input_depth + (left_zero_count * input_depth);
                            std::copy(input_row_start, input_row_end, im2col_center_start);
                            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                            if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                                printf("b %d, out_y %d, out_x %d Window %d: ", batch_idx, out_y, out_x, in_y);
                                for (auto i=0; i<5; i++){
                                    printf("%.2f, ", *(input_row_start+i));
                                }
                                printf("\n");
                            }
                            #endif
                        }
                        if (right_zero_count > 0) {
                            DtypeForCpuOp* im2col_right_start =
                                im2col_row_start + filter_y * filter_width * input_depth +
                                ((left_zero_count + center_copy_count) * input_depth);
                            DtypeForCpuOp* im2col_right_end =
                                im2col_right_start + (right_zero_count * input_depth);
                            std::fill(im2col_right_start, im2col_right_end, DtypeForCpuOp(0));
                        }
                    }
                }
                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                    // printf(") ");
                    // for (int idx=0; idx < patch_size; idx++){
                    //     printf("%.0f, ", *(im2col_row_start+idx));
                    // }
                    // printf("\n");
                if (batch_idx == 3 && out_y==3 && (out_x==2|| out_x==1)){
                    printf(") \n");
                    for (int idx=0; idx < patch_size; idx++){
                        if (idx%64 == 0)
                            printf("b %d, out_y %d, out_x %d im2col line %d: ", batch_idx, out_y, out_x, idx/64);
                        if (idx%64<5)
                            printf("%.2f, ", *(im2col_row_start+idx));
                        if ((idx+1)%64 == 0)
                            printf("\n");
                    }
                    printf("\n");
                }
                #endif
                im2col_row_idx_in_chunk ++;
                // output_row_idx_in_chunk ++;
                #ifdef PRINT_RUN_TIME_INFO
                    im2col_construction_time += get_elapsed_time(im2col_construction_start_time, get_time());
                #endif

                // for (int i=0; i<patch_size; i++){
                //     printf("%f ", im2col_row_start[i]);
                // }
                // printf("\n");

                // bool im2col_last_row = (last_chunk && row_idx_in_chunk == end_row_idx_in_operate_chunk-stride_rows && col_idx == input_width-stride_cols );
                int last_input_last_row = int((input_height-1)/stride_rows) * stride_rows;
                int last_col = int((input_width-1)/stride_cols)*stride_cols;
                bool im2col_last_row = (
                    last_chunk && 
                    input_row_idx_total == (batch-1)*input_height + last_input_last_row && 
                    // row_idx_in_chunk == end_row_idx_in_operate_chunk-stride_rows &&
                    col_idx == last_col
                );
                // if (last_chunk)
                //     printf(
                //         "last_input_last_row %d, input_row_idx_total %d, row bar %d, col_idx %d, col bar %d \n", 
                //         last_input_last_row, input_row_idx_total, (batch-1)*input_height + last_input_last_row, col_idx, last_col
                //     );
                if ((im2col_row_idx_in_chunk % max_matrix_mul_rows == 0) || 
                    im2col_last_row
                ){
                    
                    // printf(
                    //     "im2col_row_idx_total %d, im2col_row_idx_in_chunk %d\n", 
                    //     im2col_row_idx_total, im2col_row_idx_in_chunk
                    // );
                    int matrix_mul_rows;
                    if (im2col_row_idx_in_chunk % max_matrix_mul_rows == 0){
                        // assert (output_row_idx_in_chunk % max_matrix_mul_rows == 0);
                        matrix_mul_rows = max_matrix_mul_rows;
                    }
                    else
                        matrix_mul_rows = im2col_row_idx_in_chunk % max_matrix_mul_rows;
                    
                    // set im2col multiply matrix
                    auto im2col_start_row_idx_in_chunk = im2col_row_idx_in_chunk - matrix_mul_rows;
                    auto im2col_mat_start_in_chunk = im2col_chunk + im2col_start_row_idx_in_chunk * patch_size;
                    MapMatRowMajor im2col_mat(im2col_mat_start_in_chunk, matrix_mul_rows, patch_size);

                    // printf("Im2col Mat:\n");
                    // for (auto r=0; r<matrix_mul_rows; r++){
                    //     for (auto c=0; c<patch_size; c++)
                    //         printf("%.0f ", im2col_mat(r,c));
                    //     printf("\n");
                    // }

                    // Not used
                    // auto output_mat_start_in_chunk = output_chunk + (output_row_idx_in_chunk - matrix_mul_rows) * output_c;
                    // printf("Before malloc tmp_output_chunk\n");
                    DtypeForCpuOp* tmp_output_chunk = (DtypeForCpuOp*)malloc(matrix_mul_rows * output_c * sizeof(DtypeForCpuOp));
                    // printf("After malloc tmp_output_chunk, matrix_mul_rows %d, output_c %d\n", matrix_mul_rows, output_c);
                    // printf("Malloc size %d, matrix_mul_rows*output_c %d\n", mspace_usable_size(tmp_output_chunk), matrix_mul_rows*output_c*sizeof(DtypeForCpuOp) );
                    MapMatRowMajor tmp_output_mat(tmp_output_chunk, matrix_mul_rows, output_c);
                    redundant_weight_cnt = 0;
                    int out_channel_start_idx = 0;

                    

                    run_all_chunks([&](int weight_start_store_chunk, int weight_num_elem_in_store_chunk) {
                        int weight_chunk_size_in_byte = weight_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                        #ifdef PRINT_RUN_TIME_INFO
                            load_weight_start_time = get_time();
                        #endif
                        chunk_manager.GetChunk(weight_tensor->GetChunkId(weight_start_store_chunk), weight_chunk, weight_chunk_size_in_byte);
                        

                        // printf(
                        //     "Weight_chunk start %d, end %d, num_elem %d, prior redundant %d\n", 
                        //     weight_start_store_chunk, weight_start_store_chunk+weight_num_elem_in_store_chunk, weight_num_elem_in_store_chunk,
                        //     redundant_weight_cnt
                        // );
                        int valid_weight_row_cnt = (redundant_weight_cnt + weight_num_elem_in_store_chunk) / patch_size;
                        // printf("Before malloc operate_weight_chunk\n");
                        operate_weight_chunk = (DtypeForCpuOp*)malloc(valid_weight_row_cnt * patch_size * sizeof(DtypeForCpuOp));
                        // printf("After malloc operate_weight_chunk\n");
                        // printf("Test 1 size %d\n", valid_weight_row_cnt * patch_size);
                        
                        // DtypeForCpuOp* redundent_weight_end_idx = redundent_weight_chunk + redundant_weight_cnt;
                        if (redundant_weight_cnt > 0){
                            // std::copy(redundent_weight_chunk, redundent_weight_end_idx, operate_weight_chunk);
                            // printf("Test 2 redundant size %d, ", redundant_weight_cnt);
                            memcpy(operate_weight_chunk, redundent_weight_chunk, redundant_weight_cnt*sizeof(DtypeForCpuOp));
                            free(redundent_weight_chunk);
                            // printf("Finished\n");
                        }
                        
                        
                        
                        if (redundant_weight_cnt > 0){
                            int copy_to_operate_weight_num_elem = valid_weight_row_cnt * patch_size - redundant_weight_cnt;
                            DtypeForCpuOp* weight_chunk_copy_end = weight_chunk + copy_to_operate_weight_num_elem;
                            DtypeForCpuOp* operate_weight_copy_start = operate_weight_chunk + redundant_weight_cnt;
                            // printf("copy size %d, total size %d\n", copy_to_operate_weight_num_elem, copy_to_operate_weight_num_elem+redundant_weight_cnt);
                            // std::copy(weight_chunk, weight_chunk_copy_end, operate_weight_copy_start);
                            memcpy(operate_weight_copy_start, weight_chunk, copy_to_operate_weight_num_elem*sizeof(DtypeForCpuOp));
                        } else{
                            int copy_to_operate_weight_num_elem = valid_weight_row_cnt * patch_size;
                            memcpy(operate_weight_chunk, weight_chunk, copy_to_operate_weight_num_elem*sizeof(DtypeForCpuOp));
                            // printf("Test 2 size %d\n", copy_to_operate_weight_num_elem);
                        }
                        

                        int new_redundant_weight_cnt = (redundant_weight_cnt + weight_num_elem_in_store_chunk) % patch_size;
                        if (new_redundant_weight_cnt > 0){
                            DtypeForCpuOp* weight_end_in_chunk = weight_chunk + weight_num_elem_in_store_chunk;
                            DtypeForCpuOp* redundent_weight_start_in_chunk = weight_end_in_chunk - new_redundant_weight_cnt;
                            // std::copy(redundent_weight_start_in_chunk, weight_end_in_chunk, redundent_weight_chunk);
                            // printf("Before malloc redundent_weight_chunk\n");
                            redundent_weight_chunk = (DtypeForCpuOp*)malloc(new_redundant_weight_cnt * sizeof(DtypeForCpuOp));
                            // printf("After malloc redundent_weight_chunk\n");
                            memcpy(redundent_weight_chunk, redundent_weight_start_in_chunk, new_redundant_weight_cnt*sizeof(DtypeForCpuOp));
                        }
                        redundant_weight_cnt = new_redundant_weight_cnt;

                        int out_channels_in_chunk = valid_weight_row_cnt;
                        // int out_channel_start_idx = weight_start_store_chunk / patch_size;
                        // printf("Tag2\n");
                        MapMatRowMajor weight_mat(operate_weight_chunk, out_channels_in_chunk, patch_size);
                        // printf("Weight mat\n");
                        
                        // printf(
                        //     "Weight rows %d, after redundant %d \n",
                        //     valid_weight_row_cnt, redundant_weight_cnt
                        // );

                        // printf("Weight\n");
                        // for (auto r=0; r<out_channels_in_chunk; r++){
                        //     for (auto c=0; c<patch_size; c++){
                        //         printf("%.0f ", weight_mat(r,c));
                        //     }
                        //     printf("\n");
                        // }
                        #ifdef PRINT_RUN_TIME_INFO
                            load_weight_time += get_elapsed_time(load_weight_start_time, get_time());
                        #endif
                        #ifdef PRINT_RUN_TIME_INFO
                            matrix_mul_start_time = get_time();
                        #endif
                        int tmp_output_row_start;
                        if (max_output_patches_per_chunk == max_matrix_mul_rows)
                            tmp_output_row_start = 0;
                        else
                            tmp_output_row_start = im2col_start_row_idx_in_chunk;
                        // printf(
                        //     "Block info: tmp_output_row_start %d, out_channel_start_idx %d, matrix_mul_rows %d, out_channels_in_chunk %d\n", 
                        //     tmp_output_row_start, out_channel_start_idx, matrix_mul_rows, out_channels_in_chunk
                        // );
                        auto output_block = tmp_output_mat.block(
                            tmp_output_row_start, out_channel_start_idx, matrix_mul_rows, out_channels_in_chunk
                        );
                        // printf("Block\n");
                        output_block.array() = im2col_mat * weight_mat.transpose();
                        #ifdef PRINT_RUN_TIME_INFO
                            matrix_mul_time += get_elapsed_time(matrix_mul_start_time, get_time());
                        #endif
                        // printf("Multiply\n");

                        out_channel_start_idx += out_channels_in_chunk;
                        free(operate_weight_chunk);
                        // printf("Free\n");
                    }, STORE_CHUNK_ELEM, weight_tensor->GetNumElem());
                    if (redundant_weight_cnt > 0){
                        printf("redundant_weight_cnt is not 0, is %d\n", redundant_weight_cnt);
                    }
                    assert (redundant_weight_cnt == 0);


                    // add bias
                    // printf("Bias: \n");
                    // for (auto i=0; i<output_c; i++)
                    //     printf("%f ", bias_mat(0,i));
                    // printf("\n");
                    for (auto i=0; i<matrix_mul_rows; i++){
                        auto output_block = tmp_output_mat.block(i, 0, 1, output_c);
                        output_block.array() = output_block + bias_mat;
                    }


                    // printf("Output: %d\n", matrix_mul_rows);
                    // for (int r=0; r<matrix_mul_rows; r++){
                    //     for (int c=0; c<1; c++){
                    //         printf("%f ", tmp_output_mat(r,c));
                    //     }
                        
                    // }
                    // printf("\n");

                    if ((output_row_idx_in_chunk + matrix_mul_rows <= output_patches_per_chunk) ||
                        im2col_last_row
                    ){
                        #ifdef PRINT_RUN_TIME_INFO
                            save_output_start_time = get_time();
                        #endif
                        #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                            printf(
                                "Directly copy, im2col_last_row %d, ", im2col_last_row);
                            printf(
                                "im2col_last_row %d, last_chunk %d, row_idx_in_chunk %d (%d-%d), output_row_idx_in_chunk %d, total output row %d-%d\n",
                                im2col_last_row, last_chunk, row_idx_in_chunk, end_row_idx_in_operate_chunk, stride_rows,
                                output_row_idx_in_chunk, 
                                output_row_idx_total+output_row_idx_in_chunk, output_row_idx_total+output_row_idx_in_chunk+matrix_mul_rows
                            );
                            // for (auto print_output_row_idx=0; print_output_row_idx<matrix_mul_rows; print_output_row_idx++){
                            //     if (output_row_idx_total+output_row_idx_in_chunk+print_output_row_idx == 2438){
                            //         DtypeForCpuOp* p_print = tmp_output_chunk+print_output_row_idx;
                            //         printf("Row %d output %.5f\n",output_row_idx_total+output_row_idx_in_chunk+print_output_row_idx, *p_print);
                            //     }
                            // }
                        #endif
                        // directly copy
                        DtypeForCpuOp* output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                        DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + (matrix_mul_rows * output_c);
                        // std::copy(tmp_output_chunk, tmp_output_chunk_end, output_start_idx_in_chunk);
                        memcpy(output_start_idx_in_chunk, tmp_output_chunk, matrix_mul_rows*output_c*sizeof(DtypeForCpuOp));
                        output_row_idx_in_chunk += matrix_mul_rows;

                        if (im2col_last_row || output_row_idx_in_chunk == output_patches_per_chunk){
                            #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                printf(
                                    "Last Store, last_row %d, output rows %d/%d\n", 
                                    im2col_last_row, output_row_idx_in_chunk, output_patches_per_chunk
                                );
                                
                            #endif
                            // printf("output_row_idx_total(%d)*output_c(%d) = %d\n", output_row_idx_total, output_c, output_row_idx_total*output_c);
                            // save output_chunk
                            chunk_manager.StoreChunk(
                                output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                                output_row_idx_in_chunk*output_c*sizeof(DtypeForCpuOp)
                            );
                            output_row_idx_total += output_row_idx_in_chunk;
                            output_row_idx_in_chunk = 0;
                        }
                        #ifdef PRINT_RUN_TIME_INFO
                            save_output_time += get_elapsed_time(save_output_start_time, get_time());
                        #endif
                    } else{
                        #ifdef PRINT_RUN_TIME_INFO
                            save_output_start_time = get_time();
                        #endif
                        #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                            printf(
                                "Middle copy, old output rows %d -> %d id %ld, ", 
                                output_row_idx_in_chunk, output_patches_per_chunk, 
                                output_tensor->GetChunkId(output_row_idx_total*output_c)
                            );
                        #endif
                        //copy part of tmp_output, store output_chunk, load new output_chunk, and copy the rest
                        DtypeForCpuOp* output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                        int chunk_left_rows = output_patches_per_chunk - output_row_idx_in_chunk;
                        DtypeForCpuOp* tmp_output_chunk_divide = tmp_output_chunk + (chunk_left_rows * output_c);
                        std::copy(tmp_output_chunk, tmp_output_chunk_divide, output_start_idx_in_chunk);
                        output_row_idx_in_chunk += chunk_left_rows;
                        chunk_manager.StoreChunk(
                            output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                            output_row_idx_in_chunk*output_c*sizeof(DtypeForCpuOp)
                        );

                        while (matrix_mul_rows-chunk_left_rows > output_patches_per_chunk){
                            DtypeForCpuOp* tmp_output_chunk_prev_divide = tmp_output_chunk_divide;
                            output_row_idx_total += output_patches_per_chunk;
                            output_start_idx_in_chunk = output_chunk;
                            tmp_output_chunk_divide += output_patches_per_chunk * output_c;
                            std::copy(tmp_output_chunk_prev_divide, tmp_output_chunk_divide, output_start_idx_in_chunk);
                            chunk_manager.StoreChunk(
                                output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                                output_patches_per_chunk*output_c*sizeof(DtypeForCpuOp)
                            );
                            chunk_left_rows += output_patches_per_chunk;
                            #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                printf(
                                    "middle integral output 0 -> %d id %ld, ",
                                    output_patches_per_chunk, output_tensor->GetChunkId(output_row_idx_total*output_c)
                                );
                            #endif
                        }

                        output_row_idx_total += output_patches_per_chunk;
                        output_row_idx_in_chunk = 0;
                        output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                        DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + (matrix_mul_rows * output_c);
                        std::copy(tmp_output_chunk_divide, tmp_output_chunk_end, output_start_idx_in_chunk);
                        output_row_idx_in_chunk += (matrix_mul_rows - chunk_left_rows);
                        #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                            printf(
                                "new output rows 0 -> %d id %ld\n",
                                output_row_idx_in_chunk, output_tensor->GetChunkId(output_row_idx_total*output_c)
                            );
                        #endif
                        #ifdef PRINT_RUN_TIME_INFO
                            save_output_time += get_elapsed_time(save_output_start_time, get_time());
                        #endif
                    }

                    // Reset im2col_row_idx_in_chunk
                    if (im2col_row_idx_in_chunk + matrix_mul_rows >= im2col_patches_per_chunk){
                        im2col_row_idx_total += im2col_row_idx_in_chunk;
                        im2col_row_idx_in_chunk = 0;
                    }
                    free(tmp_output_chunk);
                }
                // test_time += get_elapsed_time(test_start, get_time());
            }
            // Hard coding for stride = 1, 2
            if (row_idx_in_batch + stride >= input_height){
                input_row_idx_total += input_height - row_idx_in_batch;
                row_idx_in_chunk += input_height - row_idx_in_batch;
            }
            else{
                input_row_idx_total += stride_rows;
                row_idx_in_chunk += stride_rows;
            }
        
        }
        free(operate_input_chunk);
        
    }, input_elem_fetch_per_chunk, input_tensor->GetNumElem());
#ifdef PRINT_RUN_TIME_INFO
    printf(
        "Load input time %lf, im2col time %lf, matrix mul time %lf, save output time %lf, load weight time %lf, forward prepare time %lf\n", 
        load_input_time, im2col_construction_time, matrix_mul_time, save_output_time, load_weight_time, forward_prepare_time );
#endif

#ifdef PRINT_RUN_TIME_INFO
    total_time += get_elapsed_time(total_start, get_time());
    printf("Total time %lf\n", total_time);
    printf("Test time %lf\n", test_time);
#endif
}



