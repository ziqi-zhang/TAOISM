#ifdef USE_SGX
#include "Enclave.h"
#endif

#include "batchnorm.hpp"
#include "chunk_manager.hpp"

using namespace std;


BatchnormBuffer::BatchnormBuffer(IdT FunId_) : FunId(FunId_) {
    NumBatchesTrackedArr = 0;
    BackwardState = false;
}

void BatchnormBuffer::init(
        IdT input, IdT output, IdT gamma, IdT beta,
        IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
        IdT mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    input_tensor = GetTenById(input);
    output_tensor = GetTenById(output);
    mu_tensor = GetTenById(mu);

    // size = num_channel * sizeof(byte)
    gamma_tensor = GetTenById(gamma);
    beta_tensor = GetTenById(beta);
    run_mean_tensor = GetTenById(run_mean);
    run_var_tensor = GetTenById(run_var);
    cur_mean_tensor = GetTenById(cur_mean);
    cur_var_tensor = GetTenById(cur_var);
    

    batch = batch_;
    channel = channel_;
    height = height_;
    width = width_;
    Affine = affine_;
    momentum = momentum_;
    epsilon = epsilon_;
    is_cumulative = is_cumulative_;

    num_elem_per_sample = channel * height * width;
    // BCHW
    num_elem_in_channel = height * width;
    total_n = height * width * batch;
    
    default_num_batches_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / num_elem_per_sample;
    default_num_rows_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / num_elem_in_channel;
    // printf("Default batches per chunk %d, rows per chunk %d\n", default_num_batches_per_chunk, default_num_rows_per_chunk);
    if (STORE_CHUNK_ELEM % num_elem_in_channel != 0)  {
        printf(
            "STORE_CHUNK_ELEM %% num_elem_in_channel != 0, STORE_CHUNK_ELEM %d, num_elem_in_channel %d, left %d\n", 
            STORE_CHUNK_ELEM, num_elem_in_channel, STORE_CHUNK_ELEM % num_elem_in_channel
        );
        return;
    }
}

DtypeForCpuOp BatchnormBuffer::get_fraction_bag(int num_elem_in_chunk) {
    int batch_in_chunk = num_elem_in_chunk / num_rows;
    return ((DtypeForCpuOp) batch_in_chunk / batch);
}

int BatchnormBuffer::get_num_batches_per_chunk(int num_elem_in_chunk) {
    return num_elem_in_chunk / num_rows;
}


void BatchnormBuffer::forward(int training) {
    Training = training;

    vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>> small_chunks;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *data_chunk, *mu_chunk;
    ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
    ChunkGuard<DtypeForCpuOp> mu_guard(StoreChunkPool::GetChunkPool(), mu_chunk);

    // EigenMatrixMap data_mat(data_chunk, num_rows, default_num_batches_per_chunk);
    // EigenMatrixMap mu_mat(mu_chunk, num_rows, default_num_batches_per_chunk);

    DtypeForCpuOp *gamma_chunk = get_small_chunk(gamma_tensor, small_chunks);
    DtypeForCpuOp *beta_chunk = get_small_chunk(beta_tensor, small_chunks);
    DtypeForCpuOp *run_mean_chunk = get_small_chunk(run_mean_tensor, small_chunks);
    DtypeForCpuOp *run_var_chunk = get_small_chunk(run_var_tensor, small_chunks);
    DtypeForCpuOp *cur_mean_chunk = get_small_chunk(cur_mean_tensor, small_chunks);
    DtypeForCpuOp *cur_var_chunk = get_small_chunk(cur_var_tensor, small_chunks);

    int total_input_row_idx = 0;

    if (training) {
        // NumBatchesTrackedArr += 1;
        // const DtypeForCpuOp chosen_momentum = (is_cumulative) ? (1 / (DtypeForCpuOp) NumBatchesTrackedArr) : momentum;

        // fill(cur_mean_chunk, cur_mean_chunk + channel, 0);
        // fill(cur_var_chunk, cur_var_chunk + channel, epsilon);

        // run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
        //     int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
        //     int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
        //     chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

        //     for(uint32_t i = 0; i < channel; i++) {
        //         auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
        //         cur_mean_chunk[i] += data_block.mean() * get_fraction_bag(num_elem_in_store_chunk);
        //     }
        // }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        // run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
        //     int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
        //     int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
        //     chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

        //     for(uint32_t i = 0; i < channel; i++) {
        //         auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
        //         auto mu_block = mu_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
        //         mu_block = data_block.array() - cur_mean_chunk[i];
        //         cur_var_chunk[i] += (mu_block).cwiseProduct(mu_block).mean() * get_fraction_bag(num_elem_in_store_chunk);
        //     }

        //     chunk_manager.StoreChunk(mu_tensor->GetChunkId(start_store_chunk), mu_chunk, chunk_size_in_byte);
        // }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        // run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
        //     int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
        //     int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
        //     chunk_manager.GetChunk(mu_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

        //     for(uint32_t i = 0; i < channel; i++) {
        //         auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
        //         if (Affine) {
        //             data_block = (data_block.array() / sqrt(cur_var_chunk[i])) * gamma_chunk[i] + beta_chunk[i];
        //         } else {
        //             data_block = data_block / sqrt(cur_var_chunk[i]);
        //         }
        //     }

        //     chunk_manager.StoreChunk(output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
        // }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        // for (int i = 0; i < channel; i++) {
        //     run_mean_chunk[i] = (cur_mean_chunk[i] - run_mean_chunk[i]) * chosen_momentum + run_mean_chunk[i];
        //     run_var_chunk[i] = (cur_var_chunk[i] - run_var_chunk[i]) * chosen_momentum + run_var_chunk[i];
        // }
    } else {
        run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
            int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            int num_rows_in_chunk = num_elem_in_store_chunk / num_elem_in_channel;
            MapMatRowMajor data_mat(data_chunk, num_rows_in_chunk, num_elem_in_channel);

            // printf("data_chunk\n");
            // for (auto i=0; i<num_elem_in_store_chunk; i++){
            //     printf("%.2f, ", data_chunk[i]);
            // }
            // printf("\n");
            
            // printf("data_mat Mat:\n");
            // for (auto r=0; r<num_rows_in_chunk; r++){
            //     for (auto c=0; c<num_elem_in_channel; c++)
            //         printf("%.2f ", data_mat(r,c));
            //     printf("\n");
            // }

            for (auto row_idx_in_chunk=0; row_idx_in_chunk<num_rows_in_chunk; row_idx_in_chunk++){
                auto channel_idx = total_input_row_idx % channel;
                auto data_block = data_mat.block(row_idx_in_chunk, 0, 1, num_elem_in_channel);
                data_block = data_block.array() - run_mean_chunk[channel_idx];
                if (Affine) {
                    data_block = (data_block.array() / sqrt(run_var_chunk[channel_idx]+1e-5)) * gamma_chunk[channel_idx] + beta_chunk[channel_idx];
                    // running_var here is actually 1 / sqrt(running_var)
                    // printf("var %f\n", run_var_chunk[i]);
                    // data_block = (data_block.array() * run_var_chunk[i]) * gamma_chunk[i] + beta_chunk[i];
                } else {
                    printf("Check var first!!!!!!==================\n");
                }
                total_input_row_idx++;
            }

            // printf("Bias: ");
            // for (auto i=0; i<channel; i++)
            //     printf("%f ", beta_chunk[i]);
            // printf("\n");
            // printf("RunVar: ");
            // for (auto i=0; i<channel; i++)
            //     printf("%f ", run_var_chunk[i]);
            // printf("\n");

            // for(uint32_t i = 0; i < channel; i++) {
            //     auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
            //     data_block = data_block.array() - run_mean_chunk[i];
            //     if (Affine) {
            //         data_block = (data_block.array() / sqrt(run_var_chunk[i]+1e-5)) * gamma_chunk[i] + beta_chunk[i];
            //         // running_var here is actually 1 / sqrt(running_var)
            //         // printf("var %f\n", run_var_chunk[i]);
            //         // data_block = (data_block.array() * run_var_chunk[i]) * gamma_chunk[i] + beta_chunk[i];
            //     } else {
            //         printf("Check var first!!!!!!==================\n");
            //         data_block = data_block / sqrt(run_var_chunk[i]);
            //     }
            // }

            chunk_manager.StoreChunk(output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
        }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());
    }

    store_small_chunks(small_chunks);

    BackwardState = true;
}


void BatchnormBuffer::backward() {}

