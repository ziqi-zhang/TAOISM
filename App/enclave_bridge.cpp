#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <memory>
#include <omp.h>
#include <unordered_map>
#include <cstdlib>
#include <malloc.h>
#include <time.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "sgx_tseal.h"
#include "sgx_urts.h"
#include "Enclave_u.h"

#include "common_with_enclaves.h"
#include "common_utils.cpp"
#include "crypto_common.h"
#include "ThreadPool.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define TOKEN_FILENAME   "enclave.token"
#define ENCLAVE_FILENAME "enclave.signed.so"

using namespace std::chrono;
using std::shared_ptr;
using std::make_shared;

#define eidT unsigned long int eid

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

void print_error_message(sgx_status_t ret);

DimsT CreateDims(int dim0_, int dim1_, int dim2_, int dim3_) {
    DimsT res = {dim0_, dim1_, dim2_, dim3_};
    return res;
}

inline int CeilDiv(int x, int y) {
    return (x + y - 1)/y;
}

inline int FloorDiv(int x, int y) {
    return x / y;
}

// Eigen::Map<Eigen::Matrix<DtypeForCpuOp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
void ModP(MapMatRowMajor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<DtypeForCpuOp>(1) / PrimeLimit;
    MatRowMajor mat = m;
    // m = m - (mat * invPLimit) * PLimit;
    m.array() = m.array() - (mat * invPLimit).array().floor() * PLimit;
}

void ModP(MapEigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<DtypeForCpuOp>(1) / PrimeLimit;
    m = m - (m * invPLimit).floor() * PLimit;
}

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

thread_local std::chrono::time_point<std::chrono::high_resolution_clock> start;

void ocall_start_clock()
{
	start = std::chrono::high_resolution_clock::now();
}

void ocall_end_clock(const char * str)
{
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf(str, elapsed.count());
}

double ocall_get_time()
{
    auto now = std::chrono::high_resolution_clock::now();
	return time_point_cast<microseconds>(now).time_since_epoch().count();
}

void* ocall_allocate_mem(int num_byte) {
    void* res = (void*)malloc(num_byte);
    return res;
}


class UntrustedChunkManager {
public:
    static UntrustedChunkManager& getInstance() {
        static UntrustedChunkManager instance; // Guaranteed to be destroyed.
        return instance;
    }
    UntrustedChunkManager(UntrustedChunkManager const&) = delete;
    void operator=(UntrustedChunkManager const&) = delete;

    void StoreChunk(IdT id, void* src_buf, int num_byte) {
        void* dst_buf;
        bool is_diff_size = false;
        auto it = IdAddress.begin();
        auto end = IdAddress.end();
        {
            std::unique_lock <std::mutex> lock(address_mutex);
            it = IdAddress.find(id);
            end = IdAddress.end();
        }
        if (it == end) {
            dst_buf = (void*)malloc(num_byte);
            {
                std::unique_lock<std::mutex> lock(address_mutex);
                IdAddress[id] = dst_buf;
                IdSize[id] = num_byte;
            }
        } else {
            std::unique_lock<std::mutex> lock(address_mutex);
            dst_buf = IdAddress[id];
            if (IdSize[id] != num_byte) {
                is_diff_size = true;
            }
        }
        if (is_diff_size) {
            throw std::invalid_argument("A id has assigned with multiple size.");
        }
        memcpy(dst_buf, src_buf, num_byte);
    }

    void GetChunk(IdT id, void* dst_buf, int num_byte) {
        auto it = IdAddress.begin();
        auto end = IdAddress.end();
        void* src_buf;
        {
            std::unique_lock <std::mutex> lock(address_mutex);
            auto it = IdAddress.find(id);
            auto end = IdAddress.end();
        }
        if (it == end) {
            throw std::invalid_argument("Id doesnt exist");
        }
        {
            std::unique_lock <std::mutex> lock(address_mutex);
            src_buf = IdAddress[id];
        }
        memcpy(dst_buf, src_buf, num_byte);
    }

private:
    UntrustedChunkManager() {}
    std::unordered_map<IdT, void*> IdAddress;
    std::unordered_map<IdT, int> IdSize;
    std::mutex address_mutex;
};

ThreadPool thread_pool(THREAD_POOL_SIZE);
std::vector< std::future<TaskIdT> > task_results;
TaskIdT task_indexor = -1;
TaskIdT task_counter = 0;

const int max_async_task = 1000;

template<typename Func>
TaskIdT add_async_task(Func f) {
//    task_results.emplace_back(thread_pool.enqueue(f));
//    task_indexor++;
//    return task_indexor;

    if (task_results.size() < max_async_task) {
        task_results.resize(max_async_task + 5);
    }
    TaskIdT task_id = (task_indexor + 1) % max_async_task;
//    task_results.emplace(task_results.begin() + task_id, thread_pool.enqueue(f));
    task_results[task_id] = std::future<TaskIdT>(thread_pool.enqueue(f));
    task_indexor = task_id;
    return task_id;
}

extern "C"
{

/*
 * Initialize the enclave
 */
    uint64_t initialize_enclave(void)
    {

        std::cout << "Initializing Enclave..." << std::endl;
        printf("My enclave init");

        sgx_enclave_id_t eid = 0;
        sgx_launch_token_t token = {0};
        sgx_status_t ret = SGX_ERROR_UNEXPECTED;
        int updated = 0;

        /* call sgx_create_enclave to initialize an enclave instance */
        /* Debug Support: set 2nd parameter to 1 */
        ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }

        std::cout << "Enclave id: " << eid << std::endl;

        return eid;
    }

    /*
     * Destroy the enclave
     */
    void destroy_enclave(uint64_t eid)
    {
        std::cout << "Destroying Enclave with id: " << eid << std::endl;
        sgx_destroy_enclave(eid);
    }


    int CalcEncNeededInByte(const uint32_t txt_encrypt_size) {
        return CalcEncDataSize(0, txt_encrypt_size * sizeof(DtypeForCpuOp));
    }

    void InitTensor(EidT eid, IdT TenId, int dim0, int dim1, int dim2, int dim3) {
        DimsT Dims = CreateDims(dim0, dim1, dim2, dim3);
        sgx_status_t ret = ecall_init_tensor(eid, TenId, (void*) &Dims);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void SetTen(EidT eid, IdT TenId, DtypeForCpuOp* Arr) {
        sgx_status_t ret = ecall_set_ten(eid, TenId, (void*) Arr);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void GetTen(EidT eid, IdT TenId, DtypeForCpuOp* Arr) {
        sgx_status_t ret = ecall_get_ten(eid, TenId, (void*) Arr);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void SetSeed(EidT eid, IdT TenId, uint64_t RawSeed) {
        sgx_status_t ret = ecall_set_seed(eid, TenId, RawSeed);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void GetRandom(EidT eid, IdT TenId, DtypeForCpuOp* Arr, uint64_t RawSeed) {
        sgx_status_t ret = ecall_get_random(eid, TenId, (void*) Arr, RawSeed);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }


    void AddFromCpu(EidT eid, void* inputArr, IdT dstId) {
        sgx_status_t ret = ecall_add_from_cpu(eid, inputArr, dstId);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    TaskIdT AsyncAddFromCpu(EidT eid, void* inputArr, IdT dstId) {
        return add_async_task( [=, task_counter] {
            AddFromCpu(eid, inputArr, dstId);
            return task_counter;
        });
    }

    void AesEncryptTensor(DtypeForCpuOp* SrcBuf, uint32_t NumElem, uint8_t* DstBuf) {
        SgxEncT* Dst = (SgxEncT*) DstBuf;
        uint32_t NumByte = NumElem * sizeof(DtypeForCpuOp);
		encrypt((uint8_t *) SrcBuf, 
		        NumByte, 
		        (uint8_t *) (&(Dst->payload)), 
		        (sgx_aes_gcm_128bit_iv_t *)(&(Dst->reserved)), 
		        (sgx_aes_gcm_128bit_tag_t *)(&(Dst->payload_tag)));
    }

    void AesDecryptTensor(uint8_t* SrcBuf, uint32_t NumElem, DtypeForCpuOp* DstBuf) {
        SgxEncT* Src = (SgxEncT*) SrcBuf;
        uint32_t NumByte = NumElem * sizeof(DtypeForCpuOp);
        float* blind = (float*)malloc(NumByte);
		decrypt((uint8_t *) (&(Src->payload)),
		        NumByte, 
		        (uint8_t *) DstBuf, 
		        (sgx_aes_gcm_128bit_iv_t  *)(&(Src->reserved)), 
		        (sgx_aes_gcm_128bit_tag_t *)(&(Src->payload_tag)),
		        (uint8_t *) blind);
		free(blind);
    }

    int GetTaskStatus(TaskIdT task_id) {
        auto && task_future = task_results[task_id];
        std::future_status task_status = task_future.wait_for(std::chrono::microseconds::zero());
        if (task_status == std::future_status::ready) {
//           std::cout << "Task is ready: " << task_future.get() << std::endl;
           return 1;
        } else {
            return 0;
        }
    }

    TaskIdT AsyncGetRandom(EidT eid, IdT TenId, DtypeForCpuOp* Arr, uint64_t RawSeed) {
        return add_async_task( [=, task_counter] {
            GetRandom(eid, TenId, Arr, RawSeed);
            return task_counter;
        });
    }

    void SgdUpdate(EidT eid, IdT paramId, IdT gradId, IdT momentumId,
            DtypeForCpuOp lr, DtypeForCpuOp momentum, DtypeForCpuOp weight_decay,
            DtypeForCpuOp dampening, bool nesterov, bool first_momentum) {
        sgx_status_t ret = ecall_sgd_update(eid, paramId, gradId, momentumId,
                lr, momentum, weight_decay, dampening, nesterov, first_momentum);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    TaskIdT AsyncSgdUpdate(EidT eid, IdT paramId, IdT gradId, IdT momentumId,
                   DtypeForCpuOp lr, DtypeForCpuOp momentum, DtypeForCpuOp weight_decay,
                   DtypeForCpuOp dampening, bool nesterov, bool first_momentum) {
        return add_async_task( [=, task_counter] {
            SgdUpdate(eid, paramId, gradId, momentumId, lr, momentum, weight_decay, dampening, nesterov, first_momentum);
            return task_counter;
        });
    }

    void StochasticQuantize(EidT eid, IdT src_id, IdT dst_id, uint64_t q_tag) {
        sgx_status_t ret = ecall_stochastic_quantize(eid, src_id, dst_id, q_tag);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    TaskIdT AsyncStochasticQuantize(EidT eid, IdT src_id, IdT dst_id, uint64_t q_tag) {
        return add_async_task( [=, task_counter] {
            StochasticQuantize(eid, src_id, dst_id, q_tag);
            return task_counter;
        });
    }

    
    void ReLUfunction(EidT eid, IdT TenIdin, IdT TenIdout, uint64_t size){
        sgx_status_t ret = ecall_relu(eid, TenIdin, TenIdout, size);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void QuantReLUfunction(EidT eid, IdT TenIdin, IdT TenIdout, uint64_t size, float scale, float v_min, uint8_t zero){
        sgx_status_t ret = ecall_quant_relu(eid, TenIdin, TenIdout, size, scale, v_min, zero);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void ReLUbackward(EidT eid, IdT TenIdout, IdT TenIddout, IdT TenIddin, uint64_t size){
        sgx_status_t ret = ecall_reluback(eid, TenIdout, TenIddout, TenIddin, size);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }



    void AsyncTask(EidT eid,
            IdT TenId1, DtypeForCpuOp* Arr1, uint64_t size1, uint64_t RawSeed1,
            IdT TenId2, DtypeForCpuOp* Arr2, uint64_t size2, uint64_t RawSeed2,
            IdT TenId3, DtypeForCpuOp* Arr3, uint64_t size3, uint64_t RawSeed3,
            IdT TenId4, DtypeForCpuOp* Arr4, uint64_t size4, uint64_t RawSeed4
            ) {
        // ThreadPool pool(4);
        auto& pool = thread_pool;
        std::vector< std::future<int> > results;

        std::cout << "AsyncTask" << std::endl;

        // results.emplace_back( pool.enqueue([&] {
        //     sgx_status_t ret = ecall_get_share(eid, TenId1, (void*) Arr1, RawSeed1);
        //     if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
        //     return 1;
        // }));

        // results.emplace_back( pool.enqueue([&] {
        //     sgx_status_t ret = ecall_get_share(eid, TenId2, (void*) Arr2, RawSeed2);
        //     if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
        //     return 2;
        // }));

        // results.emplace_back( pool.enqueue([&] {
        //     sgx_status_t ret = ecall_get_share(eid, TenId3, (void*) Arr3, RawSeed3);
        //     if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
        //     return 3;
        // }));

        // results.emplace_back( pool.enqueue([&] {
        //     sgx_status_t ret = ecall_get_share(eid, TenId4, (void*) Arr4, RawSeed4);
        //     if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
        //     return 4;
        // }));

        for(auto && result: results) std::cout << result.get() << std::endl;
        std::cout << std::endl;
    }

    void InitMaxpool(EidT eid, IdT FunId, IdT TenIdin_trans, IdT TenIdout_trans){
        sgx_status_t ret = ecall_initmaxpool(eid, FunId, TenIdin_trans, TenIdout_trans);      
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void Maxpoolfunction(EidT eid, IdT FunId, IdT TenIdin, IdT TenIdout, uint32_t batch, uint32_t channel, uint32_t input_height, uint32_t input_width, uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride,uint32_t col_stride, uint32_t row_pad, uint32_t col_pad){
        sgx_status_t ret = ecall_maxpool(eid, FunId, TenIdin, TenIdout, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride, row_pad, col_pad);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void Maxpoolbackwardfunction(EidT eid, IdT FunId, IdT TenIddout, IdT TenIddin, uint32_t batch, uint32_t channel, uint32_t input_height, uint32_t input_width, uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride,uint32_t col_stride){
        sgx_status_t ret = ecall_maxpoolback(eid, FunId, TenIddout, TenIddin, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void InitBatchnorm(
            EidT eid,
            IdT FunId,
            IdT input, IdT output, IdT gamma, IdT beta,
            IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
            IdT mu,
            uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
            int affine_, int is_cumulative_, float momentum_, float epsilon_) {

        sgx_status_t ret = ecall_init_batchnorm(
                eid,
                FunId,
                input, output, gamma, beta,
                run_mean, run_var, cur_mean, cur_var,
                mu,
                batch_, channel_, height_, width_,
                affine_, is_cumulative_, momentum_, epsilon_);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void BatchnormForward(EidT eid, uint64_t FunId, int Training) {
        sgx_status_t ret = ecall_batchnorm_forward(eid, FunId, Training);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void BatchnormBackward(EidT eid, uint64_t FunId) {
        sgx_status_t ret = ecall_batchnorm_backward(eid, FunId);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void InitSGXLinear(
            EidT eid,
            IdT FunId,
            IdT input, IdT output, IdT weight, IdT bias,
            // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
            uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

        sgx_status_t ret = ecall_init_sgx_linear(
                eid,
                FunId,
                input, output, weight, bias,
                // der_input, der_output, der_weight, der_bias,
                batch_, input_size_, output_size_);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void SGXLinearForward(EidT eid, uint64_t FunId) {
        sgx_status_t ret = ecall_sgx_linear_forward(eid, FunId);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void InitSGXConv(
            EidT eid,
            IdT FunId,
            IdT input, IdT output, IdT weight, IdT bias,
            // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
            uint32_t batch_, uint32_t input_h, uint32_t input_w, uint32_t input_c, 
            uint32_t output_h, uint32_t output_w, uint32_t output_c,
            uint32_t kernel, uint32_t padding, uint32_t stride
        ) {

        sgx_status_t ret = ecall_init_sgx_conv(
                eid,
                FunId,
                input, output, weight, bias,
                // der_input, der_output, der_weight, der_bias,
                batch_, input_h, input_w, input_c, 
                output_h, output_w, output_c,
                kernel, padding, stride);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }

    void SGXConvForward(EidT eid, uint64_t FunId) {
        sgx_status_t ret = ecall_sgx_conv_forward(eid, FunId);
        if (ret != SGX_SUCCESS) { print_error_message(ret); throw ret; }
    }
}

/* Application entry */
int main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    try {
        sgx_enclave_id_t eid = initialize_enclave();

        std::cout << "Enclave id: " << eid << std::endl;

// 		const unsigned int filter_sizes[] = {3*3*3*64, 64,
// 											3*3*64*64, 64,
// 											3*3*64*128, 128,
// 											3*3*128*128, 128,
// 											3*3*128*256, 256,
// 											3*3*256*256, 256,
// 											3*3*256*256, 256,
// 											3*3*256*512, 512,
// 											3*3*512*512, 512,
// 											3*3*512*512, 512,
// 											3*3*512*512, 512,
// 											3*3*512*512, 512,
// 											3*3*512*512, 512,
// 											7 * 7 * 512 * 4096, 4096,
// 											4096 * 4096, 4096,
// 											4096 * 1000, 1000};
//
// 		float** filters = (float**) malloc(2*16*sizeof(float*));
//         for (int i=0; i<2*16; i++) {
// 			filters[i] = (float*) malloc(filter_sizes[i] * sizeof(float));
// 		}
//
// 		const unsigned int output_sizes[] = {224*224*64,
//                                              224*224*64,
//                                              112*112*128,
//                                              112*112*128,
//                                              56*56*256,
//                                              56*56*256,
//                                              56*56*256,
//                                              28*28*512,
//                                              28*28*512,
//                                              28*28*512,
//                                              14*14*512,
//                                              14*14*512,
//                                              14*14*512,
// 											 4096,
// 											 4096,
// 											 1000};
//
// 		float** extras = (float**) malloc(16*sizeof(float*));
// 		for (int i=0; i<16; i++) {
// 			extras[i] = (float*) malloc(output_sizes[i] * sizeof(float));
// 		}
//
//         float* img = (float*) malloc(224 * 224 * 3 * sizeof(float));
//         float* output = (float*) malloc(1000 * sizeof(float));
// 		printf("filters initalized\n");
//
// 		std::ifstream t("App/vgg16.json");
// 		std::stringstream buffer;
// 		buffer << t.rdbuf();
// 		std::cout << buffer.str() << std::endl;
// 		printf("Loading model...\n");
// 		load_model_float(eid, (char*)buffer.str().c_str(), filters);
// 		printf("Model loaded!\n");
//
// 		for(int i=0; i<4; i++) {
// 			auto start = std::chrono::high_resolution_clock::now();
// 			//predict_float(eid, img, output, 1);
// 			predict_verify_float(eid, img, output, extras, 1);
// 			auto finish = std::chrono::high_resolution_clock::now();
// 			std::chrono::duration<double> elapsed = finish - start;
// 			printf("predict required %4.2f sec\n", elapsed.count());
// 		}
//         printf("Enter a character to destroy enclave ...\n");
//         getchar();
//
//         // Destroy the enclave
        sgx_destroy_enclave(eid);

        printf("Info: Enclave Launcher successfully returned.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return 0;
    }
    catch (int e)
    {
        printf("Info: Enclave Launch failed!.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }
}
