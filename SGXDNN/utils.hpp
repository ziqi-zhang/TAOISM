#ifndef SGXDNN_UTILS_HPP_
#define SGXDNN_UTILS_HPP_

#include <unsupported/Eigen/CXX11/Tensor>

#ifdef USE_SGX
#include "Enclave_t.h"
#else
#include <chrono>
#endif

extern bool TIMING;

extern Eigen::array<Eigen::IndexPair<int>, 1> MATRIX_PRODUCT;
extern Eigen::array<Eigen::IndexPair<int>, 1> INNER_PRODUCT;
extern Eigen::array<int, 2> TRANSPOSE2D;

typedef Eigen::array<long, 1> array1d;
typedef Eigen::array<long, 2> array2d;
typedef Eigen::array<long, 3> array3d;
typedef Eigen::array<long, 4> array4d;

typedef long long int64;

template <typename T, int NDIMS = 1>
using Tensor = typename Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>;

template <typename T>
using Matrix = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using MatrixMap = typename Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using VectorMap = typename Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

#ifndef USE_SGX
typedef std::chrono::time_point<std::chrono::high_resolution_clock> sgx_time_t;
std::chrono::time_point<std::chrono::high_resolution_clock> get_time();

std::chrono::time_point<std::chrono::high_resolution_clock> get_time_force();

double get_elapsed_time(std::chrono::time_point<std::chrono::high_resolution_clock> start,
						std::chrono::time_point<std::chrono::high_resolution_clock> end);
#else
typedef double sgx_time_t;
double get_time();

double get_time_force();

double get_elapsed_time(double start, double end);
#endif

#endif /* SGXDNN_UTILS_HPP_ */
