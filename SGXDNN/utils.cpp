
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.hpp"

#ifdef USE_SGX
#include "Enclave_t.h"
#else
#include <chrono>
#endif

bool TIMING = true;

Eigen::array<Eigen::IndexPair<int>, 1> MATRIX_PRODUCT = { Eigen::IndexPair<int>(1, 0) };
Eigen::array<Eigen::IndexPair<int>, 1> INNER_PRODUCT = { Eigen::IndexPair<int>(0, 0) };
Eigen::array<int, 2> TRANSPOSE2D = {{1, 0}};

#ifndef USE_SGX
// typedef std::chrono::time_point<std::chrono::high_resolution_clock> sgx_time_t;
std::chrono::time_point<std::chrono::high_resolution_clock> get_time() {
	if (TIMING) {
        return std::chrono::high_resolution_clock::now();
	}
	return std::chrono::time_point<std::chrono::high_resolution_clock>();
}

std::chrono::time_point<std::chrono::high_resolution_clock> get_time_force() {
	return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(std::chrono::time_point<std::chrono::high_resolution_clock> start,
						std::chrono::time_point<std::chrono::high_resolution_clock> end) {

	std::chrono::duration<double> elapsed = end - start;
	return elapsed.count();
}
#else
// typedef double sgx_time_t;
double get_time() {
	if (TIMING) {
		double res;
		ocall_get_time(&res);
		return res;
	}
	return 0.0;
}

double get_time_force() {
	double res;
	ocall_get_time(&res);
	return res;
}

double get_elapsed_time(double start, double end) {
	return (end - start) / (1000.0 * 1000.0);
}
#endif

