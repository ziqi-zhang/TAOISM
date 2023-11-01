#ifndef XOSHIRO_H
#define XOSHIRO_H

#ifdef USE_SGX
#include "Enclave.h"
#endif


#include <cstdint>
#include <memory>
#include <unordered_map>

using namespace std;
// using std::shared_ptr;

static inline float uint32_to_float(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = UINT32_C(0x7F) << 23 | x >> 9  };
    return u.d - 1.0f;
}

static inline float float_to_uniform(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = (((UINT32_C(0x7F) << 23) | x) << 2) >> 2 };
    return u.d - 1.0f;
}

// http://prng.di.unimi.it/
class Xoshiro256 {
public:
    Xoshiro256() {}
    Xoshiro256(uint64_t raw_seed);

    void set_seed(uint64_t raw_seed);

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void);

    void rand_like(float* arr, uint64_t n_elem);

    uint64_t s[4] = {};
};

class Xoshiro128 {
public:
    Xoshiro128() {}
    Xoshiro128(uint64_t raw_seed);

    void set_seed(uint64_t raw_seed);

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void);

    uint64_t s[2] = {};
};

extern unordered_map<uint64_t, shared_ptr<Xoshiro256>> fast_rngs;
//unordered_map<uint64_t, shared_ptr<Xoshiro128>> fast_rngs;

shared_ptr<Xoshiro256> get_fast_rng(uint64_t tag);




#endif