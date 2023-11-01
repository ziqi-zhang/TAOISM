
#include "xoshiro.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

unordered_map<uint64_t, shared_ptr<Xoshiro256>> fast_rngs;

// static inline uint64_t Xoshiro256::rotl(const uint64_t x, int k) {
//     return (x << k) | (x >> (64 - k));
// }
Xoshiro256::Xoshiro256(uint64_t raw_seed) {
    set_seed(raw_seed);
}

void Xoshiro256::set_seed(uint64_t raw_seed) {
    s[0] = raw_seed;
}

uint64_t Xoshiro256::next(void) {
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}

void Xoshiro256::rand_like(float* arr, uint64_t n_elem) {
    if (n_elem % 2 != 0) {
        printf("n_elem has to be even.\n");
        throw string("n_elem has to be even.");
    }
    for (int i = 0; i < n_elem; i+=2) {
        const uint64_t rnd = next();
        const uint32_t b = rnd & ((((uint64_t) 1) << 32) - 1);
        const uint32_t a = rnd >> 32;
        arr[i]   = uint32_to_float(a);
        arr[i+1] = uint32_to_float(b);
    }
}


// // http://prng.di.unimi.it/
// class Xoshiro256 {
// public:
//     Xoshiro256() {}
//     Xoshiro256(uint64_t raw_seed) {
//         set_seed(raw_seed);
//     }

//     void set_seed(uint64_t raw_seed) {
//         s[0] = raw_seed;
//     }

//     static inline uint64_t rotl(const uint64_t x, int k) {
//         return (x << k) | (x >> (64 - k));
//     }

//     uint64_t next(void) {
//         const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

//         const uint64_t t = s[1] << 17;

//         s[2] ^= s[0];
//         s[3] ^= s[1];
//         s[1] ^= s[2];
//         s[0] ^= s[3];

//         s[2] ^= t;

//         s[3] = rotl(s[3], 45);

//         return result;
//     }

//     void rand_like(float* arr, uint64_t n_elem) {
//         if (n_elem % 2 != 0) {
//             printf("n_elem has to be even.\n");
//             throw string("n_elem has to be even.");
//         }
//         for (int i = 0; i < n_elem; i+=2) {
//             const uint64_t rnd = next();
//             const uint32_t b = rnd & ((((uint64_t) 1) << 32) - 1);
//             const uint32_t a = rnd >> 32;
//             arr[i]   = uint32_to_float(a);
//             arr[i+1] = uint32_to_float(b);
//         }
//     }

//     uint64_t s[4] = {};
// };

Xoshiro128::Xoshiro128(uint64_t raw_seed) {
    set_seed(raw_seed);
}

void Xoshiro128::set_seed(uint64_t raw_seed) {
    s[0] = raw_seed;
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t Xoshiro128::next(void) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = rotl(s0 + s1, 17) + s0;

    s1 ^= s0;
    s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
    s[1] = rotl(s1, 28); // c

    return result;
}


// class Xoshiro128 {
// public:
//     Xoshiro128() {}
//     Xoshiro128(uint64_t raw_seed) {
//         set_seed(raw_seed);
//     }

//     void set_seed(uint64_t raw_seed) {
//         s[0] = raw_seed;
//     }

//     static inline uint64_t rotl(const uint64_t x, int k) {
//         return (x << k) | (x >> (64 - k));
//     }

//     uint64_t next(void) {
//         const uint64_t s0 = s[0];
//         uint64_t s1 = s[1];
//         const uint64_t result = rotl(s0 + s1, 17) + s0;

//         s1 ^= s0;
//         s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
//         s[1] = rotl(s1, 28); // c

//         return result;
//     }

//     uint64_t s[2] = {};
// };

// unordered_map<uint64_t, shared_ptr<Xoshiro256>> fast_rngs;
// //unordered_map<uint64_t, shared_ptr<Xoshiro128>> fast_rngs;

shared_ptr<Xoshiro256> get_fast_rng(uint64_t tag) {
    if (fast_rngs.find(tag) == fast_rngs.end()) {
        fast_rngs[tag] = make_shared<Xoshiro256>(tag);
    }
    return fast_rngs[tag];
}