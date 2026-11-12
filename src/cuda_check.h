// Error-checking helpers for the CUDA runtime and cuBLAS.
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err_));                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",          \
                         __FILE__, __LINE__, static_cast<int>(st_));            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
