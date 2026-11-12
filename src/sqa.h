// Simulated quantum annealing (SQA) for fully-connected Ising models on the GPU,
// using the hierarchical update (HU) scheme and Tensor Cores (via cuBLAS).
#pragma once

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "graph.h"

struct SqaParams {
    int trotters = 16;            // M: number of Trotter slices (replicas)
    int block = 128;              // M_2: HU block size (32, 64, ..., 1024)
    int size = 0;                 // N: simulated spins; 0 = num_nodes rounded up to `block`
    int steps = 100;              // Monte Carlo steps per run
    double gamma0 = 8.0;          // initial transverse field Gamma_0
    double beta0 = 1.0 / 16.0;    // inverse temperature at the first step
    double beta1 = 16.0;          // inverse temperature reached after `steps` steps
    bool tensor_cores = true;     // let cuBLAS use Tensor Cores for the local-field updates
};

struct RunResult {
    double seconds = 0.0;         // wall time of the annealing loop
    double best_cut = 0.0;        // best cut among the M Trotter replicas
    int best_trotter = 0;
    double field_error = -1.0;    // max |h - J*sigma| after annealing (only when verified)
};

class SqaSolver {
public:
    SqaSolver(const Graph& graph, const SqaParams& params);
    ~SqaSolver();
    SqaSolver(const SqaSolver&) = delete;
    SqaSolver& operator=(const SqaSolver&) = delete;

    // Runs one independent annealing from a random initial state. When `verify`
    // is set, the incrementally maintained local fields are compared against a
    // full recomputation at the end of the run (see RunResult::field_error).
    RunResult run(uint64_t seed, bool verify);

    int size() const { return n_; }

private:
    void init_couplings();
    void compute_fields(float* out);  // out = J * sigma for all Trotter slices
    void anneal(uint64_t seed);
    // One HU block for all Trotter slices: Metropolis sweep, then the GEMM update.
    void sweep_block(int start, float beta_eff, float j_perp, uint64_t seed, uint64_t rng_offset);
    void warm_up();
    double max_field_error();

    const Graph& graph_;
    SqaParams params_;
    int n_ = 0;                       // simulated spins (multiple of params_.block)

    __half* d_couplings_ = nullptr;   // J, n x n, column-major (symmetric)
    __half* d_spins_ = nullptr;       // sigma, n x M, column-major, values +-1
    float* d_fields_ = nullptr;       // h = J * sigma, n x M, column-major
    __half* d_flips_ = nullptr;       // spin changes of the current block, block x M
    float* d_scratch_ = nullptr;      // n x M, used only for verification

    cublasHandle_t cublas_ = nullptr;
    cublasComputeType_t compute_type_ = CUBLAS_COMPUTE_32F;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};
