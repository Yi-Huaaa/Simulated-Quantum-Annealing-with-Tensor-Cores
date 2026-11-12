#include "sqa.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <curand_kernel.h>

#include "cuda_check.h"

// Model
// -----
// MAX-CUT on a graph with edge weights w_ij is mapped to an Ising model with
// couplings J_ij = -w_ij (no external field). Simulated quantum annealing
// (path-integral Monte Carlo) replicates the N spins into M Trotter slices that
// form a ring, where spin i of slice m couples to spin i of slices m-1 and m+1
// with strength J_perp. Flipping sigma_{i,m} is accepted with the Metropolis
// rule  -ln(u) > beta_eff * dE / 2, where
//
//   dE / 2  = sigma_{i,m} * ( h_{i,m} + J_perp * (sigma_{i,m-1} + sigma_{i,m+1}) )
//   h_{i,m} = sum_j J_ij sigma_{j,m}                              (local field)
//   J_perp  = floor( -(M / 2) * ln(tanh(Gamma * beta / M)) / beta )
//   beta_eff = 2 * M * beta
//
// while Gamma decreases linearly from Gamma_0 to 0 and beta increases linearly
// from beta0 to beta1. This is the schedule of the ISC'22 paper.
//
// Note: the paper writes the inter-slice term with the opposite sign
// (anti-ferromagnetic). For an even number of slices both forms are
// equivalent under the gauge transformation sigma_{i,m} -> (-1)^m sigma_{i,m},
// which leaves the cut of every slice unchanged. The ferromagnetic form used
// here is the textbook one and also works for odd M.
//
// Hierarchical update (HU)
// ------------------------
// The N spins are processed in blocks of M_2 (= `block`) spins. For each block,
// one CUDA thread block per Trotter slice visits the M_2 spins sequentially and
// only keeps the local fields of the spins *inside the block* up to date. The
// block's spin changes (0 or -2 * sigma_old) are written to an M_2 x M matrix,
// and a single GEMM  h += J[:, block] * flips  then updates the local fields of
// all N spins in all M slices on the Tensor Cores.
//
// To let neighbouring slices run concurrently without updating the same spin
// index at the same time, odd slices start halfway through the block.

namespace {

constexpr int kThreadsPerBlock = 256;

using SweepKernel = void (*)(const __half*, const float*, __half*, __half*, int, int, int, int,
                             float, float, uint64_t, uint64_t);

// Random numbers come from counter-based Philox streams, one per
// (Trotter slice, spin) pair. Offset 0 initialises the spins; Monte Carlo step
// s uses offset s + 1.

__global__ void init_spins_kernel(__half* spins, int n, int num_nodes, int trotters, uint64_t seed) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(n) * trotters) {
        return;
    }
    float s = 1.0f;  // padding spins are fixed to +1 and never flipped
    if (static_cast<int>(idx % n) < num_nodes) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        s = (curand(&rng) & 1u) ? 1.0f : -1.0f;
    }
    spins[idx] = __float2half(s);
}

__global__ void scatter_couplings_kernel(__half* couplings, int n, const Edge* edges, size_t num_edges) {
    const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }
    const Edge edge = edges[e];
    const __half j = __float2half(-edge.w);  // J_ij = -w_ij; parallel edges accumulate
    atomicAdd(&couplings[static_cast<size_t>(edge.u) * n + edge.v], j);
    atomicAdd(&couplings[static_cast<size_t>(edge.v) * n + edge.u], j);
}

// Visits the BS spins [start, start + BS) of Trotter slice m = blockIdx.x.
// Thread t owns spin i = start + t and keeps its local field in a register. In
// iteration k only the owner of the visited spin decides whether to flip it;
// the decision is broadcast through shared memory so that every thread can
// update its own local field. `flip` is double-buffered, which needs only one
// barrier per iteration.
template <int BS>
__global__ void __launch_bounds__(BS)
sweep_block_kernel(const __half* __restrict__ couplings, const float* __restrict__ fields,
                   __half* spins, __half* __restrict__ flips, int n, int num_nodes, int trotters,
                   int start, float beta_eff, float j_perp, uint64_t seed, uint64_t rng_offset) {
    __shared__ float flip[2];

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int i = start + t;
    const size_t slice = static_cast<size_t>(m) * n;
    const size_t up = static_cast<size_t>(m == 0 ? trotters - 1 : m - 1) * n;
    const size_t down = static_cast<size_t>(m == trotters - 1 ? 0 : m + 1) * n;

    float h = fields[slice + i];

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, slice + i, rng_offset, &rng);
    const float threshold = -logf(curand_uniform(&rng));  // u in (0, 1]

    const int first = (m & 1) * (BS / 2);
    for (int k = 0; k < BS; ++k) {
        const int local = (first + k) & (BS - 1);
        if (t == local) {
            float ds = 0.0f;
            if (i < num_nodes) {
                const float s = __half2float(spins[slice + i]);
                const float neighbours = __half2float(spins[up + i]) + __half2float(spins[down + i]);
                if (threshold > beta_eff * s * (h + j_perp * neighbours)) {
                    spins[slice + i] = __float2half(-s);
                    ds = -2.0f * s;
                }
            }
            flips[static_cast<size_t>(m) * BS + local] = __float2half(ds);
            flip[k & 1] = ds;
        }
        __syncthreads();
        const float ds = flip[k & 1];
        if (ds != 0.0f) {  // uniform across the thread block
            h += __half2float(couplings[static_cast<size_t>(start + local) * n + i]) * ds;
        }
    }
}

SweepKernel sweep_kernel_for(int block) {
    switch (block) {
        case 32: return sweep_block_kernel<32>;
        case 64: return sweep_block_kernel<64>;
        case 128: return sweep_block_kernel<128>;
        case 256: return sweep_block_kernel<256>;
        case 512: return sweep_block_kernel<512>;
        case 1024: return sweep_block_kernel<1024>;
        default: return nullptr;
    }
}

unsigned int grid_size(size_t work) {
    return static_cast<unsigned int>((work + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

double gib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

SqaSolver::SqaSolver(const Graph& graph, const SqaParams& params) : graph_(graph), params_(params) {
    const SqaParams& p = params_;
    if (!sweep_kernel_for(p.block)) {
        throw std::invalid_argument("block size must be one of 32, 64, 128, 256, 512, 1024");
    }
    if (p.trotters < 1) throw std::invalid_argument("the number of Trotter slices must be >= 1");
    if (p.steps < 1) throw std::invalid_argument("the number of steps must be >= 1");
    if (!(p.gamma0 > 0.0) || !(p.beta0 > 0.0) || !(p.beta1 > 0.0)) {
        throw std::invalid_argument("gamma0, beta0 and beta1 must be positive");
    }

    const long long rounded = (static_cast<long long>(graph.num_nodes) + p.block - 1) / p.block * p.block;
    const long long n = p.size > 0 ? p.size : rounded;
    if (n < graph.num_nodes) {
        throw std::invalid_argument("size (" + std::to_string(n) + ") is smaller than the number of nodes (" +
                                    std::to_string(graph.num_nodes) + ")");
    }
    if (n % p.block != 0) {
        throw std::invalid_argument("size (" + std::to_string(n) + ") must be a multiple of the block size (" +
                                    std::to_string(p.block) + ")");
    }
    if (n > (1LL << 30) / p.trotters) {
        throw std::invalid_argument("size x trotters is too large");
    }
    n_ = static_cast<int>(n);

    const size_t nn = static_cast<size_t>(n_) * n_;
    const size_t nm = static_cast<size_t>(n_) * p.trotters;
    const size_t required = nn * sizeof(__half) + nm * (sizeof(__half) + 2 * sizeof(float)) +
                            static_cast<size_t>(p.block) * p.trotters * sizeof(__half);
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (required > free_bytes) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "N = %d needs %.2f GiB of GPU memory, but only %.2f GiB are free", n_,
                      gib(required), gib(free_bytes));
        throw std::runtime_error(msg);
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    // FP16 x FP16 -> FP32 GEMMs run on the Tensor Cores by default. The math
    // mode alone does not prevent that; the pedantic compute type does.
    CUBLAS_CHECK(cublasSetMathMode(cublas_, p.tensor_cores ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH));
    compute_type_ = p.tensor_cores ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;

    CUDA_CHECK(cudaMalloc(&d_couplings_, nn * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_spins_, nm * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_fields_, nm * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_, nm * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flips_, static_cast<size_t>(p.block) * p.trotters * sizeof(__half)));

    init_couplings();

    // Load the kernels and initialise cuBLAS so that the first timed run is not slower.
    warm_up();
}

SqaSolver::~SqaSolver() {
    cudaFree(d_couplings_);
    cudaFree(d_spins_);
    cudaFree(d_fields_);
    cudaFree(d_scratch_);
    cudaFree(d_flips_);
    if (cublas_) cublasDestroy(cublas_);
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
    if (stream_) cudaStreamDestroy(stream_);
}

void SqaSolver::init_couplings() {
    size_t inexact = 0;
    for (const Edge& e : graph_.edges) {
        if (__half2float(__float2half(e.w)) != e.w) ++inexact;
    }
    if (inexact > 0) {
        std::fprintf(stderr, "warning: %zu edge weights are not exactly representable in FP16 and are rounded\n",
                     inexact);
    }

    const size_t num_edges = graph_.edges.size();
    CUDA_CHECK(cudaMemsetAsync(d_couplings_, 0, static_cast<size_t>(n_) * n_ * sizeof(__half), stream_));
    if (num_edges > 0) {
        Edge* d_edges = nullptr;
        CUDA_CHECK(cudaMalloc(&d_edges, num_edges * sizeof(Edge)));
        CUDA_CHECK(cudaMemcpyAsync(d_edges, graph_.edges.data(), num_edges * sizeof(Edge), cudaMemcpyHostToDevice,
                                   stream_));
        scatter_couplings_kernel<<<grid_size(num_edges), kThreadsPerBlock, 0, stream_>>>(d_couplings_, n_, d_edges,
                                                                                         num_edges);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(d_edges));
    }
}

void SqaSolver::compute_fields(float* out) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, n_, params_.trotters, n_, &one,
                              d_couplings_, CUDA_R_16F, n_, d_spins_, CUDA_R_16F, n_, &zero, out, CUDA_R_32F, n_,
                              compute_type_, CUBLAS_GEMM_DEFAULT));
}

void SqaSolver::anneal(uint64_t seed) {
    const SqaParams& p = params_;
    const int trotters = p.trotters;

    for (int step = 0; step < p.steps; ++step) {
        const double progress = static_cast<double>(step) / p.steps;
        const double gamma = p.gamma0 * (1.0 - progress);
        const double beta = p.beta0 + (p.beta1 - p.beta0) * progress;
        // Rounded down to an integer, as in the implementation used for the
        // paper; this switches the inter-slice coupling off once J_perp < 1
        // and gives consistently better cuts than the unrounded value.
        const double j_perp =
            trotters > 1 ? std::floor(-0.5 * trotters * std::log(std::tanh(gamma * beta / trotters)) / beta) : 0.0;
        const float beta_eff = static_cast<float>(2.0 * trotters * beta);

        for (int start = 0; start < n_; start += p.block) {
            sweep_block(start, beta_eff, static_cast<float>(j_perp), seed, step + 1);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void SqaSolver::sweep_block(int start, float beta_eff, float j_perp, uint64_t seed, uint64_t rng_offset) {
    const SqaParams& p = params_;
    const float one = 1.0f;
    sweep_kernel_for(p.block)<<<p.trotters, p.block, 0, stream_>>>(d_couplings_, d_fields_, d_spins_, d_flips_, n_,
                                                                   graph_.num_nodes, p.trotters, start, beta_eff,
                                                                   j_perp, seed, rng_offset);
    // h += J[:, start:start+block] * flips  (N x M += N x M_2 * M_2 x M)
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, n_, p.trotters, p.block, &one,
                              d_couplings_ + static_cast<size_t>(start) * n_, CUDA_R_16F, n_, d_flips_, CUDA_R_16F,
                              p.block, &one, d_fields_, CUDA_R_32F, n_, compute_type_, CUBLAS_GEMM_DEFAULT));
}

void SqaSolver::warm_up() {
    const size_t nm = static_cast<size_t>(n_) * params_.trotters;
    init_spins_kernel<<<grid_size(nm), kThreadsPerBlock, 0, stream_>>>(d_spins_, n_, graph_.num_nodes,
                                                                      params_.trotters, 0);
    CUDA_CHECK(cudaGetLastError());
    compute_fields(d_fields_);
    sweep_block(0, 1.0f, 0.0f, 0, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

double SqaSolver::max_field_error() {
    compute_fields(d_scratch_);
    const size_t nm = static_cast<size_t>(n_) * params_.trotters;
    std::vector<float> fields(nm), reference(nm);
    CUDA_CHECK(cudaMemcpyAsync(fields.data(), d_fields_, nm * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(reference.data(), d_scratch_, nm * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    double error = 0.0;
    for (size_t k = 0; k < nm; ++k) {
        error = std::max(error, static_cast<double>(std::fabs(fields[k] - reference[k])));
    }
    return error;
}

RunResult SqaSolver::run(uint64_t seed, bool verify) {
    const int trotters = params_.trotters;
    const size_t nm = static_cast<size_t>(n_) * trotters;

    init_spins_kernel<<<grid_size(nm), kThreadsPerBlock, 0, stream_>>>(d_spins_, n_, graph_.num_nodes, trotters,
                                                                      seed);
    CUDA_CHECK(cudaGetLastError());
    compute_fields(d_fields_);

    CUDA_CHECK(cudaEventRecord(start_, stream_));
    anneal(seed);
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_));

    RunResult result;
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    result.seconds = ms / 1e3;

    std::vector<__half> spins(nm);
    CUDA_CHECK(cudaMemcpyAsync(spins.data(), d_spins_, nm * sizeof(__half), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<signed char> slice(graph_.num_nodes);
    for (int m = 0; m < trotters; ++m) {
        for (int i = 0; i < graph_.num_nodes; ++i) {
            slice[i] = __half2float(spins[static_cast<size_t>(m) * n_ + i]) > 0.0f ? 1 : -1;
        }
        const double cut = cut_value(graph_, slice);
        if (m == 0 || cut > result.best_cut) {
            result.best_cut = cut;
            result.best_trotter = m;
        }
    }

    if (verify) {
        result.field_error = max_field_error();
    }
    return result;
}
