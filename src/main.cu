// Command-line driver: solves MAX-CUT on a G-set graph with GPU-accelerated SQA.
#include <getopt.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "cuda_check.h"
#include "graph.h"
#include "sqa.h"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options] <graph>\n"
        "\n"
        "Solves MAX-CUT on <graph> (G-set format) with simulated quantum annealing.\n"
        "\n"
        "Options:\n"
        "  -m, --trotters M       number of Trotter slices (default 16)\n"
        "  -b, --block M2         hierarchical-update block size: 32, 64, 128, 256,\n"
        "                         512 or 1024 (default 128)\n"
        "  -n, --size N           number of simulated spins; the graph is padded with\n"
        "                         isolated spins up to N, which must be a multiple of\n"
        "                         the block size (default: nodes rounded up to M2)\n"
        "  -s, --steps S          Monte Carlo steps per run (default 100)\n"
        "  -r, --runs R           number of independent runs (default 5)\n"
        "  -g, --gamma0 G         initial transverse field (default 8)\n"
        "      --beta0 B          inverse temperature at the first step (default 0.0625)\n"
        "      --beta1 B          inverse temperature at the end (default 16)\n"
        "      --seed X           random seed (default 1)\n"
        "      --no-tensor-cores  run the GEMMs on CUDA cores only\n"
        "      --check            after every run, verify the incrementally updated\n"
        "                         local fields against a full recomputation\n"
        "  -h, --help             show this help\n",
        prog);
}

[[noreturn]] void bad_option(const char* name, const char* value) {
    std::fprintf(stderr, "error: invalid value '%s' for --%s\n", value, name);
    std::exit(2);
}

long long parse_int(const char* value, const char* name, long long min) {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || v < min || v > 0x7fffffff) bad_option(name, value);
    return v;
}

double parse_positive(const char* value, const char* name) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !(v > 0.0)) bad_option(name, value);
    return v;
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

enum LongOnly { kBeta0 = 1000, kBeta1, kSeed, kNoTensorCores, kCheck };

}  // namespace

int main(int argc, char** argv) {
    SqaParams params;
    int runs = 5;
    uint64_t seed = 1;
    bool check = false;

    static const option long_options[] = {
        {"trotters", required_argument, nullptr, 'm'},
        {"block", required_argument, nullptr, 'b'},
        {"size", required_argument, nullptr, 'n'},
        {"steps", required_argument, nullptr, 's'},
        {"runs", required_argument, nullptr, 'r'},
        {"gamma0", required_argument, nullptr, 'g'},
        {"beta0", required_argument, nullptr, kBeta0},
        {"beta1", required_argument, nullptr, kBeta1},
        {"seed", required_argument, nullptr, kSeed},
        {"no-tensor-cores", no_argument, nullptr, kNoTensorCores},
        {"check", no_argument, nullptr, kCheck},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:b:n:s:r:g:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': params.trotters = static_cast<int>(parse_int(optarg, "trotters", 1)); break;
            case 'b':
                params.block = static_cast<int>(parse_int(optarg, "block", 32));
                if (params.block > 1024 || (params.block & (params.block - 1)) != 0) bad_option("block", optarg);
                break;
            case 'n': params.size = static_cast<int>(parse_int(optarg, "size", 1)); break;
            case 's': params.steps = static_cast<int>(parse_int(optarg, "steps", 1)); break;
            case 'r': runs = static_cast<int>(parse_int(optarg, "runs", 1)); break;
            case 'g': params.gamma0 = parse_positive(optarg, "gamma0"); break;
            case kBeta0: params.beta0 = parse_positive(optarg, "beta0"); break;
            case kBeta1: params.beta1 = parse_positive(optarg, "beta1"); break;
            case kSeed: seed = static_cast<uint64_t>(parse_int(optarg, "seed", 0)); break;
            case kNoTensorCores: params.tensor_cores = false; break;
            case kCheck: check = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 2;
        }
    }
    if (optind != argc - 1) {
        print_usage(argv[0]);
        return 2;
    }
    const std::string path = argv[optind];

    try {
        const Graph graph = load_gset(path);
        SqaSolver solver(graph, params);

        int device = 0;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::printf("graph        : %s (%d nodes, %zu edges)\n", path.c_str(), graph.num_nodes, graph.edges.size());
        std::printf("device       : %s\n", prop.name);
        std::printf("spins (N)    : %d\n", solver.size());
        std::printf("trotters (M) : %d\n", params.trotters);
        std::printf("block (M2)   : %d\n", params.block);
        std::printf("steps        : %d\n", params.steps);
        std::printf("schedule     : gamma %g -> 0, beta %g -> %g\n", params.gamma0, params.beta0, params.beta1);
        std::printf("tensor cores : %s\n", params.tensor_cores ? "on" : "off");
        std::printf("seed         : %llu\n\n", static_cast<unsigned long long>(seed));

        std::printf("%4s %12s %12s %8s%s\n", "run", "time [s]", "cut", "trotter", check ? "  field error" : "");
        double total_time = 0.0, total_cut = 0.0, best_cut = 0.0, worst_error = 0.0;
        for (int r = 0; r < runs; ++r) {
            const RunResult res = solver.run(splitmix64(seed + static_cast<uint64_t>(r)), check);
            total_time += res.seconds;
            total_cut += res.best_cut;
            if (r == 0 || res.best_cut > best_cut) best_cut = res.best_cut;
            std::printf("%4d %12.6f %12.10g %8d", r, res.seconds, res.best_cut, res.best_trotter);
            if (check) {
                std::printf("  %11.3g", res.field_error);
                if (res.field_error > worst_error) worst_error = res.field_error;
            }
            std::printf("\n");
        }

        std::printf("\navg time [s] : %.6f\n", total_time / runs);
        std::printf("avg cut      : %.10g\n", total_cut / runs);
        std::printf("best cut     : %.10g\n", best_cut);

        if (check) {
            // Integer weights are represented exactly, so any drift indicates a bug.
            const double tolerance = 1e-3;
            const bool ok = worst_error <= tolerance;
            std::printf("field check  : %s (max error %.3g)\n", ok ? "PASSED" : "FAILED", worst_error);
            if (!ok) return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
