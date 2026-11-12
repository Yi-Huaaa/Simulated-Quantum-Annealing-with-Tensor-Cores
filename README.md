# Simulated Quantum Annealing with Tensor Cores

GPU-accelerated simulated quantum annealing (SQA) for fully-connected Ising
models, applied to MAX-CUT. This is the code accompanying:

> Yi-Hua Chung, Cheng-Jhih Shih, Shih-Hao Hung.
> **Accelerating Simulated Quantum Annealing with GPU and Tensor Cores.**
> ISC High Performance 2022, pp. 174–191.
> [doi:10.1007/978-3-031-07312-0_9](https://doi.org/10.1007/978-3-031-07312-0_9)

[Slides](docs/isc22-slides.pdf) ·
[Project page](https://yi-huaaa.github.io/publications/sqa-gpu-tensor-cores/) ·
[Blog post with talk recording](https://yi-huaaa.github.io/2022/05/10/Accelerating%20Simulated%20Quantum%20Annealing%20with%20GPU%20and%20Tensor%20Cores/)

Inspired by quantum annealing, SQA mimics quantum tunneling on classical
computers by running a path-integral Monte Carlo simulation. Previous work
accelerated SQA with GPUs and FPGAs. We propose a parallelization strategy
called **hierarchical update (HU)**, which accelerates state-of-the-art SQA
implementations by 7x to 47.2x in our case studies. We also introduce a
tensorizing scheme that runs the local-field updates on the **Tensor Cores**
of modern GPUs, which adds up to 1.83x of speedup. Overall, the paper reports
86.6x speedup over existing GPU-based SQA solvers and 14x over FPGA-based ones
on fully-connected Ising models.

## Requirements

- An NVIDIA GPU with Tensor Cores (Volta, compute capability 7.0, or newer)
- CUDA Toolkit 11 or newer, including cuBLAS
- GNU Make and Bash; `curl` to download the benchmark graphs

Tested with CUDA 12.0 on an NVIDIA RTX A4000 (Ubuntu 24.04).

## Quick start

```bash
make            # build build/sqa for the GPU of this machine
make test       # end-to-end tests on graphs with known maximum cuts
make gset       # download the G-set graphs used in the paper into data/gset/
./build/sqa data/gset/G1
```

`make` targets the local GPU (`ARCH=native`). To build for another GPU, or if
your `nvcc` does not support `-arch=native`, set the architecture explicitly,
for example `make ARCH=sm_80` for an A100.

Example output:

```text
graph        : data/gset/G1 (800 nodes, 19176 edges)
device       : NVIDIA RTX A4000
spins (N)    : 896
trotters (M) : 16
block (M2)   : 128
steps        : 100
schedule     : gamma 8 -> 0, beta 0.0625 -> 16
tensor cores : on
seed         : 1

 run     time [s]          cut  trotter
   0     0.022109        11519       15
   1     0.022134        11509       10
   2     0.022105        11498        2
   3     0.022111        11510        3
   4     0.022071        11460        9

avg time [s] : 0.022106
avg cut      : 11499.2
best cut     : 11519
```

Every run starts from a new random state. `time` covers the annealing loop only,
`cut` is the best cut found among the M Trotter slices, and `trotter` is the
slice that produced it.

## Usage

```text
Usage: sqa [options] <graph>

  -m, --trotters M       number of Trotter slices (default 16)
  -b, --block M2         hierarchical-update block size: 32, 64, 128, 256,
                         512 or 1024 (default 128)
  -n, --size N           number of simulated spins; the graph is padded with
                         isolated spins up to N, which must be a multiple of
                         the block size (default: nodes rounded up to M2)
  -s, --steps S          Monte Carlo steps per run (default 100)
  -r, --runs R           number of independent runs (default 5)
  -g, --gamma0 G         initial transverse field (default 8)
      --beta0 B          inverse temperature at the first step (default 0.0625)
      --beta1 B          inverse temperature at the end (default 16)
      --seed X           random seed (default 1)
      --no-tensor-cores  run the GEMMs on CUDA cores only
      --check            after every run, verify the incrementally updated
                         local fields against a full recomputation
```

All parameters are set at run time, so you can sweep N, M and M2 without
recompiling.

**Input format.** Graphs use the
[G-set](https://web.stanford.edu/~yyye/yyye/Gset/) format: a header
`<nodes> <edges>` followed by one `<u> <v> <weight>` line per edge, with 1-based
node indices. The coupling matrix is stored densely in FP16 (2·N² bytes of GPU
memory), so integer weights up to ±2048 are represented exactly.

**Other problems.** The solver minimizes the Ising energy with
J<sub>ij</sub> = −w<sub>ij</sub>, which is equivalent to maximizing the cut. To
solve another QUBO or Ising problem, map it to Ising couplings and adapt
`init_couplings()` and the cut evaluation in `src/sqa.cu`.

## How it works

SQA replicates the N spins into M Trotter slices that form a ring. Spin i of
slice m interacts with every spin of its own slice through J and with spin i of
the neighboring slices through J⊥. A spin flip is accepted when

```text
-ln(u) > 2·M·β · σ(i,m) · ( h(i,m) + J⊥ · (σ(i,m-1) + σ(i,m+1)) ),   u ~ U(0, 1]
h(i,m) = Σ_j J(i,j) · σ(j,m)                                        (local field)
J⊥     = floor( -(M/2) · ln(tanh(Γ·β/M)) / β )
```

Over the run, Γ decreases linearly from Γ₀ to 0, and β increases linearly from
β₀ to β₁.

The cost is dominated by keeping the N×M local fields up to date after every
flip. The **hierarchical update** splits the spins into blocks of M2:

1. For each block, one CUDA thread block per Trotter slice visits the M2 spins
   in turn. It keeps only the local fields of the spins inside the block up to
   date, in registers. Neighboring slices start at different halves of the
   block, so they can run concurrently.
2. The flips of the block form an M2×M matrix ΔΣ. One GEMM,
   `h += J[:, block] · ΔΣ`, then updates the local fields of all N spins in all
   M slices. cuBLAS runs it on the **Tensor Cores** (FP16 inputs, FP32
   accumulation, exact for integer weights).

The block size M2 trades sequential work in step 1 against GEMM efficiency in
step 2, and the best value depends on the GPU (see the paper). The
implementation is in [`src/sqa.cu`](src/sqa.cu).

## Benchmarks

`make bench` runs the N × M sweep from the paper. Each size N uses one G-set
graph, padded with isolated spins up to N (G1, G22, G48, G65, G77 and G81 for
N = 1024 … 32768). The results are written to `results/benchmark.csv`. Use
environment variables to select a subset:

```bash
SIZES="1024 8192" TROTTERS="16 64" BLOCKS="64 128 256" RUNS=3 make bench
```

Results on an RTX A4000 with CUDA 12.0, M = 16, M2 = 128, 100 steps, and the
average of 5 runs:

| Graph | Nodes | N     | Time per run [s] | Avg. cut | Best cut |
|-------|------:|------:|-----------------:|---------:|---------:|
| G1    |   800 |  1024 |            0.024 |  11496.8 |    11512 |
| G22   |  2000 |  2048 |            0.056 |    13070 |    13081 |
| G48   |  3000 |  4096 |            0.103 |     6000 |     6000 |
| G65   |  8000 |  8192 |            0.234 |   5338.8 |     5358 |
| G77   | 14000 | 16384 |            0.541 |   9557.2 |     9582 |
| G81   | 20000 | 32768 |            1.467 |    13506 |    13540 |

Solution quality depends strongly on M, the schedule (`--gamma0`, `--beta0`,
`--beta1`) and the number of steps. The defaults follow the paper and are not
tuned per instance.

## Testing

`make test` solves small graphs with known maximum cuts (tori, complete and
complete bipartite graphs, negative weights). It covers odd and even numbers of
Trotter slices, every block size, padding, CUDA-core GEMMs and invalid input.
Every test run uses `--check`, which fails if the incrementally updated local
fields drift from a full recomputation of J·σ.

The kernels are clean under `compute-sanitizer` (memcheck, racecheck,
synccheck and initcheck).

## Repository layout

```text
src/
  main.cu                command-line interface
  sqa.h, sqa.cu          SQA solver: kernels, hierarchical update, cuBLAS GEMMs
  graph.h, graph.cpp     G-set reader and cut evaluation
  cuda_check.h           CUDA / cuBLAS error checking
tests/run_tests.sh       end-to-end tests (make test)
scripts/
  download_gset.sh       downloads G-set instances (make gset)
  benchmark.sh           N x M x M2 sweep (make bench)
docs/isc22-slides.pdf    slides of the ISC 2022 talk
```

## Differences from the original research code

The code used for the paper is in the git history (`SQA_fully/v5-1-9.cu` and
earlier versions). This rewrite keeps the algorithm and fixes the following
problems:

- **Stale local fields across runs.** Only Trotter slice 0 was reset before
  the local fields were rebuilt, so slices 1 … M−1 kept the fields of all
  earlier runs. Only the first run of each execution was correct. These
  corrupted runs often reported *higher* cuts than correct SQA does, so
  average cuts from the old code are not directly comparable.
- **Data races in the flip kernel.** Every thread of a block evaluated and
  flipped the same spin. Warps could undo each other's flips, and the local
  fields drifted from J·σ. A single thread now decides each flip and
  broadcasts it.
- **Random numbers.** `rand()` under OpenMP and an unsynchronized copy on a
  second stream are replaced by per-spin Philox streams on the GPU. Runs are
  reproducible with `--seed`.
- **N > 32768.** The GEMM offsets and leading dimension were wrong for large
  N. Indexing now uses 64-bit offsets, and N = 65536 has been verified with
  `--check`.
- **Inter-slice coupling sign.** The paper writes the J⊥ term with the
  opposite (anti-ferromagnetic) sign. For an even M both forms are equivalent
  under σ(i,m) → (−1)<sup>m</sup>·σ(i,m), and the cut of every slice is
  unchanged. The ferromagnetic form used here also works for odd M.
- **Rounding of J⊥.** The original code passed J⊥ to the kernel as an
  `int`. This is kept on purpose (`floor`): it gave consistently better cuts
  than the unrounded value in our tests.
- **Usability.** N, M and M2 are command-line options instead of macros
  patched with `sed`. The cut is computed exactly from the edge list, and all
  slices are evaluated, not only slice 0.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chung2022accelerating,
  title     = {Accelerating Simulated Quantum Annealing with {GPU} and Tensor Cores},
  author    = {Chung, Yi-Hua and Shih, Cheng-Jhih and Hung, Shih-Hao},
  booktitle = {International Conference on High Performance Computing},
  pages     = {174--191},
  year      = {2022},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-07312-0_9}
}
```

## References

- Tohoku GPU: [A GPU-Based Quantum Annealing Simulator for Fully-Connected Ising Models Utilizing Spatial and Temporal Parallelism](https://ieeexplore.ieee.org/abstract/document/9057502)
- Tohoku FPGA: [Highly-Parallel FPGA Accelerator for Simulated Quantum Annealing](https://ieeexplore.ieee.org/document/8918417)
- [An Ising Computer Based on Simulated Quantum Annealing by Path Integral Monte Carlo Method](https://ieeexplore.ieee.org/abstract/document/8123652)
- [Accelerator Architecture for Simulated Quantum Annealing Based on Resource-Utilization-Aware Scheduling and its Implementation Using OpenCL](https://ieeexplore.ieee.org/abstract/document/8923263)
- [Quantum Annealing in the Transverse Ising Model](https://arxiv.org/abs/cond-mat/9804280)
- [G-set MAX-CUT instances](https://web.stanford.edu/~yyye/yyye/Gset/)

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The slides in
`docs/isc22-slides.pdf` are not covered by the Apache License and remain
subject to their existing copyright.
