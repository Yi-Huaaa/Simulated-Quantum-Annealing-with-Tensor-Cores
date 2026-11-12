# Simulated Quantum Annealing with Tensor Cores

**GPU-accelerated simulated quantum annealing for fully-connected Ising models**

[![Paper](https://img.shields.io/badge/paper-ISC%202022-blue)](https://doi.org/10.1007/978-3-031-07312-0_9)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

This solver finds low-energy states of fully-connected Ising models, such as
MAX-CUT instances, with simulated quantum annealing (SQA) on NVIDIA GPUs. A
*hierarchical update* lets every Trotter slice examine a block of spins at a
time. The local-field updates of all slices then become one matrix
multiplication, which runs on the GPU's Tensor Cores.

This repository is the artifact of our ISC High Performance 2022 paper
[*Accelerating Simulated Quantum Annealing with GPU and Tensor
Cores*](https://doi.org/10.1007/978-3-031-07312-0_9)
([PDF](https://yi-huaaa.github.io/uploads/publications/2022-sqa-gpu-tensor-cores.pdf),
[slides](https://yi-huaaa.github.io/uploads/publications/2022-sqa-gpu-tensor-cores-slides.pdf)).
It contains the SQA solver and the scripts that reproduce the single-GPU
experiments of the paper.

## Highlights

- **Hierarchical update.** Each Trotter slice examines a block of M2 spins at a
  time and updates only their local fields, so all slices run in parallel with
  few kernel launches. In the paper, this alone made SQA 7x to 48x faster than
  the state-of-the-art GPU solver of Tohoku University.
- **Tensor Cores.** The local fields of all N·M spins are updated with FP16
  GEMMs and FP32 accumulation. On an RTX A4000 this speeds up the whole
  annealing by 1.2x to 1.35x for M ≥ 128. The paper measured up to 1.83x on a
  V100.
- **Large fully-connected models.** 100 Monte Carlo steps for N = 32768 spins and
  M = 16 Trotter slices take 1.5 s, and N = 65536 fits on a 16 GB GPU.
- **Verified results.** `--check` compares the incrementally updated local
  fields with a full recomputation. All 168 experiment runs passed with zero
  error.

Measured on an NVIDIA RTX A4000; see [results/](results/README.md) for the full
tables.

## Requirements

- An NVIDIA GPU with Tensor Cores (Volta, compute capability 7.0, or newer)
- **CUDA 11 or newer**, including cuBLAS
- GNU Make and Bash; `curl` to download the G-set graphs and Python 3 for the
  experiment scripts

## Quick start

```bash
git clone https://github.com/Yi-Huaaa/Simulated-Quantum-Annealing-with-Tensor-Cores.git
cd Simulated-Quantum-Annealing-with-Tensor-Cores
make            # builds build/sqa for the GPU of this machine
make test       # optional: end-to-end tests on graphs with known maximum cuts
make gset       # downloads the G-set graphs of the experiments into data/gset/
```

Solve G1 (800 nodes, 19176 edges) with the default settings, and check the local
fields after every run:

```bash
./build/sqa --check data/gset/G1
```

```
graph        : data/gset/G1 (800 nodes, 19176 edges)
device       : NVIDIA RTX A4000
spins (N)    : 896
trotters (M) : 16
block (M2)   : 128
steps        : 100
runs         : 5
schedule     : gamma 8 -> 0, beta 0.0625 -> 16
tensor cores : on
seed         : 1

 run     time [s]          cut  trotter  field error
   0     0.022148        11519       15            0
   1     0.022130        11509       10            0
   2     0.022146        11498        2            0
   3     0.022110        11510        3            0
   4     0.022124        11460        9            0

avg time [s] : 0.022132
avg cut      : 11499.2
best cut     : 11519
field check  : PASSED (max error 0)
```

## How SQA with hierarchical update works

1. **Trotter slices.** SQA replicates the N spins into M Trotter slices that
   form a ring. Spin i of slice m interacts with all spins of its slice through
   the couplings J, and with spin i of the two neighboring slices through J⊥.
   For MAX-CUT with edge weights w, J = −w. A flip of σ(i,m) is accepted when

   ```text
   -ln(u) > 2·M·β · σ(i,m) · ( h(i,m) + J⊥ · (σ(i,m-1) + σ(i,m+1)) ),   u ~ U(0, 1]
   h(i,m) = Σ_j J(i,j) · σ(j,m)                                        (local field)
   J⊥     = floor( -(M/2) · ln(tanh(Γ·β/M)) / β )
   ```

   During the annealing, Γ falls linearly from Γ₀ to 0 and β rises linearly
   from β₀ to β₁.
2. **Hierarchical update.** The spins are processed in blocks of M2 (blk_sz in
   the paper). One CUDA thread block per Trotter slice examines the M2 spins of a
   block one after another, and updates the local fields of only those M2 spins
   in registers. Odd slices start in the middle of the block, so neighboring
   slices never flip the same spin at the same time.
3. **Tensor Cores.** The flips of the block form an M2 × M matrix ΔΣ. One GEMM,
   `h += J[:, block] · ΔΣ`, then updates the local fields of all N spins in all
   M slices. cuBLAS runs it on the Tensor Cores, with FP16 inputs and FP32
   accumulation, which is exact for integer weights.

The implementation is in [src/sqa.cu](src/sqa.cu).

## Usage

```
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

Graphs use the [G-set](https://web.stanford.edu/~yyye/yyye/Gset/) format: a line
`<nodes> <edges>`, then one line `<u> <v> <weight>` per edge with 1-based node
indices. The coupling matrix is stored densely in FP16, which takes 2·N² bytes
of GPU memory (2 GiB for N = 32768) and represents integer weights up to ±2048
exactly. To solve another Ising or QUBO problem, map it to couplings J and adapt
`init_couplings()` and the cut evaluation in [src/sqa.cu](src/sqa.cu).

**Choosing M and M2.** More Trotter slices M give SQA more quantum
fluctuations, which can improve the solution, but the time grows with M. M2
trades the sequential work of step 2 against the efficiency of the GEMM in
step 3. The best value depends on the GPU; the paper uses M2 = 128.

**Annealing schedule.** The defaults (Γ₀ = 8, β from 1/16 to 16, 100 steps) are
the settings of the speed experiments in the paper. For better cuts, use more
steps, more slices and a colder end. The quality experiment uses
`--steps 1000 --trotters 64 --beta1 64` for every graph; see
[results/](results/README.md).

**Reported metrics.**

- *time* covers the annealing loop of one run. It excludes reading the graph,
  building the coupling matrix on the GPU, and computing the initial local
  fields.
- *cut* is the best cut among the M Trotter slices at the end of a run,
  computed exactly from the edge list. *trotter* is the slice that has it.
- *avg time*, *avg cut* and *best cut* summarize all runs. Every run starts from
  a new random state, and `--seed` makes the runs reproducible.

## Reproducing the paper

1. **Build and get the graphs**: `make && make gset`.
2. **Run the experiments.**

| Script | Experiment of the paper |
|---|---|
| `scripts/run_quality.sh` | Sec. 4.5, Fig. 9: solution quality on the G-set graphs of Table 2 |
| `scripts/run_block_size.sh` | Sec. 4.3, Fig. 7: annealing time for block sizes M2 = 32 … 1024 |
| `scripts/run_tensor_cores.sh` | Sec. 4.4, Fig. 8: annealing time with and without Tensor Cores |
| `scripts/run_all.sh` | all of the above |

Each script appends one CSV row per configuration to
`results/local/<experiment>.csv`, keeps the full solver output in
`results/local/logs/`, and prints Markdown tables at the end. To print them
again, run `scripts/summarize.py quality results/local/quality.csv` (or
`block_size`, `tensor_cores`). To run only some graphs of the quality
experiment, pass their names: `scripts/run_quality.sh G43 G48`.

The settings are environment variables. Their defaults are the settings of the
paper:

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `SIZES` | 1024 … 32768 | tensor_cores | problem sizes N; each pads one G-set graph (G1, G22, G48, G65, G77, G81) |
| `SIZE` | 8192 | block_size | problem size N |
| `TROTTERS` | 4 … 512 (quality: 64) | all | Trotter numbers M |
| `BLOCKS` / `BLOCK` | 32 … 1024 / 128 | block_size / others | block sizes M2 |
| `STEPS` | 100 (quality: 1000) | all | Monte Carlo steps per run |
| `RUNS` | 10 (block_size: 5) | all | runs per configuration |
| `OPTIONS` | `--beta1 64` | quality | extra solver options |
| `GSET_DIR` | `data/gset` | all | where the G-set graphs are |
| `RESULTS_DIR` | `results/local` | all | where the results go |

For example: `TROTTERS="16 64" RUNS=3 scripts/run_tensor_cores.sh`. Every run
uses `--check`, and a failed check stops the script. With the default settings,
`scripts/run_all.sh` takes about 21 minutes on an RTX A4000.

Two experiments of the paper are not included. Sec. 4.1 (Fig. 6) needs an
implementation without the hierarchical update. Sec. 4.2 (Table 1) compares
against the GPU and FPGA solvers of Tohoku University, using numbers that their
authors provided.

Our results, and the machine and software versions they were measured with, are
in [results/](results/README.md).

## Build options

`make` compiles `build/sqa` with `nvcc`. Override these variables on the command
line:

- `ARCH`: GPU architecture, `native` (the GPU of the build machine) by default.
  For example, `make ARCH=sm_80` builds for an A100. Set it explicitly if your
  `nvcc` does not support `-arch=native`.
- `NVCC`: the CUDA compiler, for example `make NVCC=/usr/local/cuda-12.4/bin/nvcc`.
- `NVCCFLAGS`: compiler flags, `-O3 -std=c++17 -lineinfo` with host warnings by
  default.

`make test` runs [tests/run_tests.sh](tests/run_tests.sh). It solves graphs with
known maximum cuts (tori, complete and complete bipartite graphs, negative
weights) for odd and even M, every block size, padding and CUDA-core GEMMs, and
checks invalid input. Every test run uses `--check`. The kernels are also clean
under `compute-sanitizer` (memcheck, racecheck, synccheck and initcheck).

## Repository layout

```
src/          the solver: command-line tool (main.cu), SQA kernels, hierarchical
              update and cuBLAS GEMMs (sqa.cu), G-set reader and cut (graph.cpp)
tests/        end-to-end tests (make test)
scripts/      G-set download, experiment scripts and result tables
results/      results of the experiments on our machine
```

## Changes from the code used for the paper

The code used for the paper is in the git history (`SQA_fully/v5-1-9.cu` and
earlier versions). This version keeps the algorithm, and fixes the following:

- **Stale local fields across runs.** Only Trotter slice 0 was reset before the
  local fields were rebuilt, so slices 1 … M−1 kept the fields of all earlier
  runs, and only the first run of a program execution was correct. These runs
  often reported higher cuts than correct SQA does, so the average cuts of the
  old code are not directly comparable.
- **Data races in the flip kernel.** Every thread of a block examined and
  flipped the same spin, so warps could undo each other's flips and the local
  fields drifted from J·σ. Now one thread decides each flip and broadcasts it.
- **Random numbers.** `rand()` under OpenMP and an unsynchronized copy on a
  second CUDA stream are replaced by Philox streams on the GPU.
- **Large N.** The GEMM offsets and leading dimension were wrong for
  N > 32768. All offsets are now 64-bit, and N = 65536 passes `--check`.
- **CUDA cores.** Disabling Tensor Cores now uses a pedantic compute type.
  The pedantic math mode alone still ran the GEMMs on the Tensor Cores.
- **Kept on purpose.** J⊥ is rounded down to an integer, as the old kernel took
  it as an `int`. In our tests this gives better cuts than the exact value. The
  paper writes the J⊥ term with the opposite sign; for even M both signs are
  equivalent under σ(i,m) → (−1)<sup>m</sup>·σ(i,m), and the sign used here also
  works for odd M.
- **Usability.** N, M and M2 are command-line options instead of macros
  patched with `sed`, and the cut is evaluated for all slices, not only for
  slice 0.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{chung2022accelerating,
  author    = {Chung, Yi-Hua and Shih, Cheng-Jhih and Hung, Shih-Hao},
  title     = {Accelerating Simulated Quantum Annealing with {GPU} and Tensor Cores},
  booktitle = {High Performance Computing (ISC High Performance 2022)},
  pages     = {174--191},
  publisher = {Springer},
  year      = {2022},
  doi       = {10.1007/978-3-031-07312-0_9}
}
```

## Contact

For questions about this code, please email Yi-Hua Chung at
[yihua.chung@wisc.edu](mailto:yihua.chung@wisc.edu).

## License

This code is released under the [Apache License 2.0](LICENSE).
Copyright 2021-2026 Yi-Hua Chung, Cheng-Jhih Shih, and contributors.
