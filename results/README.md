# Results

These are the results of the experiment scripts (`scripts/run_*.sh`) on the
machine below, with their default settings.

- `rtx-a4000/<experiment>.csv`: one row per configuration, parsed from the
  solver output
- `rtx-a4000/logs/<experiment>/`: the full solver output of every configuration
- `rtx-a4000/run.out`: the console output of `scripts/run_all.sh`, with the
  commands, CSV rows and summary tables
- The tables below were produced with
  `scripts/summarize.py <experiment> rtx-a4000/<experiment>.csv`.

## Machine

| | |
|---|---|
| GPU | NVIDIA RTX A4000 (Ampere, sm_86, 48 SMs, 16 GB), driver 550.163.01 |
| CPU | Intel Core i5-13500 (14 cores, 20 threads) |
| Memory | 125 GiB |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-138 |
| Compilers | CUDA 12.0 (nvcc 12.0.140, cuBLAS 12.0.2), GCC 13.3.0 |
| Build | `make` (`-O3`, `ARCH=native`, which is sm_86 here) |
| Code | commit `3b0cf20` |

## Settings

- Each configuration runs R times from independent random states (seeds
  derived from `--seed 1`). Times are the mean annealing time per run, and
  cuts are the mean and the best over the R runs.
- Every run used `--check`: after the timed loop, the incrementally updated
  local fields were compared with a full recomputation of J·σ. All 168
  configurations passed, with a maximum error of 0.
- With these settings, `scripts/run_all.sh` took 21 minutes on this machine.

| Experiment | Graphs | N | M | M2 | Steps | Schedule | Runs |
|---|---|---|---|---|---|---|---|
| Solution quality | Table 2 of the paper | nodes rounded up to M2 | 64 | 128 | 1000 | Γ 8 → 0, β 1/16 → 64 | 10 |
| Block size | G65, padded | 8192 | 4 … 512 | 32 … 1024 | 100 | Γ 8 → 0, β 1/16 → 16 | 5 |
| Tensor Cores | G1, G22, G48, G65, G77, G81, padded | 1024 … 32768 | 4 … 512 | 128 | 100 | Γ 8 → 0, β 1/16 → 16 | 10 |

## Solution quality (Sec. 4.5, Fig. 9)

Cuts relative to the best known cuts in Table 2 of the paper. One schedule is
used for every graph. The paper tuned the schedule and reports best cuts within
99.66% and average cuts within 98.97% of the best known cuts. With a single
schedule, most graphs with ±1 weights (G9, G18 – G21, G31, G39, G41, G42) stay
at 93% – 97%. G13 and G34 also have ±1 weights but reach about 98.5%.

| Graph | Nodes | Edges | Best known | Avg. cut | Best cut | Avg. / known | Best / known |
|---:|---:|---:|---:|---:|---:|---:|---:|
| G5 | 800 | 19176 | 11631 | 11541.7 | 11585 | 99.23% | 99.60% |
| G9 | 800 | 19176 | 2054 | 1950.6 | 1986 | 94.97% | 96.69% |
| G13 | 800 | 1600 | 582 | 573.8 | 576 | 98.59% | 98.97% |
| G18 | 800 | 4694 | 992 | 944.1 | 959 | 95.17% | 96.67% |
| G19 | 800 | 4661 | 906 | 853.4 | 865 | 94.19% | 95.47% |
| G20 | 800 | 4672 | 941 | 888.3 | 902 | 94.40% | 95.86% |
| G21 | 800 | 4667 | 931 | 880 | 890 | 94.52% | 95.60% |
| G26 | 2000 | 19990 | 13328 | 13133.4 | 13160 | 98.54% | 98.74% |
| G31 | 2000 | 19990 | 3309 | 3107.4 | 3150 | 93.91% | 95.19% |
| G34 | 2000 | 4000 | 1384 | 1361.8 | 1366 | 98.40% | 98.70% |
| G38 | 2000 | 11779 | 7688 | 7555.7 | 7567 | 98.28% | 98.43% |
| G39 | 2000 | 11778 | 2408 | 2251.7 | 2271 | 93.51% | 94.31% |
| G41 | 2000 | 11785 | 2405 | 2240.9 | 2291 | 93.18% | 95.26% |
| G42 | 2000 | 11779 | 2481 | 2319.7 | 2351 | 93.50% | 94.76% |
| G43 | 1000 | 9990 | 6660 | 6567.4 | 6584 | 98.61% | 98.86% |
| G44 | 1000 | 9990 | 6650 | 6568.3 | 6591 | 98.77% | 99.11% |
| G47 | 1000 | 9990 | 6657 | 6565.9 | 6587 | 98.63% | 98.95% |
| G48 | 3000 | 6000 | 6000 | 6000 | 6000 | 100.00% | 100.00% |
| G49 | 3000 | 6000 | 6000 | 5982 | 6000 | 99.70% | 100.00% |
| G50 | 3000 | 6000 | 5880 | 5876.4 | 5880 | 99.94% | 100.00% |
| G51 | 1000 | 5909 | 3848 | 3787.5 | 3797 | 98.43% | 98.67% |
| G53 | 1000 | 5914 | 3850 | 3794.7 | 3802 | 98.56% | 98.75% |
| G54 | 1000 | 5916 | 3852 | 3792.9 | 3798 | 98.47% | 98.60% |
| G63 | 7000 | 41459 | 27045 | 26529.1 | 26550 | 98.09% | 98.17% |

Mean over the 24 graphs: the average cut is 97.07% and the best cut is 97.72%
of the best known cut (lowest: 93.18% and 94.31%).

## Block size (Sec. 4.3, Fig. 7)

Annealing time [s] of 100 steps for N = 8192. The fastest block size of each
row is in bold.

| M | M2=32 | M2=64 | M2=128 | M2=256 | M2=512 | M2=1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.374 | 0.262 | 0.245 | **0.240** | 0.241 | 0.262 |
| 8 | 0.315 | 0.260 | 0.242 | 0.240 | **0.235** | 0.263 |
| 16 | 0.319 | 0.263 | 0.246 | 0.240 | **0.236** | 0.263 |
| 32 | 0.329 | 0.264 | 0.241 | 0.242 | **0.239** | 0.258 |
| 64 | 0.364 | 0.285 | 0.265 | **0.246** | 0.255 | 0.448 |
| 128 | 0.768 | 0.479 | 0.377 | 0.305 | **0.297** | 0.662 |
| 256 | 1.422 | 0.814 | 0.548 | **0.422** | 0.551 | 1.285 |
| 512 | 2.595 | 1.435 | 0.912 | **0.805** | 1.060 | 2.387 |

As in the paper, the time first falls as M2 grows and then rises again. On the
RTX A4000 the best block size is 256 or 512; the paper found 128 on an RTX 3080.
For M ≥ 128, M2 = 256 is 1.1x to 1.3x faster than the default of 128.

## Tensor Cores (Sec. 4.4, Fig. 8)

Annealing time [s] of 100 steps with Tensor Cores, with M2 = 128:

| N | M=4 | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 | M=512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.024 | 0.024 | 0.024 | 0.025 | 0.025 | 0.027 | 0.030 | 0.041 |
| 2048 | 0.054 | 0.054 | 0.055 | 0.055 | 0.058 | 0.061 | 0.075 | 0.110 |
| 4096 | 0.101 | 0.099 | 0.103 | 0.103 | 0.106 | 0.124 | 0.180 | 0.295 |
| 8192 | 0.242 | 0.236 | 0.238 | 0.236 | 0.257 | 0.371 | 0.545 | 0.912 |
| 16384 | 0.532 | 0.535 | 0.551 | 0.611 | 0.752 | 1.074 | 1.714 | 3.029 |
| 32768 | 1.317 | 1.364 | 1.484 | 1.760 | 2.283 | 3.470 | 5.879 | 10.989 |

Speedup of the whole annealing from Tensor Cores (time with `--no-tensor-cores`
divided by the time above):

| N | M=4 | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 | M=512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.02x | 1.04x | 1.16x | 1.16x | 1.20x | 1.17x | 1.24x | 1.26x |
| 2048 | 1.19x | 1.04x | 1.13x | 1.13x | 1.13x | 1.26x | 1.21x | 1.23x |
| 4096 | 1.01x | 1.02x | 1.25x | 1.13x | 1.19x | 1.22x | 1.23x | 1.22x |
| 8192 | 1.01x | 1.02x | 1.32x | 1.18x | 1.76x | 1.20x | 1.28x | 1.24x |
| 16384 | 1.03x | 1.19x | 1.21x | 1.14x | 1.12x | 1.23x | 1.29x | 1.35x |
| 32768 | 1.01x | 1.23x | 1.19x | 1.05x | 1.13x | 1.25x | 1.33x | 1.32x |

Fig. 8 of the paper measures only the local-field update (the GEMMs). These
times also include the flip kernel, which does not use the Tensor Cores, so the
speedup is smaller. As in the paper, the Tensor Cores help most for large M,
where the GEMMs dominate.
