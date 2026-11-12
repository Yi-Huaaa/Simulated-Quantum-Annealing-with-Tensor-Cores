#!/usr/bin/env bash
# Section 4.4 / Fig. 8 of the paper: annealing time with and without Tensor
# Cores for N = 1024 ... 32768 and M = 4 ... 512 (blk_sz = 128, 100 steps).
# Each size N uses one G-set graph padded with isolated spins up to N.
#
# Usage: scripts/run_tensor_cores.sh
# Settings: SIZES, TROTTERS, BLOCK, STEPS, RUNS (see the README).
set -euo pipefail
source "$(dirname "$0")/common.sh"

SIZES=${SIZES:-"1024 2048 4096 8192 16384 32768"}
TROTTERS=${TROTTERS:-"4 8 16 32 64 128 256 512"}
BLOCK=${BLOCK:-128}
STEPS=${STEPS:-100}
RUNS=${RUNS:-10}

require_solver
csv="$RESULTS_DIR/tensor_cores.csv"
init_csv "$csv"

for n in $SIZES; do
    graph=${GRAPH_FOR_SIZE[$n]:?no G-set graph configured for N = $n}
    require_graph "$graph"
    for m in $TROTTERS; do
        for tc in on off; do
            args=(--size "$n" --trotters "$m" --block "$BLOCK" --steps "$STEPS" --runs "$RUNS")
            [[ "$tc" == off ]] && args+=(--no-tensor-cores)
            row=$(run_case tensor_cores "N$n-M$m-tc_$tc" "${args[@]}" "$GSET_DIR/$graph")
            echo "$row" >> "$csv"
            echo "$row"
        done
    done
done

echo
python3 "$SCRIPT_DIR/summarize.py" tensor_cores "$csv"
