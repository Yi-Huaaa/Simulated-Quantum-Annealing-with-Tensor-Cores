#!/usr/bin/env bash
# Section 4.3 / Fig. 7 of the paper: annealing time for different HU block
# sizes blk_sz (M2) and Trotter numbers M, with N = 8192 and 100 steps.
#
# Usage: scripts/run_block_size.sh
# Settings: SIZE, TROTTERS, BLOCKS, STEPS, RUNS (see the README).
set -euo pipefail
source "$(dirname "$0")/common.sh"

SIZE=${SIZE:-8192}
TROTTERS=${TROTTERS:-"4 8 16 32 64 128 256 512"}
BLOCKS=${BLOCKS:-"32 64 128 256 512 1024"}
STEPS=${STEPS:-100}
RUNS=${RUNS:-5}

require_solver
graph=${GRAPH_FOR_SIZE[$SIZE]:?no G-set graph configured for N = $SIZE}
require_graph "$graph"
csv="$RESULTS_DIR/block_size.csv"
init_csv "$csv"

for m in $TROTTERS; do
    for b in $BLOCKS; do
        row=$(run_case block_size "N$SIZE-M$m-M2_$b" --size "$SIZE" --trotters "$m" --block "$b" \
              --steps "$STEPS" --runs "$RUNS" "$GSET_DIR/$graph")
        echo "$row" >> "$csv"
        echo "$row"
    done
done

echo
python3 "$SCRIPT_DIR/summarize.py" block_size "$csv"
