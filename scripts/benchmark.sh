#!/usr/bin/env bash
# Speed / quality sweep over problem sizes and Trotter numbers, following the
# setup of the paper: each size N is paired with a G-set instance that is
# padded with isolated spins up to N.
#
# Usage: scripts/benchmark.sh [path/to/sqa]
#
# Environment variables (defaults in brackets):
#   SIZES     problem sizes N          [1024 2048 4096 8192 16384 32768]
#   TROTTERS  Trotter numbers M        [4 8 16 32 64 128 256 512]
#   BLOCKS    HU block sizes M_2       [128]
#   RUNS      runs per configuration   [5]
#   STEPS     Monte Carlo steps        [100]
#   EXTRA     extra solver options     []
#   GSET_DIR  G-set directory          [data/gset]
#   OUT       output CSV               [results/benchmark.csv]
set -euo pipefail

SQA=${1:-build/sqa}
SIZES=${SIZES:-"1024 2048 4096 8192 16384 32768"}
TROTTERS=${TROTTERS:-"4 8 16 32 64 128 256 512"}
BLOCKS=${BLOCKS:-128}
RUNS=${RUNS:-5}
STEPS=${STEPS:-100}
EXTRA=${EXTRA:-}
GSET_DIR=${GSET_DIR:-data/gset}
OUT=${OUT:-results/benchmark.csv}

declare -A GRAPH=([1024]=G1 [2048]=G22 [4096]=G48 [8192]=G65 [16384]=G77 [32768]=G81)

if [[ ! -x "$SQA" ]]; then
    echo "error: '$SQA' not found; run 'make' first" >&2
    exit 1
fi

# Extracts "<key> : <value>" from the solver's summary in $out.
value() { awk -F: -v key="$1" '$1 ~ "^"key { gsub(/ /, "", $2); print $2 }' <<< "$out"; }

mkdir -p "$(dirname "$OUT")"
echo "graph,N,M,M2,steps,runs,avg_time_s,avg_cut,best_cut" > "$OUT"

for n in $SIZES; do
    graph=${GRAPH[$n]:-}
    if [[ -z "$graph" ]]; then
        echo "error: no G-set instance configured for N = $n" >&2
        exit 1
    fi
    if [[ ! -s "$GSET_DIR/$graph" ]]; then
        echo "error: $GSET_DIR/$graph is missing; run 'make gset'" >&2
        exit 1
    fi
    for m in $TROTTERS; do
        for b in $BLOCKS; do
            # shellcheck disable=SC2086
            out=$("$SQA" --size "$n" --trotters "$m" --block "$b" --steps "$STEPS" --runs "$RUNS" $EXTRA \
                  "$GSET_DIR/$graph")
            line="$graph,$n,$m,$b,$STEPS,$RUNS,$(value "avg time"),$(value "avg cut"),$(value "best cut")"
            echo "$line" | tee -a "$OUT"
        done
    done
done

echo "results written to $OUT"
