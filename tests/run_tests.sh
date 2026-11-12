#!/usr/bin/env bash
# End-to-end tests on small graphs whose maximum cut is known.
#
# Usage: tests/run_tests.sh [path/to/sqa]
#
# Every run uses --check, which fails if the incrementally updated local
# fields drift from a full recomputation of J * sigma.
set -euo pipefail

SQA=${1:-build/sqa}
if [[ ! -x "$SQA" ]]; then
    echo "error: '$SQA' not found; run 'make' first" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# 2D torus (periodic grid) with even side length: bipartite, max cut = #edges.
torus() {
    awk -v L="$1" 'BEGIN {
        print L * L, 2 * L * L
        for (r = 0; r < L; r++) for (c = 0; c < L; c++) {
            v = r * L + c + 1
            print v, r * L + (c + 1) % L + 1, 1
            print v, ((r + 1) % L) * L + c + 1, 1
        }
    }'
}

# Complete graph K_n with weight w: max cut = floor(n/2) * ceil(n/2) * w for
# w > 0, and 0 (no edge cut) for w < 0.
complete() {
    awk -v n="$1" -v w="$2" 'BEGIN {
        print n, n * (n - 1) / 2
        for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) print i, j, w
    }'
}

# Complete bipartite graph K_{a,b}: max cut = a * b.
bipartite() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        print a + b, a * b
        for (i = 1; i <= a; i++) for (j = 1; j <= b; j++) print i, a + j, 1
    }'
}

torus 16 > "$tmp/torus16"        # 256 nodes, max cut 512
torus 30 > "$tmp/torus30"        # 900 nodes, max cut 1800
complete 40 1 > "$tmp/k40"       # max cut 400
complete 41 1 > "$tmp/k41"       # max cut 420
complete 40 -1 > "$tmp/k40neg"   # max cut 0
bipartite 37 60 > "$tmp/k37_60"  # max cut 2220

passed=0
failed=0

expect_cut() {
    local expected=$1 graph=$2
    shift 2
    local out best
    if ! out=$("$SQA" --check --runs 3 "$@" "$tmp/$graph" 2>&1); then
        echo "FAIL  $graph $*: solver exited with an error"
        echo "$out" | sed 's/^/      /'
        failed=$((failed + 1))
        return
    fi
    best=$(awk -F: '/^best cut/ { gsub(/ /, "", $2); print $2 }' <<< "$out")
    if [[ "$best" == "$expected" ]]; then
        echo "ok    $graph $* (cut $best)"
        passed=$((passed + 1))
    else
        echo "FAIL  $graph $*: best cut $best, expected $expected"
        failed=$((failed + 1))
    fi
}

expect_exit() {
    local expected=$1
    shift
    local status=0
    "$SQA" "$@" > /dev/null 2>&1 || status=$?
    if [[ "$status" == "$expected" ]]; then
        echo "ok    exit status $status for: $*"
        passed=$((passed + 1))
    else
        echo "FAIL  exit status $status (expected $expected) for: $*"
        failed=$((failed + 1))
    fi
}

echo "== known optima"
expect_cut 512 torus16
expect_cut 1800 torus30
expect_cut 400 k40
expect_cut 420 k41
expect_cut 0 k40neg
expect_cut 2220 k37_60

echo "== Trotter slices (including odd M and plain SA with M = 1)"
for m in 1 2 3 4 7 32 128; do
    expect_cut 512 torus16 --trotters "$m"
done

echo "== block sizes"
for b in 32 64 128 256 512 1024; do
    expect_cut 400 k40 --block "$b"
done

echo "== padding, CUDA cores, schedule options"
expect_cut 512 torus16 --size 1024
expect_cut 2220 k37_60 --block 32 --size 2048 --trotters 8
expect_cut 1800 torus30 --no-tensor-cores
expect_cut 400 k40 --steps 200 --gamma0 4 --beta0 0.1 --beta1 8 --seed 42

echo "== invalid input"
expect_exit 2 --block 48 "$tmp/k40"
expect_exit 2 --trotters 0 "$tmp/k40"
expect_exit 2 --beta1 -1 "$tmp/k40"
expect_exit 2
expect_exit 1 --size 32 --block 32 "$tmp/k40"
expect_exit 1 --size 100 --block 64 "$tmp/k40"
expect_exit 1 "$tmp/does-not-exist"
printf '3 2\n1 2 1\n' > "$tmp/truncated"
expect_exit 1 "$tmp/truncated"
printf '3 1\n1 4 1\n' > "$tmp/bad-node"
expect_exit 1 "$tmp/bad-node"

echo
echo "$passed passed, $failed failed"
[[ "$failed" == 0 ]]
