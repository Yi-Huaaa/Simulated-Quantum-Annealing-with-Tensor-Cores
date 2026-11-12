#!/usr/bin/env bash
# Runs every experiment: solution quality, block size and Tensor Cores.
set -euo pipefail
dir=$(dirname "$0")
echo "start $(date)"
echo "commit $(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
"$dir/run_quality.sh"
"$dir/run_block_size.sh"
"$dir/run_tensor_cores.sh"
echo "end $(date)"
