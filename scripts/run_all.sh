#!/usr/bin/env bash
# Runs every experiment: solution quality, block size and Tensor Cores.
set -euo pipefail
dir=$(dirname "$0")
"$dir/run_quality.sh"
"$dir/run_block_size.sh"
"$dir/run_tensor_cores.sh"
