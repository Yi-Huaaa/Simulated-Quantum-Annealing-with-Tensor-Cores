#!/usr/bin/env bash
# Downloads G-set MAX-CUT instances from https://web.stanford.edu/~yyye/yyye/Gset/
#
# Usage: scripts/download_gset.sh [G1 G22 ...]
#
# Without arguments, downloads the instances used in the paper's benchmarks.
# Files are stored in $GSET_DIR (default: data/gset).
set -euo pipefail

GSET_DIR=${GSET_DIR:-data/gset}
BASE_URL=https://web.stanford.edu/~yyye/yyye/Gset

if [[ $# -gt 0 ]]; then
    graphs=("$@")
else
    graphs=(G1 G22 G48 G65 G77 G81)
fi

mkdir -p "$GSET_DIR"
for g in "${graphs[@]}"; do
    if [[ -s "$GSET_DIR/$g" ]]; then
        echo "$GSET_DIR/$g already exists"
        continue
    fi
    echo "downloading $g -> $GSET_DIR/$g"
    curl -fsSL --retry 3 -o "$GSET_DIR/$g.part" "$BASE_URL/$g"
    mv "$GSET_DIR/$g.part" "$GSET_DIR/$g"
done
