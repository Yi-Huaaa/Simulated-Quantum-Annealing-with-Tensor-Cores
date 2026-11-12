#!/usr/bin/env bash
# Downloads G-set MAX-CUT instances from https://web.stanford.edu/~yyye/yyye/Gset/
#
# Usage: scripts/download_gset.sh [G1 G22 ...]
#
# Without arguments, downloads every instance used by the experiment scripts:
# the graphs of the speed experiments and the graphs of Table 2 in the paper.
# Files are stored in $GSET_DIR (default: data/gset).
set -euo pipefail

GSET_DIR=${GSET_DIR:-data/gset}
BASE_URL=https://web.stanford.edu/~yyye/yyye/Gset

if [[ $# -gt 0 ]]; then
    graphs=("$@")
else
    graphs=(G1 G22 G48 G65 G77 G81
            G5 G9 G13 G18 G19 G20 G21 G26 G31 G34 G38 G39 G41 G42
            G43 G44 G47 G49 G50 G51 G53 G54 G63)
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
