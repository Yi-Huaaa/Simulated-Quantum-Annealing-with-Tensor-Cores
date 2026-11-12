#!/usr/bin/env bash
# Section 4.5 / Fig. 9 of the paper: solution quality on the G-set graphs of
# Table 2, relative to their best known cuts, with 1000 annealing steps.
#
# The paper tuned the schedule per experiment. Here a single setting is used
# for every graph: M = 64 and beta rising to 64 at the end, which gave the best
# cuts of the M and beta0/beta1 values we tried.
#
# Usage: scripts/run_quality.sh [G43 G48 ...]
# Settings: TROTTERS, BLOCK, STEPS, RUNS, plus extra solver options in OPTIONS;
# see the README.
set -euo pipefail
source "$(dirname "$0")/common.sh"

TROTTERS=${TROTTERS:-64}
BLOCK=${BLOCK:-128}
STEPS=${STEPS:-1000}
RUNS=${RUNS:-10}
OPTIONS=${OPTIONS:---beta1 64}

# Best known cuts, as listed in Table 2 of the paper.
declare -A BEST_KNOWN=(
    [G5]=11631 [G9]=2054 [G13]=582 [G18]=992 [G19]=906 [G20]=941 [G21]=931
    [G26]=13328 [G31]=3309 [G34]=1384 [G38]=7688 [G39]=2408 [G41]=2405 [G42]=2481
    [G43]=6660 [G44]=6650 [G47]=6657 [G48]=6000 [G49]=6000 [G50]=5880
    [G51]=3848 [G53]=3850 [G54]=3852 [G63]=27045
)
ALL_GRAPHS="G5 G9 G13 G18 G19 G20 G21 G26 G31 G34 G38 G39 G41 G42 G43 G44 G47 G48 G49 G50 G51 G53 G54 G63"

if [[ $# -gt 0 ]]; then
    graphs=("$@")
else
    read -r -a graphs <<< "$ALL_GRAPHS"
fi

require_solver
csv="$RESULTS_DIR/quality.csv"
mkdir -p "$RESULTS_DIR"
[[ -s "$csv" ]] || echo "$CSV_HEADER,best_known" > "$csv"

for g in "${graphs[@]}"; do
    best_known=${BEST_KNOWN[$g]:?no best known cut listed for $g}
    require_graph "$g"
    # shellcheck disable=SC2086
    row=$(run_case quality "$g" --trotters "$TROTTERS" --block "$BLOCK" --steps "$STEPS" --runs "$RUNS" \
          $OPTIONS "$GSET_DIR/$g")
    echo "$row,$best_known" >> "$csv"
    echo "$row,$best_known"
done

echo
python3 "$SCRIPT_DIR/summarize.py" quality "$csv"
