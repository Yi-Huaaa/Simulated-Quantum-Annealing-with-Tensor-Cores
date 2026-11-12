# Shared settings and helpers of the experiment scripts. Source it, do not run it.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

SQA=${SQA:-$ROOT_DIR/build/sqa}
GSET_DIR=${GSET_DIR:-$ROOT_DIR/data/gset}
RESULTS_DIR=${RESULTS_DIR:-$ROOT_DIR/results/local}

CSV_HEADER="graph,nodes,edges,N,M,M2,steps,tensor_cores,runs,avg_time_s,avg_cut,best_cut"

# G-set instance that is padded to each problem size N of the speed experiments.
declare -A GRAPH_FOR_SIZE=([1024]=G1 [2048]=G22 [4096]=G48 [8192]=G65 [16384]=G77 [32768]=G81)

require_solver() {
    if [[ ! -x "$SQA" ]]; then
        echo "error: $SQA not found; run 'make' first" >&2
        exit 1
    fi
}

require_graph() {
    if [[ ! -s "$GSET_DIR/$1" ]]; then
        echo "error: $GSET_DIR/$1 is missing; run 'make gset'" >&2
        exit 1
    fi
}

# Writes the CSV header to $1 unless the file already exists.
init_csv() {
    mkdir -p "$(dirname "$1")"
    [[ -s "$1" ]] || echo "$CSV_HEADER" > "$1"
}

# run_case <experiment> <label> <solver arguments...>
# Runs the solver with --check, keeps its output in
# $RESULTS_DIR/logs/<experiment>/<label>.log and prints one CSV row (see
# CSV_HEADER). The check runs after the timed annealing loop, so it does not
# change the measured times; if it fails, the script stops.
run_case() {
    local experiment=$1 label=$2
    shift 2
    local log="$RESULTS_DIR/logs/$experiment/$label.log"
    mkdir -p "$(dirname "$log")"
    echo "[$experiment] $label: sqa --check $*" >&2
    if ! "$SQA" --check "$@" > "$log"; then
        echo "error: run failed, see $log" >&2
        exit 1
    fi
    parse_log "$log"
}

# Prints the CSV row of a solver log.
parse_log() {
    awk -F' *: *' '
        $1 == "graph" {
            split($2, f, /[ (,)]+/)
            name = f[1]; sub(/.*\//, "", name)
            nodes = f[2]; edges = f[4]
        }
        $1 == "spins (N)"     { n = $2 }
        $1 == "trotters (M)"  { m = $2 }
        $1 == "block (M2)"    { m2 = $2 }
        $1 == "steps"         { steps = $2 }
        $1 == "tensor cores"  { tc = $2 }
        $1 == "runs"          { runs = $2 }
        $1 == "avg time [s]"  { t = $2 }
        $1 == "avg cut"       { avg = $2 }
        $1 == "best cut"      { best = $2 }
        END { print name "," nodes "," edges "," n "," m "," m2 "," steps "," tc "," runs "," t "," avg "," best }
    ' "$1"
}
