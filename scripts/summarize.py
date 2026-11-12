#!/usr/bin/env python3
"""Prints the results of an experiment script as Markdown tables.

Usage: scripts/summarize.py {tensor_cores|block_size|quality} <results.csv>

If a configuration appears more than once in the CSV, its last row is used.
"""
import csv
import sys


def read_rows(path, key):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[key(row)] = row
    return list(rows.values())


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def tensor_cores(path):
    rows = read_rows(path, lambda r: (int(r["N"]), int(r["M"]), r["tensor_cores"]))
    time = {(int(r["N"]), int(r["M"]), r["tensor_cores"]): float(r["avg_time_s"]) for r in rows}
    sizes = sorted({n for n, _, _ in time})
    trotters = sorted({m for _, m, _ in time})
    steps = rows[0]["steps"]

    def cells(n, f):
        return [f(n, m) for m in trotters]

    def tc_time(n, m):
        t = time.get((n, m, "on"))
        return f"{t:.3f}" if t is not None else ""

    def speedup(n, m):
        on, off = time.get((n, m, "on")), time.get((n, m, "off"))
        return f"{off / on:.2f}x" if on and off else ""

    header = ["N"] + [f"M={m}" for m in trotters]
    print(f"Annealing time with Tensor Cores [s] ({steps} steps, mean of {rows[0]['runs']} runs)\n")
    print(table(header, [[n] + cells(n, tc_time) for n in sizes]))
    print("\nSpeedup from Tensor Cores (time on CUDA cores / time on Tensor Cores)\n")
    print(table(header, [[n] + cells(n, speedup) for n in sizes]))


def block_size(path):
    rows = read_rows(path, lambda r: (int(r["M"]), int(r["M2"])))
    time = {(int(r["M"]), int(r["M2"])): float(r["avg_time_s"]) for r in rows}
    trotters = sorted({m for m, _ in time})
    blocks = sorted({b for _, b in time})
    print(f"Annealing time [s] for N = {rows[0]['N']}, {rows[0]['steps']} steps, mean of {rows[0]['runs']} runs "
          "(fastest block size in bold)\n")
    body = []
    for m in trotters:
        fastest = min((time[(m, b)] for b in blocks if (m, b) in time), default=None)
        line = [m]
        for b in blocks:
            t = time.get((m, b))
            line.append("" if t is None else (f"**{t:.3f}**" if t == fastest else f"{t:.3f}"))
        body.append(line)
    print(table(["M"] + [f"M2={b}" for b in blocks], body))


def quality(path):
    rows = read_rows(path, lambda r: r["graph"])
    rows.sort(key=lambda r: int(r["graph"].lstrip("G")))
    body, avg_ratios, best_ratios = [], [], []
    for r in rows:
        known = float(r["best_known"])
        avg, best = float(r["avg_cut"]), float(r["best_cut"])
        avg_ratios.append(avg / known)
        best_ratios.append(best / known)
        body.append([r["graph"], r["nodes"], r["edges"], f"{known:g}", f"{avg:g}", f"{best:g}",
                     f"{100 * avg / known:.2f}%", f"{100 * best / known:.2f}%"])
    r0 = rows[0]
    print(f"Cuts with M = {r0['M']}, M2 = {r0['M2']}, {r0['steps']} steps, {r0['runs']} runs per graph\n")
    print(table(["Graph", "Nodes", "Edges", "Best known", "Avg. cut", "Best cut", "Avg. / known", "Best / known"],
                body))
    n = len(rows)
    print(f"\nMean over {n} graphs: average cut {100 * sum(avg_ratios) / n:.2f}%, "
          f"best cut {100 * sum(best_ratios) / n:.2f}% of the best known cut "
          f"(lowest: {100 * min(avg_ratios):.2f}% and {100 * min(best_ratios):.2f}%)")


def main():
    reports = {"tensor_cores": tensor_cores, "block_size": block_size, "quality": quality}
    if len(sys.argv) != 3 or sys.argv[1] not in reports:
        sys.exit(__doc__.strip())
    reports[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
