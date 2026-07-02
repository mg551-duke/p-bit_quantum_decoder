#!/usr/bin/env python3
"""Collect the B1 LER-vs-p sweep into a ladder x p table.

Scans ``code_capacity/results/b1_psweep/p<p>/<step>/summary.csv`` (written by
run_code_capacity.py --device cuda) and prints, per ladder step, the legacy and
content LER at every p, plus the 2x2 grid (valid/invalid x correct/incorrect) for
the deepest step. Run any time while the sweep is in flight; missing cells show
as '-'.

Usage: python code_capacity/psweep_report.py [results_dir]
"""
import csv
import glob
import math
import os
import re
import sys

DEF_DIR = os.path.join(os.path.dirname(__file__), "results", "b1_psweep")


def _f(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def load(results_dir):
    """Return {step: {p: row}} and sorted (steps, ps)."""
    table = {}
    ps = set()
    steps = set()
    for path in glob.glob(os.path.join(results_dir, "p*", "*", "summary.csv")):
        parts = path.split(os.sep)
        p_dir, step = parts[-3], parts[-2]  # 'p0.06', 'L3_s1500_l10'
        m = re.match(r"p([0-9.]+)", p_dir)
        if not m:
            continue
        p = float(m.group(1))
        try:
            row = next(iter(csv.DictReader(open(path))))
        except StopIteration:
            continue
        table.setdefault(step, {})[p] = row
        ps.add(p)
        steps.add(step)
    # ladder steps sort by their leading L<number>
    def step_key(s):
        m = re.match(r"L(\d+)", s)
        return int(m.group(1)) if m else 99
    return table, sorted(steps, key=step_key), sorted(ps)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEF_DIR
    table, steps, ps = load(results_dir)
    if not table:
        print(f"# no results yet under {results_dir}")
        return

    print(f"# B1 LER-vs-p sweep: {results_dir}")
    print(f"# ladder steps (cheap->expensive): {', '.join(steps)}")

    for metric, key in (("LER(legacy=logic+nonconv)", "ler_legacy"),
                        ("LER(content=logic+nonconv_logic)", "ler_content")):
        print(f"\n== {metric} ==")
        header = f"{'step':14s}" + "".join(f"{('p='+format(p,'g')):>12s}" for p in ps)
        print(header)
        for step in steps:
            line = f"{step:14s}"
            for p in ps:
                row = table[step].get(p)
                if row is None:
                    line += f"{'-':>12s}"
                else:
                    ler = _f(row, key)
                    err = int(_f(row, "logic")) + (
                        int(_f(row, "nonconv")) if key == "ler_legacy"
                        else int(_f(row, "nonconv_logic")))
                    line += f"{ler:>9.2e}({err:>2d})" if err else f"{('<'+format(3.0/max(_f(row,'runs'),1),'.0e')):>12s}"
            print(line)

    # 2x2 grid for the deepest available step, per p
    deep = steps[-1]
    print(f"\n== 2x2 outcome grid @ deepest step {deep} ==")
    print(f"{'p':>6s} {'runs':>8s} {'valid_corr':>11s} {'valid_incorr':>12s} "
          f"{'inval_corr':>11s} {'inval_incorr':>12s} {'valid_frac':>11s}")
    for p in ps:
        row = table[deep].get(p)
        if row is None:
            continue
        runs = int(_f(row, "runs"))
        vc = int(_f(row, "valid_correct"))
        vi = int(_f(row, "valid_incorrect"))
        ic = int(_f(row, "invalid_correct"))
        ii = int(_f(row, "invalid_incorrect"))
        vf = _f(row, "valid_fraction", _f(row, "same") + _f(row, "degen"))
        print(f"{p:>6g} {runs:>8d} {vc:>11d} {vi:>12d} {ic:>11d} {ii:>12d} {vf:>11.5f}")


if __name__ == "__main__":
    main()
