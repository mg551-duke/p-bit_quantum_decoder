#!/usr/bin/env python3
"""Pool auto_deep_sweeps logs into an LER-vs-sweeps curve + emit an extend/plateau decision.

Reads <out>/s<NNNN>.log (one bench run per sweep depth, all sharing one fixed seed so
the curve is paired / common-random-numbers), parses tv=1..5 (err/shots/invalid),
prints a table sorted by sweeps, writes <out>/curve.csv, and prints ONE final
machine-readable line that the driver consumes:

    DECISION EXTEND <s1> [s2]        # deepest point still significantly improving
    DECISION PLATEAU <sweeps> <ler>  # floor reached (no significant further gain)
    DECISION WAIT                    # fewer than 2 finished points

"Significant" = the deepest point's LER drop vs the next-deepest exceeds both
--min-rel (relative) and --sig * SE_of_difference (Poisson), i.e. the gain is real
at the current shot statistics rather than noise.

Usage:
    python deep_sweeps_report.py --out /tmp/ftune/autodeep [--tv 5]
                                 [--max-sweeps 24000] [--min-rel 0.03] [--sig 1.0]
"""
import argparse
import glob
import math
import os
import re

RE_PAIR = re.compile(r"\((\d+)/(\d+)\)")
RE_TV = re.compile(r"\btv=(\d+)")
RE_INV = re.compile(r"\binvalid=(\d+)")
RE_SW = re.compile(r"(\d+)")


def parse(path):
    """Return (sweeps, {tv: [err, shots, invalid]}, done)."""
    cfg = {k: [0, 0, 0] for k in range(1, 6)}
    name = os.path.splitext(os.path.basename(path))[0]
    m = RE_SW.search(name)
    sweeps = int(m.group(1)) if m else -1
    done = False
    for line in open(path):
        if line.strip() == "DONE":
            done = True
        if " tv=" not in line or "LER=" not in line:
            continue
        mtv = RE_TV.search(line)
        pr = RE_PAIR.findall(line)
        if not mtv or not pr:
            continue
        tv = int(mtv.group(1))
        e, n = int(pr[0][0]), int(pr[0][1])
        inv = int(RE_INV.search(line).group(1)) if RE_INV.search(line) else 0
        if 1 <= tv <= 5:
            c = cfg[tv]
            c[0] += e
            c[1] += n
            c[2] += inv
    return sweeps, cfg, done


def nice(x):
    """Round to a tidy sweep count (nearest 500 below 4k, nearest 1000 above)."""
    step = 500 if x < 4000 else 1000
    return int(round(x / step) * step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tv", type=int, default=5)
    ap.add_argument("--max-sweeps", type=int, default=24000)
    ap.add_argument("--min-rel", type=float, default=0.03)
    ap.add_argument("--sig", type=float, default=1.0)
    a = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(a.out, "s*.log")):
        sweeps, cfg, done = parse(path)
        if sweeps >= 0:
            rows.append((sweeps, cfg, done))
    rows.sort(key=lambda r: r[0])

    # ---- table -----------------------------------------------------------
    print(f"# deep-sweep LER curve: {a.out}  (sorted by sweeps; decision tv={a.tv})")
    hdr = f"{'sweeps':>7s} {'status':8s} {'shots':>7s}"
    for tv in range(1, 6):
        hdr += f"   tv{tv}_LER (err)"
    print(hdr)
    for sweeps, cfg, done in rows:
        n = cfg[a.tv][1]
        st = "done" if done else ("partial" if n else "running")
        line = f"{sweeps:>7d} {st:8s} {n:>7d}"
        for tv in range(1, 6):
            e, nn, _ = cfg[tv]
            ler = e / nn if nn else 0.0
            line += f"  {ler:.2e} ({e:>3d})"
        print(line)

    # ---- csv -------------------------------------------------------------
    csv_path = os.path.join(a.out, "curve.csv")
    with open(csv_path, "w") as f:
        cols = ["sweeps", "shots", "done"]
        for tv in range(1, 6):
            cols += [f"tv{tv}_err", f"tv{tv}_ler"]
        f.write(",".join(cols) + "\n")
        for sweeps, cfg, done in rows:
            n = cfg[a.tv][1]
            vals = [str(sweeps), str(n), "1" if done else "0"]
            for tv in range(1, 6):
                e, nn, _ = cfg[tv]
                vals += [str(e), f"{(e / nn if nn else 0.0):.6e}"]
            f.write(",".join(vals) + "\n")

    # ---- decision (on finished points, at the chosen tv) -----------------
    pts = []
    for sweeps, cfg, done in rows:
        e, n, _ = cfg[a.tv]
        if done and n > 0:
            pts.append((sweeps, e / n, e, n))
    pts.sort(key=lambda x: x[0])

    if len(pts) < 2:
        print("DECISION WAIT")
        return

    best = min(pts, key=lambda x: x[1])
    (sa, la, ea, na), (sb, lb, eb, nb) = pts[-2], pts[-1]
    diff = la - lb
    rel = diff / la if la else 0.0
    se = math.sqrt(ea / na**2 + eb / nb**2)  # SE of the LER difference (Poisson)
    significant = diff > a.sig * se

    print(
        f"# best tv{a.tv}: sweeps={best[0]} LER={best[1]:.3e} ({best[2]}/{best[3]})"
    )
    print(
        f"# deepest pair s{sa}->s{sb}: dLER={diff:+.2e} rel={rel:+.1%} "
        f"se={se:.2e} significant={significant}"
    )

    tested = {p[0] for p in pts}
    nxt = []
    for cand in (nice(sb * 1.5), nice(sb * 2.0)):
        cand = min(cand, a.max_sweeps)
        if cand > sb and cand not in tested and cand not in nxt:
            nxt.append(cand)

    keep_going = (
        best[0] == sb
        and rel > a.min_rel
        and significant
        and sb < a.max_sweeps
        and nxt
    )
    if keep_going:
        print("DECISION EXTEND " + " ".join(str(x) for x in nxt))
    else:
        print(f"DECISION PLATEAU {best[0]} {best[1]:.3e}")


if __name__ == "__main__":
    main()
