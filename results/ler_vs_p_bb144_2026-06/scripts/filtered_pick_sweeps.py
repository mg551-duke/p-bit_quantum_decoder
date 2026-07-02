#!/usr/bin/env python3
"""Auto-select the sweeps "knee" from a filtered_tune phase dir.

Groups logs by their --sweeps value, pools tv=5 errors/shots, and picks the
CHEAPEST sweeps whose LER is statistically tied with the best (within 1 combined
stderr). Prints the chosen sweeps (int) to stdout; rationale to stderr.

Usage: python filtered_pick_sweeps.py /tmp/ftune/phase3 [tv=5]
"""
import sys, glob, re, os, math

RE_PAIR = re.compile(r'\((\d+)/(\d+)\)')
RE_TV   = re.compile(r'\btv=(\d+)')
RE_SW   = re.compile(r'\bsweeps=(\d+)')

def main():
    d = sys.argv[1]
    tv = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    groups = {}  # sweeps -> [err, shots]
    for f in sorted(glob.glob(os.path.join(d, '*.log'))):
        sweeps = None
        rows = []
        for line in open(f):
            m = RE_SW.search(line)
            if m and sweeps is None and 'shots=' in line:
                sweeps = int(m.group(1))
            if ' tv=' in line and 'LER=' in line:
                mtv = RE_TV.search(line); pr = RE_PAIR.findall(line)
                if mtv and pr and int(mtv.group(1)) == tv:
                    rows.append((int(pr[0][0]), int(pr[0][1])))
        if sweeps is None or not rows:
            continue
        e, n = rows[-1]
        g = groups.setdefault(sweeps, [0, 0])
        g[0] += e; g[1] += n
    if not groups:
        print(1000); print("no data; defaulting sweeps=1000", file=sys.stderr); return
    stats = {}
    for sw, (e, n) in groups.items():
        ler = e / n if n else 9.9
        se = math.sqrt(e) / n if (e and n) else (1.0 / n if n else 9.9)
        stats[sw] = (ler, se, e, n)
    best_sw = min(stats, key=lambda s: stats[s][0])
    best_ler, best_se, _, _ = stats[best_sw]
    # cheapest sweeps tied with best within 1 combined sigma
    chosen = best_sw
    for sw in sorted(stats):
        ler, se, e, n = stats[sw]
        comb = math.sqrt(se * se + best_se * best_se)
        if ler - best_ler <= comb:
            chosen = sw
            break
    print(chosen)
    print(f"sweeps knee analysis (tv={tv}):", file=sys.stderr)
    for sw in sorted(stats):
        ler, se, e, n = stats[sw]
        mark = " <- BEST" if sw == best_sw else (" <- CHOSEN" if sw == chosen else "")
        print(f"  sweeps={sw:5d}  LER={ler:.3e} +-{se:.1e}  ({e}/{n}){mark}", file=sys.stderr)
    if chosen != best_sw:
        print(f"  chose {chosen} (cheapest within 1sigma of best {best_sw})", file=sys.stderr)

if __name__ == '__main__':
    main()
