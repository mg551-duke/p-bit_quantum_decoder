#!/usr/bin/env python3
"""Pool filtered_tune.sh logs: per-config tv=1..5 errors/shots/invalid -> LER table.

Usage: python filtered_tune_pool.py /tmp/ftune/<TAG>  [tv_to_sort_by=5]
"""
import sys, glob, re, math, os

RE_PAIR = re.compile(r'\((\d+)/(\d+)\)')
RE_TV   = re.compile(r'\btv=(\d+)')
RE_INV  = re.compile(r'\binvalid=(\d+)')

def parse(path):
    cfg = {k: [0, 0, 0] for k in range(1, 6)}  # tv -> [err, shots, invalid]
    name = os.path.splitext(os.path.basename(path))[0]
    done = False
    for line in open(path):
        if line.strip() == 'DONE':
            done = True
        if ' tv=' not in line or 'LER=' not in line:
            continue
        mtv = RE_TV.search(line); pr = RE_PAIR.findall(line)
        if not mtv or not pr:
            continue
        tv = int(mtv.group(1)); e, n = int(pr[0][0]), int(pr[0][1])
        inv = int(RE_INV.search(line).group(1)) if RE_INV.search(line) else 0
        c = cfg[tv]; c[0] += e; c[1] += n; c[2] += inv
    return name, cfg, done

def main():
    tag_dir = sys.argv[1]
    sort_tv = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = []
    for path in sorted(glob.glob(os.path.join(tag_dir, '*.log'))):
        name, cfg, done = parse(path)
        if cfg[sort_tv][1] == 0:
            status = 'NODATA' if done else 'RUNNING'
            rows.append((name, cfg, done, status)); continue
        rows.append((name, cfg, done, 'done' if done else 'partial'))
    # sort by tv=sort_tv LER (lower is better); no-data last
    def key(r):
        e, n, _ = r[1][sort_tv]
        return (e / n) if n else 9.99
    rows.sort(key=key)
    print(f"# filtered tune pool: {tag_dir}  (sorted by tv={sort_tv} LER)")
    hdr = f"{'config':14s} {'status':8s} {'shots':>8s}"
    for tv in range(1, 6):
        hdr += f"   tv{tv}_LER  (err)"
    print(hdr)
    for name, cfg, done, status in rows:
        n5 = cfg[sort_tv][1]
        line = f"{name:14s} {status:8s} {n5:>8d}"
        for tv in range(1, 6):
            e, n, inv = cfg[tv]
            ler = e / n if n else 0.0
            line += f"  {ler:.2e} ({e:>3d})"
        print(line)
    # best with a 95% upper-bound for context
    print()
    for name, cfg, done, status in rows[:3]:
        e, n, inv = cfg[sort_tv]
        if not n:
            continue
        ler = e / n; se = math.sqrt(e) / n if e else 0
        ul = (e + 1.64 * math.sqrt(e)) / n if e else 3.0 / n
        print(f"BEST tv{sort_tv}: {name:14s} LER={ler:.3e} +-{se:.1e} ul95={ul:.2e}  ({e}/{n}) inv={inv}")

if __name__ == '__main__':
    main()
