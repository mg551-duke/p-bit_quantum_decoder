#!/usr/bin/env python3
"""Pool tune288_filtered.sh logs -> per-config tv=1..5 LER + invalid + meanW table.

Ranks by TRUE (content) LER -- logically-incorrect shots only; non-convergent
but logically-trivial shots are NOT failures -- ascending, with ms/shot (latency)
as the tiebreaker, then invalid-rate.  strict LER is shown for reference only.

Usage: python tune288_pool.py /tmp/ftune288/<TAG> [sort_tv=5]
"""
import sys, glob, re, math, os

RE_TV = re.compile(r'\btv=(\d+)')
RE_PAIR = re.compile(r'\((\d+)/(\d+)\)')
RE_INV = re.compile(r'\binvalid=(\d+)')
RE_W = re.compile(r'meanW=([\d.]+)')
RE_MS = re.compile(r'([\d.]+)\s*ms/shot')
RE_STRICT = re.compile(r'LER_strict=[\d.eE+-]+\s*\((\d+)/(\d+)\)')


def parse(path):
    cfg = {k: [0, 0, 0, 0] for k in range(1, 6)}  # tv -> [err, shots, invalid, strict_err]
    w = ms = None
    done = False
    for line in open(path):
        if line.strip() == 'DONE':
            done = True
        if ' tv=' not in line or 'LER=' not in line:
            continue
        mtv = RE_TV.search(line)
        pr = RE_PAIR.findall(line)
        if not mtv or not pr:
            continue
        tv = int(mtv.group(1))
        e, n = int(pr[0][0]), int(pr[0][1])
        inv = int(RE_INV.search(line).group(1)) if RE_INV.search(line) else 0
        ms_strict = RE_STRICT.search(line)
        se = int(ms_strict.group(1)) if ms_strict else e
        c = cfg[tv]
        c[0] += e
        c[1] += n
        c[2] = inv
        c[3] += se
        mw = RE_W.search(line)
        mm = RE_MS.search(line)
        if mw:
            w = float(mw.group(1))
        if mm:
            ms = float(mm.group(1))
    return os.path.splitext(os.path.basename(path))[0], cfg, w, ms, done


def main():
    d = sys.argv[1]
    stv = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = [parse(p) for p in sorted(glob.glob(os.path.join(d, '*.log')))]

    def key(r):
        e, n, inv, se = r[1][stv]
        ler = e / n if n else 9.9
        ms = r[3] if r[3] is not None else 9e9
        invr = inv / n if n else 9.9
        return (ler, ms, invr)

    rows.sort(key=key)
    print(f"# tune288 pool {d}  (sorted by tv={stv}: TRUE/content LER, then ms/shot, then invalid-rate)")
    print(f"{'config':16s} {'st':3s} {'shots':>7s} {'meanW':>7s} {'ms/sh':>6s}  "
          f"{'trueLER':>9s} (err,inv)  {'strictLER':>9s} (serr)")
    for name, cfg, w, ms, done in rows:
        e5, n5, i5, se5 = cfg[stv]
        ler5 = e5 / n5 if n5 else 0.0
        sler5 = se5 / n5 if n5 else 0.0
        st = 'ok' if done else 'run'
        ws = f"{w:7.1f}" if w is not None else "    n/a"
        mss = f"{ms:6.1f}" if ms is not None else "   n/a"
        print(f"{name:16s} {st:3s} {n5:>7d} {ws} {mss}  {ler5:9.2e} ({e5},i{i5})  {sler5:9.2e} ({se5})")
    print()
    for name, cfg, w, ms, done in rows[:3]:
        e, n, inv, se = cfg[stv]
        if not n:
            continue
        ler = e / n
        ul = (3.0 if e == 0 else (e + 1.64 * math.sqrt(e))) / n
        mss = f"{ms:.1f}" if ms is not None else "n/a"
        print(f"BEST tv{stv}: {name:16s} trueLER={ler:.3e} ul95={ul:.2e} ({e}/{n}) inv={inv} ms/shot={mss} meanW={w}")


if __name__ == '__main__':
    main()
