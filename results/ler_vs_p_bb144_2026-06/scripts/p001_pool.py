#!/usr/bin/env python3
"""Pool the resume-safe p=0.001 campaign (scripts/p001_campaign.sh).

Reads logs/p001/<decoder>_g*.res, sums shots + per-tv errors/invalid across every
completed shard of each decoder, and prints LER, stderr, and a 95% one-sided upper
limit (normal approx, z=1.645; rule-of-three 3/N when zero errors) per tv -- plus
total shots, wall-throughput, and progress toward the 10-error target.
"""
import glob, os, re, math, sys

LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "p001")
LOGDIR = os.path.normpath(LOGDIR)
TVS = [1, 2, 3, 4, 5]
Z = 1.645
TARGET = 10

# tv=K LER=... (E/N) ... invalid=I ... <ms> ms/shot
LINE = re.compile(r"tv=(\d+)\s+LER=\S+\s+\((\d+)/(\d+)\)\s+valid=\S+\s+invalid=(\d+).*?([\d.]+)\s+ms/shot")


def pool(decoder):
    err = {k: 0 for k in TVS}
    inv = {k: 0 for k in TVS}
    shots = 0
    ms_acc = 0.0
    ms_w = 0
    shards = 0
    files = sorted(glob.glob(os.path.join(LOGDIR, f"{decoder}_g*.res")))
    for f in files:
        with open(f) as fh:
            for line in fh:
                m = LINE.search(line)
                if not m:
                    continue
                k, e, n, iv, ms = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                   int(m.group(4)), float(m.group(5)))
                if k not in TVS:
                    continue
                err[k] += e
                inv[k] += iv
                if k == 5:               # one tv=5 line per completed shard
                    shots += n
                    shards += 1
                    ms_acc += ms * n
                    ms_w += n
    return err, inv, shots, shards, (ms_acc / ms_w if ms_w else 0.0), files


def report(decoder, label):
    err, inv, shots, shards, ms, files = pool(decoder)
    if not files:
        print(f"\n=== {label} ===  (no logs yet)")
        return
    print(f"\n=== {label} ===  shards={shards}  N={shots:,}  ~{ms:.1f} ms/shot")
    print(f"  {'tv':>3} | {'errors':>7} | {'LER':>11} | {'stderr':>10} | {'ul95':>10} | {'invalid':>7}")
    for k in TVS:
        e, n = err[k], shots
        if n == 0:
            continue
        ler = e / n
        se = math.sqrt(ler * (1 - ler) / n) if n else 0.0
        ul = (3.0 / n) if e == 0 else (ler + Z * se)
        print(f"  {k:>3} | {e:>7} | {ler:>11.3e} | {se:>10.2e} | {ul:>10.2e} | {inv[k]:>7}")
    e5 = err[5]
    if e5 >= TARGET:
        print(f"  -> tv=5 TARGET MET ({e5} >= {TARGET} errors).")
    elif e5 > 0:
        need = int(shots / e5 * TARGET)
        print(f"  -> tv=5 has {e5} err; ~{need:,} shots projected for {TARGET} (need ~{need-shots:,} more).")
    else:
        print(f"  -> tv=5 has 0 err so far; 95% UL = {3.0/shots:.2e}.")


if __name__ == "__main__":
    print(f"# p=0.001 campaign pool  ({LOGDIR})")
    report("filtered", "FILTERED  (memory_Z, basis Z; rho=0.20 sweeps=750)")
    report("correlated", "CORRELATED (memory_X, full DEM; rho=0.30 sweeps=500)")
    stop = os.path.join(LOGDIR, "STOP")
    print(f"\n[STOP file {'PRESENT -> workers exiting' if os.path.exists(stop) else 'absent -> running'}]")
