#!/usr/bin/env python3
"""Pool-to-target campaign for B1 low-p points with a chosen ladder decoder.

Shards each p across the given GPUs (0-5 ONLY -- NEVER 6/7). Each shard is an
independent ``run_code_capacity.py --device cuda`` process with a disjoint
``--base-seed`` so error realizations never overlap and the shots pool into one
i.i.d. sample. The controller polls each shard's ``progress.txt``, sums the
CONTENT (true-logical) errors = ``valid_incorrect + invalid_incorrect`` across
shards, and once the POOLED count reaches ``--target-errors`` it terminates that
p's shards, writes a pooled RESULT, and moves on. p-values run sequentially, all
GPUs per p, so the slowest (lowest) p gets full parallelism.

Example (the L3 mid decoder on GPUs 3-5, >=10 true-logical errors each):
  setsid nohup .venv-gpu/bin/python code_capacity/run_lowp_pool.py \
      --p 0.05 0.04 --gpus 3 4 5 --target-errors 10 &
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-gpu" / "bin" / "python"
CODE = ROOT / "code_capacity" / "codes" / "B1_HxHzLxLz.npz"
HARNESS = ROOT / "code_capacity" / "run_code_capacity.py"

_FIELDS = ("runs", "valid_correct", "valid_incorrect", "invalid_correct", "invalid_incorrect")
_RE = {k: re.compile(rf"{k}=(\d+)") for k in _FIELDS}


def parse_progress(path: Path):
    """Cumulative (runs, vc, vi, ic, ii) from the last [gpu] progress line, else zeros."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return (0, 0, 0, 0, 0)
    for line in reversed(lines):
        if "[gpu] runs=" not in line:
            continue
        try:
            return tuple(int(_RE[k].search(line).group(1)) for k in _FIELDS)
        except (AttributeError, ValueError):
            return (0, 0, 0, 0, 0)
    return (0, 0, 0, 0, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", nargs="+", type=float, required=True)
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--target-errors", type=int, default=10)
    ap.add_argument("--name", default="L3_s1500_l10")
    ap.add_argument("--sweeps", type=int, default=1500)
    ap.add_argument("--max-legs", type=int, default=10)
    ap.add_argument("--target-valid", type=int, default=8)
    ap.add_argument("--chains", type=int, default=256)
    ap.add_argument("--rho-scale", type=float, default=0.35)
    ap.add_argument("--beta-start", type=float, default=0.5)
    ap.add_argument("--beta-stop", type=float, default=5.0)
    ap.add_argument("--shape", default="linear")
    ap.add_argument("--gpu-round", type=int, default=2000)
    ap.add_argument("--max-shots-shard", type=int, default=50_000_000)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=770_000_000)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "code_capacity" / "results" / "b1_lowp")
    args = ap.parse_args()

    # HARD RULE: only CUDA devices 0-5 are ever allowed; refuse 6/7 outright.
    gpus = []
    for g in args.gpus:
        if g in (6, 7) or not (0 <= g <= 5):
            print(f"REFUSING forbidden/invalid GPU {g} (only 0-5 allowed)", flush=True)
            continue
        gpus.append(g)
    if not gpus:
        raise SystemExit("no allowed GPUs (need a subset of 0-5)")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    drv = out / "DRIVER.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with drv.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"=== B1 low-p pool START p={args.p} gpus={gpus} step={args.name} "
        f"(s={args.sweeps} l={args.max_legs} tv={args.target_valid} chains={args.chains} "
        f"rho={args.rho_scale}) target_content_errors={args.target_errors} ===")

    for p in args.p:
        ptag = int(round(p * 100))
        pdir = out / f"p{p:g}" / args.name
        procs, shard_dirs = [], []
        for k, g in enumerate(gpus):
            sd = pdir / f"g{g}"
            sd.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(sd)
            base_seed = args.seed_base + ptag * 1_000_000_000 + k * 100_000_000
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=str(ROOT / "src"))
            cmd = [
                str(PY), str(HARNESS), "--codes", str(CODE), "--p", f"{p:g}",
                "--device", "cuda", "--chains", str(args.chains), "--sweeps", str(args.sweeps),
                "--rho-scale", str(args.rho_scale), "--beta-start", str(args.beta_start),
                "--beta-stop", str(args.beta_stop), "--shape", args.shape,
                "--max-legs", str(args.max_legs), "--target-valid", str(args.target_valid),
                "--target-metric", "logical_content", "--target-errors", str(10 ** 9),
                "--max-shots", str(args.max_shots_shard), "--gpu-round", str(args.gpu_round),
                "--base-seed", str(base_seed), "--output-dir", str(sd),
                "--name", f"b1_lowp_p{p:g}_g{g}",
            ]
            fh = (sd / "shard.log").open("w")
            procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          start_new_session=True))
            log(f"p={p:g} launched shard gpu={g} base_seed={base_seed} -> {sd}")

        while True:
            time.sleep(args.poll)
            tot = [0, 0, 0, 0, 0]
            alive = 0
            for sd, pr in zip(shard_dirs, procs):
                tot = [a + b for a, b in zip(tot, parse_progress(sd / "progress.txt"))]
                if pr.poll() is None:
                    alive += 1
            runs, vc, vi, ic, ii = tot
            content = vi + ii
            if runs:
                log(f"p={p:g} pooled runs={runs} content={content}/{args.target_errors} "
                    f"(vi={vi} ii={ii}) valid_frac={vc / runs:.5f} alive={alive}")
            else:
                log(f"p={p:g} pooled runs=0 alive={alive}")
            if content >= args.target_errors or alive == 0:
                break

        for pr in procs:
            if pr.poll() is None:
                try:
                    os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for pr in procs:
            try:
                pr.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        tot = [0, 0, 0, 0, 0]
        for sd in shard_dirs:
            tot = [a + b for a, b in zip(tot, parse_progress(sd / "progress.txt"))]
        runs, vc, vi, ic, ii = tot
        content = vi + ii
        legacy = vi + ic + ii
        lerc = content / runs if runs else float("nan")
        lerc_se = (math.sqrt(content) / runs) if (runs and content) else 0.0
        lerl = legacy / runs if runs else float("nan")
        res = (
            f"=== B1 p={p:g} POOLED {args.name} (gpus={gpus}) ===\n"
            f"  runs={runs}  valid_correct={vc} valid_incorrect={vi} "
            f"invalid_correct={ic} invalid_incorrect={ii}  "
            f"valid_frac={vc / runs if runs else float('nan'):.5f}\n"
            f"  content errors (true logical = logic+nonconv_logic) = {content}\n"
            f"  LER(content) = {lerc:.3e} +- {lerc_se:.1e}\n"
            f"  LER(legacy)  = {lerl:.3e}\n"
        )
        (pdir / "POOLED_RESULT.txt").write_text(res, encoding="utf-8")
        log(f"p={p:g} DONE pooled runs={runs} content={content} "
            f"LER(content)={lerc:.3e} +- {lerc_se:.1e}")

    log("=== B1 low-p pool ALL COMPLETE ===")


if __name__ == "__main__":
    main()
