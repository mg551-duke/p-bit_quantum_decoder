#!/usr/bin/env bash
# Sharpen the B1 code-capacity p=0.04 LER for the two cheap ladder steps (L1, L2),
# which only had 5 and 1 content errors in the fixed-budget sweep grid.
#
# Pools fresh, disjoint-seed shots across GPUs 3,4,5 (NEVER 6/7) until >=10 pooled
# CONTENT errors are collected for each step. L1 runs first (fast), then L2.
# Writes to results/b1_lowp_l1l2/DRIVER.log (separate from the L3/L4 pool log).
set -uo pipefail

ROOT=/home/mert/p-bit_quantum_decoder
PY="$ROOT/.venv-gpu/bin/python"
POOL="$ROOT/code_capacity/run_lowp_pool.py"
OUT="$ROOT/code_capacity/results/b1_lowp_l1l2"
GPUS="3 4 5"
TARGET=10
mkdir -p "$OUT"

echo "[$(date '+%F %T')] L1/L2 p=0.04 sharpen START gpus=[$GPUS] target=$TARGET" >> "$OUT/CHAIN.log"

# --- L1: sweeps=300, max_legs=1, target_valid=1 ---
"$PY" "$POOL" --p 0.04 --gpus $GPUS --target-errors $TARGET \
  --name L1_s300_l1 --sweeps 300 --max-legs 1 --target-valid 1 \
  --chains 256 --rho-scale 0.35 --beta-start 0.5 --beta-stop 5.0 --shape linear \
  --gpu-round 2000 --seed-base 20000000000 --output-dir "$OUT"

echo "[$(date '+%F %T')] L1 done; starting L2" >> "$OUT/CHAIN.log"

# --- L2: sweeps=800, max_legs=4, target_valid=4 ---
"$PY" "$POOL" --p 0.04 --gpus $GPUS --target-errors $TARGET \
  --name L2_s800_l4 --sweeps 800 --max-legs 4 --target-valid 4 \
  --chains 256 --rho-scale 0.35 --beta-start 0.5 --beta-stop 5.0 --shape linear \
  --gpu-round 2000 --seed-base 30000000000 --output-dir "$OUT"

echo "[$(date '+%F %T')] L1/L2 p=0.04 sharpen ALL DONE" >> "$OUT/CHAIN.log"
