#!/usr/bin/env bash
# Orchestrator: wait for the in-flight L3 low-p pool to finish, then launch the
# deep L4 pool (s3000 / l24) on the SAME GPUs (3,4,5) and SAME low-p points,
# running to 10 content (true-logical) errors each -- per the user's choice.
#
# L4 nests under b1_lowp/p<val>/L4_s3000_l24/ (no collision with L3_s1500_l10)
# and uses a distinct --seed-base so its error realizations are independent of
# L3's. It appends to the same DRIVER.log once it begins.
#
# Started detached:
#   setsid nohup bash code_capacity/run_l4_after_l3.sh >/dev/null 2>&1 &
# Watch:
#   tail -f code_capacity/results/b1_lowp/CHAIN_L4.log     (watcher + L4 stdout)
#   tail -f code_capacity/results/b1_lowp/DRIVER.log       (once L4 begins)
set -u

ROOT=/home/mert/p-bit_quantum_decoder
PY=$ROOT/.venv-gpu/bin/python
OUT=$ROOT/code_capacity/results/b1_lowp
LOG=$OUT/CHAIN_L4.log
L3_PID=${L3_PID:-1802097}

log(){ echo "[$(date '+%F %H:%M:%S')] $*" >> "$LOG"; }

log "watcher up (pid $$): waiting for L3 pool pid=$L3_PID to exit, then L4 on GPUs 3-5"
while kill -0 "$L3_PID" 2>/dev/null; do
  sleep 60
done
log "L3 pool pid=$L3_PID exited; settling 15s for GPUs to free"
sleep 15

log "launching L4 pool: --p 0.05 0.04 --gpus 3 4 5 --name L4_s3000_l24 (s3000 l24 tv8) target=10"
"$PY" "$ROOT/code_capacity/run_lowp_pool.py" \
  --p 0.05 0.04 --gpus 3 4 5 \
  --name L4_s3000_l24 --sweeps 3000 --max-legs 24 --target-valid 8 \
  --target-errors 10 --seed-base 7700000000 \
  --output-dir "$OUT" >> "$LOG" 2>&1
rc=$?
log "L4 pool returned rc=$rc; watcher done"
