#!/usr/bin/env bash
# B1 code-capacity LER-vs-p sweep on the GPU engine, GPUs 1-7 in parallel (one p
# per GPU). For each p we climb a cheap -> expensive decoder ladder so the LER
# drops a meaningful step at a time. The LER here is convergence-limited (logic=0
# for B1), so the lever that matters is the annealing BUDGET = sweeps x max_legs;
# target_valid grows too (bigger valid-sample ensemble for min_weight).
#
#   ladder (name | sweeps | max_legs | target_valid):
#     L1_s300_l1    300  1  1     ~300    sweep-units  (cheapest single shallow pass)
#     L2_s800_l4    800  4  4     ~3200                (~10x budget)
#     L3_s1500_l10  1500 10 8     ~15000               (~5x)
#     L4_s3000_l24  3000 24 8     ~72000               (~5x; deep restart ensemble)
#
# Each (p, step) run stops at TARGET legacy errors or MAXSHOTS, whichever first,
# so high-p (high-LER) points finish fast and low-p deep points cap out with an
# upper bound. Reporting is the 2x2 grid (valid/invalid x correct/incorrect).
#
# Watch:  tail -f code_capacity/results/b1_psweep/DRIVER.log
#         code_capacity/.venv.../python code_capacity/psweep_report.py   (table)
set -u

ROOT=/home/mert/p-bit_quantum_decoder
PY=$ROOT/.venv-gpu/bin/python
CODE=$ROOT/code_capacity/codes/B1_HxHzLxLz.npz
OUT=$ROOT/code_capacity/results/b1_psweep
mkdir -p "$OUT"
DRV="$OUT/DRIVER.log"

TARGET=${TARGET:-40}
MAXSHOTS=${MAXSHOTS:-60000}
CHAINS=${CHAINS:-256}
RHO=${RHO:-0.35}
GPU_ROUND=${GPU_ROUND:-5000}
PS=${PS:-"0.04 0.05 0.06 0.07 0.08 0.09 0.10"}
# HARD RULE: never use GPU 6 or 7 on this machine -- allowed devices are 0-5 only.
# GPU 0 is reserved for the deep p06 run, so the sweep pool defaults to 1-5; the
# 7 p-values queue 2-per on two of them. 6/7 are also filtered out below as a guard.
GPUS=${GPUS:-"1 2 3 4 5"}

# cheap -> expensive ladder, one step per line: name|sweeps|max_legs|target_valid
LADDER="L1_s300_l1|300|1|1
L2_s800_l4|800|4|4
L3_s1500_l10|1500|10|8
L4_s3000_l24|3000|24|8"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$DRV"; }

run_ladder() { # $1=gpu  $2=p   (runs the full ladder for one p on one GPU)
  local gpu=$1 p=$2 name sw legs tv odir ler val
  while IFS='|' read -r name sw legs tv; do
    [ -z "$name" ] && continue
    odir="$OUT/p${p}/${name}"
    if [ -f "$odir/summary.csv" ]; then
      log "p=$p gpu=$gpu SKIP $name (already complete)"
      continue
    fi
    log "p=$p gpu=$gpu START $name (sweeps=$sw legs=$legs tv=$tv)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT/src $PY \
      $ROOT/code_capacity/run_code_capacity.py \
      --codes "$CODE" --p "$p" --device cuda --chains "$CHAINS" --sweeps "$sw" \
      --rho-scale "$RHO" --beta-start 0.5 --beta-stop 5.0 --shape linear \
      --max-legs "$legs" --target-valid "$tv" \
      --target-errors "$TARGET" --max-shots "$MAXSHOTS" --gpu-round "$GPU_ROUND" \
      --output-dir "$odir" --name "b1_p${p}_${name}" \
      > "$OUT/p${p}_${name}.log" 2>&1
    ler=$(grep -h "LER(logic+nonconv)" "$odir/RESULT.txt" 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')
    val=$(grep -hE "\[gpu\]" "$odir/progress.txt" 2>/dev/null | tail -1 | grep -oE "valid=[0-9.]+")
    log "p=$p gpu=$gpu DONE  $name -> $ler  $val"
  done <<< "$LADDER"
  log "p=$p gpu=$gpu LADDER COMPLETE"
}

# Belt-and-suspenders: drop any forbidden device (6/7) from the pool no matter what.
gpus=()
for g in $GPUS; do
  case "$g" in
    6|7) log "REFUSING forbidden GPU $g (only 0-5 allowed)";;
    *)   gpus+=("$g");;
  esac
done
ps=($PS)
ng=${#gpus[@]}
log "=== B1 p-sweep START  ps=[$PS] gpus=[${gpus[*]}] target=$TARGET maxshots=$MAXSHOTS chains=$CHAINS rho=$RHO ==="
# One worker per GPU; each GPU runs its assigned p-values (round-robin) SEQUENTIALLY
# through the full ladder, so #p-values may exceed #GPUs without oversubscribing.
for (( j=0; j<ng; j++ )); do
  gpu=${gpus[$j]}
  (
    k=$j
    while [ $k -lt ${#ps[@]} ]; do
      run_ladder "$gpu" "${ps[$k]}"
      k=$(( k + ng ))
    done
  ) &
done
wait
log "=== B1 p-sweep ALL COMPLETE ==="
"$PY" "$ROOT/code_capacity/psweep_report.py" >> "$DRV" 2>&1 || true
