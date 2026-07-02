#!/usr/bin/env bash
# p=0.001 LER campaign for BOTH decoders, resume-safe, interruptible.
#   filtered   = memory_Z + --basis-filter Z (936 det / 8784 mech), rho=0.20, sweeps=750
#   correlated = memory_X full DEM (1728 det / 67824 mech), rho=0.30, sweeps=500
# Both capture tv=1..5 in one run (snapshot) and use mid-leg compaction (default ON).
#
# Allocation (override via ASSIGN): GPU0->filtered, GPU1->correlated, GPU2->correlated.
# Each worker loops fixed-size shards; every completed shard appends its 5 tv lines to
#   logs/p001/<decoder>_g<gpu>.res  and bumps a persisted seed counter <decoder>_g<gpu>.ctr,
# so killing the driver (or a worker) and re-launching simply CONTINUES with fresh seeds --
# no shot is ever re-counted and none is lost except a single in-flight shard.
#
# Stop politely:  touch logs/p001/STOP        (workers exit after the current shard)
# Hard stop:      kill -- -$(cat logs/p001/PGID)    (kills the whole process group)
# Monitor:        .venv-gpu/bin/python scripts/p001_pool.py     (or .venv/bin/python)
#
# HARD RULE: only GPUs in {0,1,2,3,4,5} are ever allowed; 6 and 7 are refused.
set -u
ROOT=/home/mert/p-bit_quantum_decoder
PY=$ROOT/.venv-gpu/bin/python
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
SUF="distance=12,rounds=12,error_rate=0.001,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
C_FILT="$TD/circuit=bicycle_bivariate_144_12_12_memory_Z,$SUF"
C_CORR="$TD/circuit=bicycle_bivariate_144_12_12_memory_X,$SUF"
LOGDIR=$ROOT/results/ler_vs_p_bb144_2026-06/logs/p001
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/STOP"

# decoder -> bench args (everything except --shots/--seed/--circuit)
ARGS_filtered="--device cuda --chains 128 --sweeps 750 --max-legs 24 --rho-scales 0.20 --gamma 1.0 --selection min_weight --basis-filter Z --capture-targets 1,2,3,4,5"
ARGS_correlated="--device cuda --chains 128 --sweeps 500 --max-legs 24 --rho-scales 0.30 --gamma 1.0 --selection min_weight --capture-targets 1,2,3,4,5"
CIRC_filtered="$C_FILT"
CIRC_correlated="$C_CORR"
# shots per shard (~6 min each): filtered ~9.8 ms/shot, correlated ~96 ms/shot
CHUNK_filtered=${CHUNK_FILTERED:-40000}
CHUNK_correlated=${CHUNK_CORRELATED:-4000}
# disjoint seed bases so no two workers ever share a seed
declare -A SEEDBASE=( [g0]=100000000 [g1]=200000000 [g2]=300000000 [g3]=400000000 [g4]=500000000 [g5]=600000000 )

ASSIGN=${ASSIGN:-"0:filtered 1:correlated 2:correlated"}

worker(){
  local gpu="$1" dec="$2"
  local name="${dec}_g${gpu}"
  local res="$LOGDIR/${name}.res" ctr="$LOGDIR/${name}.ctr"
  local argsv="ARGS_${dec}" circv="CIRC_${dec}" chunkv="CHUNK_${dec}"
  local args="${!argsv}" circ="${!circv}" chunk="${!chunkv}"
  local base=${SEEDBASE[g${gpu}]}
  [ -f "$ctr" ] || echo 0 > "$ctr"
  echo "# worker $name start $(date) chunk=$chunk base=$base args: $args" >> "$res"
  while [ ! -f "$LOGDIR/STOP" ]; do
    local k; k=$(cat "$ctr"); local seed=$((base + k))
    local out
    out=$(cd "$ROOT" && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu $PY experiments/bench_circuit_pbit.py \
          $args --shots "$chunk" --seed "$seed" --circuit "$circ" 2>&1)
    echo $((k+1)) > "$ctr"                       # advance seed even on failure (no infinite reuse)
    echo "$out" | tail -25 > "$LOGDIR/${name}.last"
    if echo "$out" | grep -q "tv=5 "; then
      { echo "## shard k=$k seed=$seed shots=$chunk $(date)"; echo "$out" | grep "tv="; } >> "$res"
    else
      echo "## SHARD_FAIL k=$k seed=$seed $(date) (see ${name}.last)" >> "$res"
    fi
  done
  echo "# worker $name stopped $(date)" >> "$res"
}

# ---- launch ----
echo $$ > "$LOGDIR/PGID"                          # this script's PID == process-group leader
echo "campaign start $(date) assign='$ASSIGN'" >> "$LOGDIR/driver.log"
pids=()
for tok in $ASSIGN; do
  gpu="${tok%%:*}"; dec="${tok##*:}"
  case "$gpu" in 6|7) echo "REFUSING GPU $gpu (forbidden)"; continue;; esac
  case "$dec" in filtered|correlated) ;; *) echo "unknown decoder '$dec'"; continue;; esac
  worker "$gpu" "$dec" &
  pids+=($!)
  echo "launched $dec on GPU $gpu (pid $!)" | tee -a "$LOGDIR/driver.log"
done
wait "${pids[@]}"
echo "campaign all workers exited $(date)" >> "$LOGDIR/driver.log"
