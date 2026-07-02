#!/usr/bin/env bash
# Autonomous p=0.002 LER campaign. Runs by itself on GPUs 0-5 (NEVER 6/7),
# pools errors, and auto-stops + writes /tmp/p002_RESULT.txt once >=10 errors.
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
C2="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,error_rate=0.002,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
CFG="--chains 128 --sweeps 500 --max-legs 12 --target-valid 1 --rho-scales 0.3"
CHUNK=20000
TARGET_ERRORS=10
GPUS="0 1 2 3 4 5"
LOGDIR=/tmp/p002
RESULT=/tmp/p002_RESULT.txt
mkdir -p "$LOGDIR"; rm -f "$LOGDIR"/g*.log "$LOGDIR"/progress.txt "$RESULT"

# --- launch one worker per GPU (sequential chunks, unique seeds) ---
for gpu in $GPUS; do
  setsid bash -c "
    export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT
    for c in \$(seq 0 99999); do
      seed=\$(( 2000000 + $gpu*100000 + c ))
      $PY experiments/bench_circuit_pbit.py --device cuda --shots $CHUNK --seed \$seed --circuit \"$C2\" $CFG || true
    done
  " > "$LOGDIR/g${gpu}.log" 2>&1 &
done
echo "$(date '+%F %T') launched workers on GPUs $GPUS" >> "$LOGDIR/progress.txt"

# --- monitor: pool errors, stop at target ---
while true; do
  sleep 120
  read e n < <(cat "$LOGDIR"/g*.log 2>/dev/null | grep -oE '\([0-9]+/[0-9]+\)' | tr -d '()' \
               | awk -F/ '{e+=$1; n+=$2} END{print e+0, n+0}')
  echo "$(date '+%F %T') errors=$e shots=$n" >> "$LOGDIR/progress.txt"
  if [ "${e:-0}" -ge "$TARGET_ERRORS" ]; then
    pkill -f 'error_rate=0.002' 2>/dev/null
    for g in $GPUS; do
      for pid in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill "$pid" 2>/dev/null
      done
    done
    ler=$(awk -v e="$e" -v n="$n" 'BEGIN{ if(n>0) printf "%.3e +- %.2e", e/n, sqrt((e/n)*(1-e/n)/n); else print "NA"}')
    minv=$(cat "$LOGDIR"/g*.log 2>/dev/null | grep -oE 'valid=[0-9.]+' | sed 's/valid=//' | sort -n | head -1)
    {
      echo "p=0.002 campaign COMPLETE  $(date '+%F %T')"
      echo "errors=$e  shots=$n  LER=$ler  min_valid=${minv:-NA}"
      echo "config: chains=128 sweeps=500 max_legs=12 target_valid=1 rho_scale=0.3"
    } > "$RESULT"
    echo "$(date '+%F %T') DONE -> $RESULT" >> "$LOGDIR/progress.txt"
    break
  fi
done
