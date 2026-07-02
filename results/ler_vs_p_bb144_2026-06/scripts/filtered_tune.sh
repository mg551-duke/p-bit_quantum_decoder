#!/usr/bin/env bash
# filtered_tune.sh -- hyperparameter sweep on the FILTERED memory_Z DEM (basis Z),
# one config per GPU, to push the p-bit decoder toward the correlated/full-DEM
# and RelayBP-300sets reference on bb144 [[144,12]] d=12.
#
# Each config runs ONE bench invocation with --capture-targets 1,2,3,4,5 so every
# target_valid is scored from a single run. Configs are launched detached (setsid),
# round-robin across GPUs; poll the logs for "DONE" then pool with filtered_tune_pool.py.
#
# Usage:
#   P=0.004 SHOTS=24000 TAG=phase1 SEED0=1000 bash filtered_tune.sh configs/ftune_phase1.txt
#
# Config file: one config per non-comment line, "name | <extra bench flags>".
#   e.g.   rho030 | --chains 128 --sweeps 500 --rho-scales 0.30 --gamma 1.0
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
CIRCDIR=/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate
P=${P:-0.004}; SHOTS=${SHOTS:-24000}; TAG=${TAG:-tune}; SEED0=${SEED0:-1000}; NGPU=${NGPU:-8}
COMMON="--selection min_weight --max-legs 24"
CAP="--capture-targets 1,2,3,4,5"
CFG="${1:?usage: filtered_tune.sh <configfile>}"
CIRC="$CIRCDIR/circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=$P,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
OUT=/tmp/ftune/$TAG; mkdir -p "$OUT"
[ -f "$CIRC" ] || { echo "missing circuit: $CIRC"; exit 1; }
echo "tag=$TAG p=$P shots=$SHOTS out=$OUT" | tee "$OUT/META.txt"

i=0
while IFS='|' read -r name flags; do
  name="$(echo "$name" | xargs)"
  case "$name" in ''|\#*) continue;; esac
  gpu=$(( i % NGPU )); seed=$(( SEED0 + i ))
  echo "$name | $flags" >> "$OUT/META.txt"
  setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
    echo '### name=$name p=$P shots=$SHOTS seed=$seed flags=$flags';
    $PY experiments/bench_circuit_pbit.py --circuit \"$CIRC\" --basis-filter Z \
      --device cuda --shots $SHOTS --seed $seed $COMMON $CAP $flags;
    echo DONE" > "$OUT/$name.log" 2>&1 &
  echo "launched $name on gpu $gpu seed $seed"
  i=$(( i + 1 ))
done < "$CFG"
echo "total $i configs launched; poll: grep -l DONE $OUT/*.log | wc -l"
