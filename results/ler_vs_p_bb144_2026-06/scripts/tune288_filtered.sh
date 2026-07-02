#!/usr/bin/env bash
# Beta/rho optimization for FILTERED decoding (memory_Z + --basis-filter Z) of the
# bb288 [[288,12,18]] d=18 circuit-level problem at p=0.003.  Mirrors filtered_tune.sh.
#
# Grid: --rho-scales x --beta-stop  (beta-start fixed), one bench run per config with
# --capture-targets 1,2,3,4,5 so every t_v is scored from a single trajectory.
# Throttled to ONE config per GPU per wave (no GPU memory oversubscription).
# Logs -> /tmp/ftune288/<TAG>/<name>.log (trailing 'DONE').  Pool with tune288_pool.py.
#
# Run detached so it survives the launching shell:
#   TAG=beta_rho setsid bash tune288_filtered.sh > /tmp/ftune288/launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD=/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate
P=${P:-0.003}; SHOTS=${SHOTS:-16000}; TAG=${TAG:-beta_rho}; SEED=${SEED:-2000}
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}; BSTART=${BSTART:-0.5}
RHOS=${RHOS:-"0.15 0.20 0.25 0.30"}
BSTOPS=${BSTOPS:-"6 8 12 16"}
GPUS=${GPUS:-"0 1 2"}; read -r -a G <<< "$GPUS"; NG=${#G[@]}
CIRC="$TD/circuit=bicycle_bivariate_288_12_18_memory_Z,distance=18,rounds=18,error_rate=$P,noise_model=uniform_circuit,basis=CX,A=x^3+y^2+y^7,B=y^3+x+x^2.stim"
[ -f "$CIRC" ] || { echo "MISSING circuit: $CIRC"; exit 2; }
OUT=/tmp/ftune288/$TAG; mkdir -p "$OUT"
echo "tune288 tag=$TAG p=$P shots=$SHOTS sweeps=$SWEEPS chains=$CHAINS legs=$LEGS bstart=$BSTART rhos='$RHOS' bstops='$BSTOPS' gpus='$GPUS' start $(date)" | tee "$OUT/META.txt"

slot=0
for rho in $RHOS; do
  for bs in $BSTOPS; do
    name="rho${rho}_bs${bs}"
    gpu=${G[$(( slot % NG ))]}
    seed=$(( SEED + slot ))
    ( echo "### name=$name p=$P shots=$SHOTS rho=$rho beta=$BSTART->$bs sweeps=$SWEEPS legs=$LEGS seed=$seed";
      cd "$ROOT" && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/bench_circuit_pbit.py \
        --circuit "$CIRC" --basis-filter Z --device cuda --shots "$SHOTS" --seed "$seed" \
        --chains "$CHAINS" --sweeps "$SWEEPS" --max-legs "$LEGS" --selection min_weight --gamma 1.0 \
        --beta-start "$BSTART" --beta-stop "$bs" --rho-scales "$rho" --capture-targets 1,2,3,4,5;
      echo DONE ) > "$OUT/$name.log" 2>&1 &
    echo "launched $name on GPU $gpu (seed $seed)"
    slot=$(( slot + 1 ))
    if [ $(( slot % NG )) -eq 0 ]; then wait; fi   # one config per GPU per wave
  done
done
wait
echo "tune288 ALL $slot configs done $(date)" | tee -a "$OUT/META.txt"
