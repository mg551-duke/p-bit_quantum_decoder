#!/usr/bin/env bash
# Beta/rho optimization for the rotated SURFACE CODE d=11 memory_X circuit-level
# problem at p=0.003.  The rotated surface code memory experiment has a single
# logical observable, so the DEM is already single-basis -> NO --basis-filter.
# Mirrors tune288_filtered.sh (same grid shape, ranking, capture-targets).
#
# Grid: --rho-scales x --beta-stop (beta-start fixed); one bench run per config with
# --capture-targets 1,2,3,4,5 so every t_v is scored from a single trajectory.
# One config per GPU per wave (no GPU memory oversubscription).
# Logs -> /tmp/sctune/<TAG>/<name>.log (trailing 'DONE').  Pool with tune288_pool.py.
#
# Detached launch:
#   TAG=beta_rho SHOTS=60000 setsid bash tune_sc11.sh > /tmp/sctune/launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD=/home/mert/CascadeProjects/relay/tests/testdata/surface
P=${P:-0.003}; SHOTS=${SHOTS:-60000}; TAG=${TAG:-beta_rho}; SEED=${SEED:-2000}
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}; BSTART=${BSTART:-0.5}
RHOS=${RHOS:-"0.15 0.20 0.25 0.30"}
BSTOPS=${BSTOPS:-"6 8 12 16"}
GPUS=${GPUS:-"0 1 2 3 4 5"}
# HARD GUARD: never use GPU 6 or 7 (user constraint).
read -r -a _g <<< "$GPUS"; G=()
for x in "${_g[@]}"; do [ "$x" = "6" ] && continue; [ "$x" = "7" ] && continue; G+=("$x"); done
NG=${#G[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (6/7 excluded)"; exit 2; }
CIRC="$TD/circuit=rotated_surface_code_memory_X,distance=11,rounds=11,error_rate=$P,noise_model=uniform_circuit,basis=CX.stim"
[ -f "$CIRC" ] || { echo "MISSING circuit: $CIRC"; exit 2; }
OUT=/tmp/sctune/$TAG; mkdir -p "$OUT"
echo "tune_sc11 tag=$TAG p=$P shots=$SHOTS sweeps=$SWEEPS chains=$CHAINS legs=$LEGS bstart=$BSTART rhos='$RHOS' bstops='$BSTOPS' gpus='${G[*]}' start $(date)" | tee "$OUT/META.txt"

slot=0
for rho in $RHOS; do
  for bs in $BSTOPS; do
    name="rho${rho}_bs${bs}"
    gpu=${G[$(( slot % NG ))]}
    seed=$(( SEED + slot ))
    ( echo "### name=$name p=$P shots=$SHOTS rho=$rho beta=$BSTART->$bs sweeps=$SWEEPS legs=$LEGS seed=$seed";
      cd "$ROOT" && PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/bench_circuit_pbit.py \
        --circuit "$CIRC" --device cuda --shots "$SHOTS" --seed "$seed" \
        --chains "$CHAINS" --sweeps "$SWEEPS" --max-legs "$LEGS" --selection min_weight --gamma 1.0 \
        --beta-start "$BSTART" --beta-stop "$bs" --rho-scales "$rho" --capture-targets 1,2,3,4,5;
      echo DONE ) > "$OUT/$name.log" 2>&1 &
    echo "launched $name on GPU $gpu (seed $seed)"
    slot=$(( slot + 1 ))
    if [ $(( slot % NG )) -eq 0 ]; then wait; fi   # one config per GPU per wave
  done
done
wait
echo "tune_sc11 ALL $slot configs done $(date)" | tee -a "$OUT/META.txt"
