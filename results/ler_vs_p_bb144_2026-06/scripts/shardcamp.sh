#!/usr/bin/env bash
# Reusable sharded campaign. Args: TAG CIRCUIT RHO CHUNK SEEDBASE GPUS...
# Each listed GPU runs sequential CHUNK-shot sub-runs (unique seeds) of the
# best-LER config; pool /tmp/${TAG}_g*.log until >=50 errors then kill.
# NOTE: only GPUs 0-5 may be used (never 6/7).
TAG="$1"; CIRCUIT="$2"; RHO="$3"; CHUNK="$4"; SEEDBASE="$5"; shift 5; GPUS="$*"
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
CFG="${SC_CFG:---chains 128 --sweeps 400 --max-legs 16 --target-valid 1}"
for gpu in $GPUS; do
  setsid bash -c "
    export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT
    for c in \$(seq 0 499); do
      seed=\$(( $SEEDBASE + $gpu*1000 + c ))
      $PY experiments/bench_circuit_pbit.py --device cuda --shots $CHUNK --seed \$seed --rho-scales $RHO --circuit \"$CIRCUIT\" $CFG
    done
    echo ${TAG}_g${gpu}_DONE
  " > /tmp/${TAG}_g${gpu}.log 2>&1 &
done
echo "launched $TAG on GPUs: $GPUS"
