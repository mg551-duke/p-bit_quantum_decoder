#!/usr/bin/env bash
# 50-error LER campaign at p=0.003, best-LER config. Each A100 runs sequential
# 10k-shot chunks with unique seeds; pooled until >=50 logical errors.
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
CFG="--chains 128 --sweeps 400 --max-legs 16 --target-valid 1 --rho-scales 0.3"
CHUNK=10000
NCHUNKS=20   # up to 1.6M shots total; we stop early once pooled errors >=50

for gpu in 0 1 2 3 4 5 6 7; do
  setsid bash -c "
    export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT
    for c in \$(seq 0 $((NCHUNKS-1))); do
      seed=\$(( $gpu*1000 + c ))
      $PY experiments/bench_circuit_pbit.py --device cuda --shots $CHUNK --seed \$seed $CFG
    done
    echo GPU${gpu}_ALL_DONE
  " > /tmp/camp_g$gpu.log 2>&1 &
done
echo "campaign launched on all 8 GPUs"
