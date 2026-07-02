#!/usr/bin/env bash
# Low-rho sweep at p=0.003 (rho is the dominant min-weight/LER lever). Paired
# (seed 0), 10000 shots, chains=128 sweeps=500 max_legs=16 target_valid=4.
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
COMMON="--chains 128 --sweeps 500 --max-legs 16 --target-valid 4"
run() { # gpu rho
  local gpu="$1"; local rho="$2"
  setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
    echo '### rho=$rho'; $PY experiments/bench_circuit_pbit.py --device cuda --shots 10000 --seed 0 $COMMON --rho-scales $rho; echo DONE" \
    > /tmp/rho_g$gpu.log 2>&1 &
}
run 0 0.20
run 1 0.25
run 2 0.30
run 3 0.35
run 5 0.45
run 6 0.50
echo "rho sweep launched on 0,1,2,3,5,6"
