#!/usr/bin/env bash
# Accuracy-focused grid at p=0.003. One config per A100 (parallel), SAME seed so
# every config decodes identical shots (paired comparison). 10000 shots each.
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
SHOTS=10000

launch() { # gpu "label" args...
  local gpu="$1"; local label="$2"; shift 2
  setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
    echo '### $label'; $PY experiments/bench_circuit_pbit.py --device cuda --shots $SHOTS --seed 0 $*; echo DONE" \
    > /tmp/acc_g$gpu.log 2>&1 &
}

launch 0 "strong"        --chains 128 --sweeps 400 --max-legs 12 --target-valid 4 --rho-scales 0.6
launch 1 "more_sweeps"   --chains 128 --sweeps 600 --max-legs 16 --target-valid 4 --rho-scales 0.6
launch 2 "more_chains"   --chains 256 --sweeps 400 --max-legs 12 --target-valid 4 --rho-scales 0.6
launch 3 "more_coset"    --chains 128 --sweeps 400 --max-legs 12 --target-valid 8 --rho-scales 0.6
launch 4 "rho_high"      --chains 128 --sweeps 400 --max-legs 12 --target-valid 4 --rho-scales 1.0
launch 5 "rho_low"       --chains 128 --sweeps 400 --max-legs 12 --target-valid 4 --rho-scales 0.4
launch 6 "valid_push"    --chains 128 --sweeps 600 --max-legs 20 --target-valid 1 --rho-scales 0.6
launch 7 "kitchen_sink"  --chains 256 --sweeps 600 --max-legs 16 --target-valid 8 --rho-scales 0.6
echo "8 configs launched"
