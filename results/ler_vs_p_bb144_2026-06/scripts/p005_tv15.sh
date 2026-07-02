#!/usr/bin/env bash
# Fill the missing p=0.005 tv=1..5 point for the LER-vs-p figure, using the
# EXACT production config as the tv-sweep campaign (chains=128 sweeps=500
# rho=0.3 max_legs=24, capture tv=1..5 from one trajectory). 6 GPU shards, one
# per GPU 0-5; pool the 6 logs afterwards. High LER at p=0.005 (~5e-3) so a
# modest shot count yields plenty of errors at every tv.
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
circ="$TD/circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
COMMON="--chains 128 --sweeps 500 --rho-scales 0.3 --device cuda --max-legs 24 --capture-targets 1,2,3,4,5"
OUT=/tmp/tvsweep
ch=3000
rm -f "$OUT"/p0.005_hires.done
for gpu in 0 1 2 3 4 5; do
  s=$(( 90000000 + gpu ))
  setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
    $PY experiments/bench_circuit_pbit.py $COMMON --shots $ch --seed $s --circuit \"$circ\"" \
    > "$OUT/p0.005_hires_g$gpu.log" 2>&1 &
done
wait
echo "P005 DONE N=$((6*ch)) @ $(date)" > "$OUT/p0.005_hires.done"
