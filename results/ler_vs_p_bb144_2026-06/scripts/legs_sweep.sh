#!/usr/bin/env bash
# Paired target_valid / max_legs sweep at p=0.005.
# Same seeds (9000000+gpu) => identical shots across all configs, so error
# counts are directly comparable (paired). Everything else fixed:
#   chains=128 sweeps=500 rho_scale=0.3 gibbs min_weight beta 0.5->8.0
# Each config runs one shard per GPU (0-5); configs run sequentially so each
# gets all 6 GPUs. Results are pooled into /tmp/SWEEP_RESULT.txt after each.
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
C5="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,error_rate=0.005,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
COMMON="--chains 128 --sweeps 500 --rho-scales 0.3 --device cuda"
SHOTS=2500
RESULT=/tmp/SWEEP_RESULT.txt

names=(A_tv1_ml12 B_tv4_ml12 C_tv8_ml24)
declare -A KNOBS=(
  [A_tv1_ml12]="--max-legs 12 --target-valid 1"
  [B_tv4_ml12]="--max-legs 12 --target-valid 4"
  [C_tv8_ml24]="--max-legs 24 --target-valid 8"
)

pool() {  # $1 = config name -> append pooled line to RESULT
  local name="$1"
  grep -hE "rho_scale=" /tmp/SW_${name}_g*.log 2>/dev/null | awk -v cfg="$name" '
    {
      for (i=1;i<=NF;i++) {
        if ($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevLER) { split(substr($i,2,length($i)-2),a,"/"); E+=a[1]; N+=a[2]; prevLER=0 }
        if ($i ~ /^LER=/) prevLER=1
        if ($i ~ /^invalid=/) INV+=substr($i,9)+0
        if ($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevStrict) { split(substr($i,2,length($i)-2),b,"/"); ES+=b[1]; prevStrict=0 }
        if ($i ~ /^LER_strict=/) prevStrict=1
        if ($i ~ /^wall/) { w=$i; gsub(/[^0-9]/,"",w); WALL+=w+0 }
        if ($i ~ /^legs=/)    { L+=substr($i,6)+0; nc++ }
        if ($i ~ /^sw_exec=/) { SW+=substr($i,9)+0 }
      }
    }
    END {
      ler=(N? E/N:0); se=(N? sqrt(E)/N:0);
      printf "%-12s LER=%.3e +-%.1e  errors=%d/%d  invalid=%d  strict=%d  legs=%.2f sw_exec=%.0f  cost=%dGPU-s (%.0f ms/shot)\n",
             cfg, ler, se, E, N, INV, ES, (nc?L/nc:0), (nc?SW/nc:0), WALL, (N?1000.0*WALL/N:0)
    }'
}

echo "paired tv/legs sweep @ p=0.005  shots/config=$((SHOTS*6))  $(date)" > "$RESULT"
for name in "${names[@]}"; do
  rm -f /tmp/SW_${name}_g*.log
  for gpu in 0 1 2 3 4 5; do
    seed=$((9000000 + gpu))
    setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
      $PY experiments/bench_circuit_pbit.py $COMMON --shots $SHOTS --seed $seed \
        --circuit \"$C5\" ${KNOBS[$name]}" > "/tmp/SW_${name}_g$gpu.log" 2>&1 &
  done
  wait
  pool "$name" >> "$RESULT"
  echo "done $name @ $(date)"
done
echo "ALLDONE @ $(date)" >> "$RESULT"
