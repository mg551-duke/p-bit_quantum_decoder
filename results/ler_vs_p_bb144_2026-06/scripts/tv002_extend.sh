#!/usr/bin/env bash
# p=0.002 high-resolution EXTENSION of the tv-sweep campaign (+20M shots).
#
# Continues into /tmp/tvsweep with NEW round numbers (>= last+1), reusing the
# EXACT production config and the original seed formula. The original campaign
# used seed = 50000000 + pidx*1000000 + round*10 + gpu with pidx=2 for p=0.002,
# and reached round 35 (max seed 52000355). Continuing the same formula from
# round 36 yields seeds >= 52000360, disjoint from all existing seeds, so the
# new shots are independent and pool cleanly with the existing ~2.04M.
#
# Does NOT touch the original RESULT.txt / progress.txt: writes its own
# extend_progress.txt and RESULT_extended.txt. The pooled final p=0.002 table
# (existing + new) is what lands in RESULT_extended.txt.
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
SUF="distance=12,rounds=12,REP,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
# EXACT production config (identical to tv_sweep_campaign.sh).
COMMON="--chains 128 --sweeps 500 --rho-scales 0.3 --device cuda --max-legs 24 --capture-targets 1,2,3,4,5"
p=0.002
pidx=2          # matches original seed scheme base = 50000000 + pidx*1000000
ch=10000        # shots per shard (6 shards/round => 60000 shots/round)
ADD=20000000    # additional shots to collect on top of whatever already exists

OUT=/tmp/tvsweep
PROG="$OUT/extend_progress.txt"
RESULT="$OUT/RESULT_extended.txt"
circ="$TD/circuit=bicycle_bivariate_144_12_12_memory_X,${SUF/REP/error_rate=$p}"
[ -f "$circ" ] || { echo "MISSING circuit: $circ" >> "$PROG"; exit 1; }

# Identical pooling logic to the original campaign (restart-safe).
pool_round() {  # $1=round  -> updates E[], INV[], N
  local rnd="$1" gpu f
  for gpu in 0 1 2 3 4 5; do
    f="$OUT/p${p}_r${rnd}_g$gpu.log"
    [ -f "$f" ] || continue
    while read -r tvk er nn iv; do
      [ -n "$tvk" ] || continue
      E[$tvk]=$(( ${E[$tvk]} + er ))
      INV[$tvk]=$(( ${INV[$tvk]} + iv ))
      if [ "$tvk" = "5" ]; then N=$(( N + nn )); fi
    done < <(tr -d '\r' < "$f" | awk '
      /tv=/ {
        tv=""; er=0; nn=0; iv=0; prevL=0
        for (i=1;i<=NF;i++) {
          if ($i ~ /^tv=/) tv=substr($i,4)+0
          if ($i ~ /^LER=/) prevL=1
          if ($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevL) { split(substr($i,2,length($i)-2),a,"/"); er=a[1]; nn=a[2]; prevL=0 }
          if ($i ~ /^invalid=/) iv=substr($i,9)+0
        }
        if (tv != "") print tv, er, nn, iv
      }')
  done
}

declare -A E=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
declare -A INV=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
N=0
last=$(ls "$OUT"/p${p}_r*_g0.log 2>/dev/null | sed -E 's/.*_r([0-9]+)_g0\.log/\1/' | sort -n | tail -1)
round=${last:-0}
for r in $(seq 1 "$round"); do pool_round "$r"; done
MAXN=$(( N + ADD ))
echo "EXTEND p=$p resumed through round=$round  N=$N  tv5=${E[5]} inv5=${INV[5]}  -> target N=$MAXN (+$ADD)  @ $(date)" >> "$PROG"

while [ "$N" -lt "$MAXN" ]; do
  round=$((round+1))
  for gpu in 0 1 2 3 4 5; do
    s=$(( 50000000 + pidx*1000000 + round*10 + gpu ))
    setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
      $PY experiments/bench_circuit_pbit.py $COMMON --shots $ch --seed $s --circuit \"$circ\"" \
      > "$OUT/p${p}_r${round}_g$gpu.log" 2>&1 &
  done
  wait
  pool_round "$round"
  echo "p=$p round=$round N=$N  tv1=${E[1]} tv2=${E[2]} tv3=${E[3]} tv4=${E[4]} tv5=${E[5]}  inv5=${INV[5]}  $(date)" >> "$PROG"
done

echo "--- p=$p EXTENDED FINAL N=$N @ $(date) ---" >> "$RESULT"
for tv in 1 2 3 4 5; do
  "$PY" -c "import math;E=${E[$tv]};N=$N;ler=E/N if N else 0;se=(math.sqrt(E)/N) if (E and N) else 0;ul=((3.0 if E==0 else (E+1.64*math.sqrt(E)))/N) if N else 0;print(f'p=$p tv=$tv  LER={ler:.3e} +-{se:.1e}  ul95={ul:.2e}  errors={E}/{N}  invalid=${INV[$tv]}')" >> "$RESULT"
done
echo "EXTEND ALLDONE @ $(date)" >> "$RESULT"
