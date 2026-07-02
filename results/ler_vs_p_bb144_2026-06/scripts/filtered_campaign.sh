#!/usr/bin/env bash
# Autonomous FILTERED LER-vs-p campaign on bb144 [[144,12]] d=12, memory_Z circuits
# with --basis-filter Z (936 det / 8784 mech) -- the SAME problem RelayBP benchmarks,
# for an apples-to-apples comparison. Captures tv=1..5 from one trajectory/shot.
#
# Tuned config (override via env): SWEEPS, RHO, CHAINS. Defaults are the phase-1/2/3
# filtered optimum (rho_scale=0.20, sweeps=1000, chains=128 -> beats RelayBP-300sets).
#
# For each p we run rounds of GPU shards, pool per-tv error counts, stop when tv=5
# reaches the per-p error target or the shot cap (MAXN) is hit. Resume-safe.
#
# Progress: $OUT/progress.txt   Final pooled table: $OUT/RESULT.txt
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
SUF="distance=12,rounds=12,REP,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"

SWEEPS=${SWEEPS:-1000}; RHO=${RHO:-0.20}; CHAINS=${CHAINS:-128}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
read -r -a GPUARR <<< "$GPUS"; NG=${#GPUARR[@]}
COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --device cuda --max-legs 24 \
        --selection min_weight --gamma 1.0 --basis-filter Z --capture-targets 1,2,3,4,5"

declare -a PS=(0.005 0.004 0.003 0.002)
declare -A TARGET=( [0.005]=200 [0.004]=150 [0.003]=100 [0.002]=50 )
declare -A CHUNK=(  [0.005]=400 [0.004]=4000 [0.003]=16000 [0.002]=32000 )
# Shot caps (low-p points are LER-limited): p=0.002 capped per prior decision.
declare -A MAXN=(   [0.005]=200000 [0.004]=2000000 [0.003]=4000000 [0.002]=2000000 )

OUT=${OUT:-/tmp/zcamp}
mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"
: > "$RESULT"
echo "FILTERED campaign (memory_Z basis Z) sweeps=$SWEEPS rho=$RHO chains=$CHAINS gpus='$GPUS' start $(date)" > "$PROG"

pool_round() {  # $1=p  $2=round
  local p="$1" rnd="$2" gpu f
  for gpu in "${GPUARR[@]}"; do
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
          if ($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevL) {
            split(substr($i,2,length($i)-2),a,"/"); er=a[1]; nn=a[2]; prevL=0
          }
          if ($i ~ /^invalid=/) iv=substr($i,9)+0
        }
        if (tv != "") print tv, er, nn, iv
      }')
  done
}

pidx=-1
for p in "${PS[@]}"; do
  pidx=$((pidx+1))
  circ="$TD/circuit=bicycle_bivariate_144_12_12_memory_Z,${SUF/REP/error_rate=$p}"
  if [ ! -f "$circ" ]; then echo "MISSING circuit for p=$p: $circ" >> "$PROG"; continue; fi
  tgt="${TARGET[$p]}"; ch="${CHUNK[$p]}"; maxn="${MAXN[$p]}"
  declare -A E=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  declare -A INV=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  N=0; round=0
  last=$(ls "$OUT"/p${p}_r*_g${GPUARR[0]}.log 2>/dev/null | sed -E 's/.*_r([0-9]+)_g[0-9]+\.log/\1/' | sort -n | tail -1)
  if [ -n "$last" ]; then
    for r in $(seq 1 "$last"); do pool_round "$p" "$r"; done
    round=$last
    echo "p=$p RESUMED through round=$round N=$N tv5=${E[5]} (target $tgt) @ $(date)" >> "$PROG"
  fi
  echo "=== p=$p target(tv5)=$tgt chunk=$ch maxn=$maxn @ $(date) ===" >> "$PROG"
  while [ "${E[5]}" -lt "$tgt" ] && [ "$N" -lt "$maxn" ]; do
    round=$((round+1))
    for gpu in "${GPUARR[@]}"; do
      s=$(( 70000000 + pidx*1000000 + round*10 + gpu ))
      setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
        $PY experiments/bench_circuit_pbit.py $COMMON --shots $ch --seed $s --circuit \"$circ\"" \
        > "$OUT/p${p}_r${round}_g$gpu.log" 2>&1 &
    done
    wait
    pool_round "$p" "$round"
    echo "p=$p round=$round N=$N  tv1=${E[1]} tv2=${E[2]} tv3=${E[3]} tv4=${E[4]} tv5=${E[5]}  inv5=${INV[5]}  $(date)" >> "$PROG"
  done
  echo "--- p=$p FINAL N=$N @ $(date) ---" >> "$RESULT"
  for tv in 1 2 3 4 5; do
    "$PY" -c "import math;E=${E[$tv]};N=$N;ler=E/N if N else 0;se=(math.sqrt(E)/N) if (E and N) else 0;ul=((3.0 if E==0 else (E+1.64*math.sqrt(E)))/N) if N else 0;print(f'p=$p tv=$tv  LER={ler:.3e} +-{se:.1e}  ul95={ul:.2e}  errors={E}/{N}  invalid=${INV[$tv]}')" >> "$RESULT"
  done
  unset E INV
done
echo "ALLDONE @ $(date)" >> "$RESULT"
