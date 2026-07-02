#!/usr/bin/env bash
# Autonomous CORRELATED (full-DEM) LER-vs-p campaign on bb288 [[288,12,18]] d=18,
# memory_Z circuits, NO basis filter (full DEM ~2736 det / ~52k mech).
# Captures tv=1..5 from one trajectory.  Companion to filtered_campaign288.sh.
#
# Config (matches correlated winner elsewhere): rho=0.30, beta 0.5->6, sweeps=1000,
# chains=128, max_legs=24.  Override via env: SWEEPS, RHO, CHAINS, BSTART, BSTOP.
#
# COST WARNING: full-DEM 288 is heavy (~200+ ms/shot) and correlated LER is low, so
# the low-p tail (p<=0.003) can take many hours-to-weeks to reach 10 errors.
# Scope is env-overridable:
#   PS="0.005 0.004 0.003"                 # which p-points (default 0.005->0.002)
#   TGT=10                                  # global tv5 error target override
#
# Backward p-order; each p sharded across all allowed GPUs; resume-safe.
# Progress: $OUT/progress.txt   Final pooled table: $OUT/RESULT.txt
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
# 288 circuit naming: distance=18, rounds=18, A=x^3+y^2+y^7, B=y^3+x+x^2
SUF="distance=18,rounds=18,REP,noise_model=uniform_circuit,basis=CX,A=x^3+y^2+y^7,B=y^3+x+x^2.stim"

SWEEPS=${SWEEPS:-1000}; RHO=${RHO:-0.30}; CHAINS=${CHAINS:-128}
BSTART=${BSTART:-0.5}; BSTOP=${BSTOP:-6}
GPUS=${GPUS:-"0 1 2 3 4 5"}
# HARD GUARD: never use GPU 6 or 7 (user constraint).
read -r -a _g <<< "$GPUS"; GPUARR=()
for x in "${_g[@]}"; do [ "$x" = "6" ] && continue; [ "$x" = "7" ] && continue; GPUARR+=("$x"); done
NG=${#GPUARR[@]}
[ "$NG" -ge 1 ] || { echo "no allowed GPUs (6/7 excluded)"; exit 2; }
# NO --basis-filter -> full DEM (correlated).
COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --beta-start $BSTART --beta-stop $BSTOP \
        --device cuda --max-legs 24 --selection min_weight --gamma 1.0 \
        --capture-targets 1,2,3,4,5"

read -r -a PS <<< "${PS:-0.005 0.004 0.003 0.002}"
TGT=${TGT:-10}
declare -A TARGET=( [0.005]=$TGT [0.004]=$TGT [0.003]=$TGT [0.002]=$TGT )
# per-GPU shots per round; smaller than filtered because ms/shot is ~2x higher.
declare -A CHUNK=(  [0.005]=400 [0.004]=1500 [0.003]=6000 [0.002]=25000 )
# generous safety caps (correlated needs more shots than filtered for same errors).
declare -A MAXN=(   [0.005]=400000 [0.004]=8000000 [0.003]=40000000 [0.002]=100000000 )

OUT=${OUT:-/tmp/bb288corr}
mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"
: > "$RESULT"
echo "CORRELATED(full-DEM) 288 campaign (memory_Z) sweeps=$SWEEPS rho=$RHO chains=$CHAINS beta=$BSTART->$BSTOP tgt=$TGT ps='${PS[*]}' gpus='${GPUARR[*]}' start $(date)" > "$PROG"

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
  circ="$TD/circuit=bicycle_bivariate_288_12_18_memory_Z,${SUF/REP/error_rate=$p}"
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
      s=$(( 90000000 + pidx*1000000 + round*10 + gpu ))
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
