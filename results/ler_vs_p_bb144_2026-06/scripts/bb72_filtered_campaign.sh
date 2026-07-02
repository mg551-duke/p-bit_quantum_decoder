#!/usr/bin/env bash
# FILTERED LER-vs-p campaign for the [[72,12,6]] bivariate-bicycle memory_Z circuits
# (d=6, rounds=6, A=x^3+y+y^2, B=y^3+x+x^2), decoded with --basis-filter Z (single-sector
# "filtered" decode). This is the low-distance anchor of the BB family 72->144->288 (fixed
# k=12, increasing d=6/12/18). Captures tv=1..5 from one trajectory/shot; ranks by tv5.
#
# Config: rho=0.30, beta 0.5->6, sweeps=1000, chains=128, max_legs=24 -- the SAME config as
# the 288 filtered campaign, so the whole BB family shares one decoder config (clean
# cross-code comparison). [[72,12,6]] is the weakest member (d=6) so its LER sits ABOVE
# 144/288; re-tune if a 72-specific optimum is wanted.
#
# Per p: sharded rounds across allowed GPUs, pool per-tv errors, stop at TGT tv5 errors OR
# the MAXN shot cap (a capped point is a low-LER upper bound). Resume-safe per p.
# GPUs 0-5 ONLY: GPU 6 LEFT ALONE (user constraint) and GPU 7 (other user) HARD-GUARDED out.
#
# Override via env: RHO BSTOP BSTART SWEEPS CHAINS LEGS BASIS TGT PS GPUS OUT.
# Detached launch:
#   OUT=/tmp/bb72filtZ setsid bash bb72_filtered_campaign.sh > /tmp/bb72filtZ_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
# 72 circuit naming: distance=6, rounds=6, A=x^3+y+y^2, B=y^3+x+x^2 (memory_Z)
PRE="circuit=bicycle_bivariate_72_12_6_memory_Z,"
SUF="distance=6,rounds=6,REP,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"

RHO=${RHO:-0.30}; BSTOP=${BSTOP:-6}; BSTART=${BSTART:-0.5}
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}
BASIS=${BASIS-Z}                  # default Z (filtered); pass BASIS="" for full/correlated DEM (note: '-' not ':-' so empty is honored)
TGT=${TGT:-10}                    # tv5 content-error target per p
GPUS=${GPUS:-"0 1 2 3 4 5"}
# HARD GUARD: never use GPU 6 (leave alone) or GPU 7 (other user).
read -r -a _g <<< "$GPUS"; GPUARR=()
for x in "${_g[@]}"; do [ "$x" = "6" ] && continue; [ "$x" = "7" ] && continue; GPUARR+=("$x"); done
NG=${#GPUARR[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (6/7 excluded)"; exit 2; }

COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --device cuda --max-legs $LEGS \
        --selection min_weight --gamma 1.0 --beta-start $BSTART --beta-stop $BSTOP \
        ${BASIS:+--basis-filter $BASIS} --capture-targets 1,2,3,4,5"

read -r -a PS <<< "${PS:-0.005 0.004 0.003 0.002 0.001}"
# round granularity (shots/GPU) + per-p shot cap, sized for the weak d=6 code (higher LER
# than 144/288; small DEM so fast). First round reveals true LER; rounds run to 10 errors.
declare -A CHUNK=( [0.005]=1000 [0.004]=2000 [0.003]=5000 [0.002]=15000 [0.001]=40000 )
declare -A MAXN=(  [0.005]=200000 [0.004]=500000 [0.003]=2000000 [0.002]=5000000 [0.001]=15000000 )
DEF_CHUNK=5000; DEF_MAXN=3000000

OUT=${OUT:-/tmp/bb72filtZ}; mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"; : > "$RESULT"
MODE=$([ -n "$BASIS" ] && echo "FILTERED-$BASIS" || echo "CORRELATED(full-DEM)")
echo "BB [[72,12,6]] $MODE (memory_Z) campaign rho=$RHO beta=$BSTART->$BSTOP sweeps=$SWEEPS chains=$CHAINS legs=$LEGS tgt=$TGT ps='${PS[*]}' gpus='${GPUARR[*]}' start $(date)" > "$PROG"

pool_round() {  # $1=p  $2=round  -> accumulate into E[] INV[] N
  local p="$1" rnd="$2" gpu f
  for gpu in "${GPUARR[@]}"; do
    f="$OUT/p${p}_r${rnd}_g$gpu.log"
    [ -f "$f" ] || continue
    while read -r tvk er nn iv; do
      [ -n "$tvk" ] || continue
      E[$tvk]=$(( ${E[$tvk]} + er )); INV[$tvk]=$(( ${INV[$tvk]} + iv ))
      [ "$tvk" = "5" ] && N=$(( N + nn ))
    done < <(tr -d '\r' < "$f" | awk '
      /tv=/ { tv="";er=0;nn=0;iv=0;prevL=0
        for(i=1;i<=NF;i++){ if($i ~ /^tv=/)tv=substr($i,4)+0; if($i ~ /^LER=/)prevL=1;
          if($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevL){split(substr($i,2,length($i)-2),a,"/");er=a[1];nn=a[2];prevL=0}
          if($i ~ /^invalid=/)iv=substr($i,9)+0 }
        if(tv!="")print tv,er,nn,iv }')
  done
}

pidx=-1
for p in "${PS[@]}"; do
  pidx=$((pidx+1))
  circ="$TD/${PRE}${SUF/REP/error_rate=$p}"
  if [ ! -f "$circ" ]; then echo "MISSING circuit for p=$p: $circ" >> "$PROG"; continue; fi
  ch="${CHUNK[$p]:-$DEF_CHUNK}"; maxn="${MAXN[$p]:-$DEF_MAXN}"
  declare -A E=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  declare -A INV=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  N=0; round=0
  last=$(ls "$OUT"/p${p}_r*_g${GPUARR[0]}.log 2>/dev/null | sed -E 's/.*_r([0-9]+)_g[0-9]+\.log/\1/' | sort -n | tail -1)
  if [ -n "$last" ]; then
    for r in $(seq 1 "$last"); do pool_round "$p" "$r"; done; round=$last
    echo "p=$p RESUMED through round=$round N=$N tv5=${E[5]} (target $TGT) @ $(date)" >> "$PROG"
  fi
  echo "=== p=$p target(tv5)=$TGT chunk=$ch maxn=$maxn @ $(date) ===" >> "$PROG"
  while [ "${E[5]}" -lt "$TGT" ] && [ "$N" -lt "$maxn" ]; do
    round=$((round+1))
    for gpu in "${GPUARR[@]}"; do
      s=$(( 50000000 + pidx*1000000 + round*10 + gpu ))
      setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
        $PY experiments/bench_circuit_pbit.py $COMMON --shots $ch --seed $s --circuit \"$circ\"" \
        > "$OUT/p${p}_r${round}_g$gpu.log" 2>&1 &
    done
    wait
    pool_round "$p" "$round"
    echo "p=$p round=$round N=$N  tv1=${E[1]} tv2=${E[2]} tv3=${E[3]} tv4=${E[4]} tv5=${E[5]}  inv5=${INV[5]}  $(date)" >> "$PROG"
  done
  ms=$(grep -hoE '[0-9.]+ ms/shot' "$OUT/p${p}_r${round}_g${GPUARR[0]}.log" 2>/dev/null | head -1 | grep -oE '^[0-9.]+')
  echo "--- p=$p FINAL N=$N ms_shot=${ms:-nan} @ $(date) ---" >> "$RESULT"
  for tv in 1 2 3 4 5; do
    "$PY" -c "import math;E=${E[$tv]};N=$N;ler=E/N if N else 0;se=(math.sqrt(E)/N) if (E and N) else 0;ul=((3.0 if E==0 else (E+1.64*math.sqrt(E)))/N) if N else 0;print(f'p=$p tv=$tv  LER={ler:.3e} +-{se:.1e}  ul95={ul:.2e}  errors={E}/{N}  invalid=${INV[$tv]}')" >> "$RESULT"
  done
  unset E INV
done
echo "ALLDONE @ $(date)" >> "$RESULT"
