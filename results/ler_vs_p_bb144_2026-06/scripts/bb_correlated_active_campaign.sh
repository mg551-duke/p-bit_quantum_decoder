#!/usr/bin/env bash
# CORRELATED (full-DEM) LER-vs-p campaign for the bivariate-bicycle family with
# optional ACTIVE-REGION updates (scaling-remedy #1). Parameterized by CODE:
#   CODE=72  -> [[72,12,6]]   memory_Z, d=6,  rounds=6,  A=x^3+y+y^2,   B=y^3+x+x^2
#   CODE=144 -> [[144,12,12]] memory_X, d=12, rounds=12, A=x^3+y+y^2,   B=y^3+x+x^2
#   CODE=288 -> [[288,12,18]] memory_Z, d=18, rounds=18, A=x^3+y^2+y^7, B=y^3+x+x^2
# Correlated = the FULL undecomposed hyperedge DEM (no --basis-filter). Captures
# tv=1..5 from one trajectory/shot and ranks by tv5. Resume-safe per p.
#
# Active-region knobs (default OFF = bit-for-bit identical to the pre-remedy run):
#   IDLE_RATE     soft idle gating: idle bits (e_j=0,v_j=0) resample w.p. this   (e.g. 0.05)
#   GAMMA_ACTIVE  prior scale for near-syndrome bits (e_j=0,v_j>0)               (e.g. 0.5)
#   GAMMA_FAR     prior scale for far/idle bits       (e_j=0,v_j=0)              (e.g. 1.5)
#
# Baseline configs to reproduce/beat (correlated): 72 -> sweeps=1000; 144 -> sweeps=500.
# Decoder config overridable via env: RHO BSTART BSTOP SWEEPS CHAINS LEGS TGT PS GPUS OUT MEM.
# GPUs 0-5 ONLY: GPU 6 left alone (user constraint), GPU 7 (other user) hard-guarded out.
#
# Detached launch example (bb72 tuning point, active-region on):
#   OUT=/tmp/bb72corr_ar CODE=72 PS="0.003 0.002" IDLE_RATE=0.05 GAMMA_ACTIVE=0.5 GAMMA_FAR=1.5 \
#     setsid bash bb_correlated_active_campaign.sh > /tmp/bb72corr_ar_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"

CODE=${CODE:-72}
case "$CODE" in
  72)  NAME=72_12_6;   DIST=6;  DEF_MEM=memory_Z; POLYS="A=x^3+y+y^2,B=y^3+x+x^2";   DEF_SWEEPS=1000;;
  144) NAME=144_12_12; DIST=12; DEF_MEM=memory_X; POLYS="A=x^3+y+y^2,B=y^3+x+x^2";   DEF_SWEEPS=500;;
  288) NAME=288_12_18; DIST=18; DEF_MEM=memory_Z; POLYS="A=x^3+y^2+y^7,B=y^3+x+x^2"; DEF_SWEEPS=1000;;
  *)   echo "unknown CODE=$CODE (use 72|144|288)"; exit 2;;
esac
MEM=${MEM:-$DEF_MEM}
PRE="circuit=bicycle_bivariate_${NAME}_${MEM},"
SUF="distance=${DIST},rounds=${DIST},REP,noise_model=uniform_circuit,basis=CX,${POLYS}.stim"

RHO=${RHO:-0.30}; BSTOP=${BSTOP:-6}; BSTART=${BSTART:-0.5}
SWEEPS=${SWEEPS:-$DEF_SWEEPS}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}
BASIS=${BASIS-}                   # empty = correlated (full DEM). '-' not ':-' so empty is honored.
TGT=${TGT:-10}                    # tv5 content-error target per p
GPUS=${GPUS:-"0 1 2 3 4 5"}
# Active-region env knobs (unset/empty -> flag omitted -> decoder default = no-op).
IDLE_RATE=${IDLE_RATE:-}; GAMMA_ACTIVE=${GAMMA_ACTIVE:-}; GAMMA_FAR=${GAMMA_FAR:-}

# HARD GUARD: never use GPU 6 (leave alone) or GPU 7 (other user).
read -r -a _g <<< "$GPUS"; GPUARR=()
for x in "${_g[@]}"; do [ "$x" = "6" ] && continue; [ "$x" = "7" ] && continue; GPUARR+=("$x"); done
NG=${#GPUARR[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (6/7 excluded)"; exit 2; }

COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --device cuda --max-legs $LEGS \
        --selection min_weight --gamma 1.0 --beta-start $BSTART --beta-stop $BSTOP \
        ${BASIS:+--basis-filter $BASIS} \
        ${IDLE_RATE:+--idle-update-rate $IDLE_RATE} \
        ${GAMMA_ACTIVE:+--gamma-active $GAMMA_ACTIVE} \
        ${GAMMA_FAR:+--gamma-far $GAMMA_FAR} \
        --capture-targets 1,2,3,4,5"

read -r -a PS <<< "${PS:-0.005 0.004 0.003 0.002 0.001}"
declare -A CHUNK=( [0.005]=1000 [0.004]=3000 [0.003]=10000 [0.002]=25000 [0.001]=50000 )
declare -A MAXN=(  [0.005]=200000 [0.004]=1000000 [0.003]=5000000 [0.002]=25000000 [0.001]=80000000 )
DEF_CHUNK=10000; DEF_MAXN=5000000

OUT=${OUT:-/tmp/bb${CODE}corr_ar}; mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"; : > "$RESULT"
MODE=$([ -n "$BASIS" ] && echo "FILTERED-$BASIS" || echo "CORRELATED(full-DEM)")
AR="idle=${IDLE_RATE:-off} g_active=${GAMMA_ACTIVE:-off} g_far=${GAMMA_FAR:-off}"
echo "BB [[$CODE]] $MODE ($MEM) $AR rho=$RHO beta=$BSTART->$BSTOP sweeps=$SWEEPS chains=$CHAINS legs=$LEGS tgt=$TGT ps='${PS[*]}' gpus='${GPUARR[*]}' start $(date)" > "$PROG"

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
      s=$(( 60000000 + pidx*1000000 + round*10 + gpu ))
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
