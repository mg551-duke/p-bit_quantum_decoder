#!/usr/bin/env bash
# CORRELATED (full-DEM) LER-vs-p campaign for the rotated SURFACE CODE d=11 memory_X
# circuits, decoded against the FULL detector-error model (1320 det / 24485 mech) --
# i.e. NO --basis-filter, keeping cross-sector correlations. Winner config from the
# correlated tuning grid: rho=0.30, beta 0.5->6 (tied-best tv5 content LER ~2.1e-5 @
# p=0.003; correlated ~15x below filtered there). Captures tv=1..5; ranks by tv5.
#
# Per p: sharded rounds across all allowed GPUs, pool per-tv error counts, stop when
# tv5 reaches TGT errors OR the MAXN shot cap is hit (a capped point is a low-LER
# upper bound). Resume-safe per p. GPUs HARD-GUARDED to exclude 6/7. Correlated DEM is
# ~7x heavier than filtered, so CHUNK is smaller (memory) and MAXN larger (LER ~15x
# lower) than the filtered campaign.
# NOTE: p=0.001 intentionally excluded -- infeasible for correlated (~1e8 shots).
#
# Override via env: RHO BSTOP BSTART SWEEPS CHAINS LEGS BASIS TGT PS GPUS OUT.
# Detached launch:
#   OUT=/tmp/sccamp/corr setsid bash sc11_correlated_campaign.sh > /tmp/sccamp/corr_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/surface"

RHO=${RHO:-0.30}; BSTOP=${BSTOP:-6}; BSTART=${BSTART:-0.5}
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}
BASIS=${BASIS:-}                  # empty = full/correlated DEM (no --basis-filter); set X/Z to filter
TGT=${TGT:-10}                    # tv5 content-error target per p
GPUS=${GPUS:-"0 1 2 3 4 5"}
# HARD GUARD: never use GPU 6 or 7 (user constraint).
read -r -a _g <<< "$GPUS"; GPUARR=()
for x in "${_g[@]}"; do [ "$x" = "6" ] && continue; [ "$x" = "7" ] && continue; GPUARR+=("$x"); done
NG=${#GPUARR[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (6/7 excluded)"; exit 2; }

COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --device cuda --max-legs $LEGS \
        --selection min_weight --gamma 1.0 --beta-start $BSTART --beta-stop $BSTOP \
        ${BASIS:+--basis-filter $BASIS} --capture-targets 1,2,3,4,5"

read -r -a PS <<< "${PS:-0.005 0.004 0.003 0.002}"
# round granularity (shots/GPU) and per-p shot cap. Correlated: small CHUNK (heavy DEM,
# memory) + large MAXN (LER ~15x below filtered). p=0.002 cap ~7e6 ~= 24h wall ceiling.
declare -A CHUNK=( [0.005]=2000 [0.004]=4000 [0.003]=5000 [0.002]=6000 )
declare -A MAXN=(  [0.005]=300000 [0.004]=1000000 [0.003]=3000000 [0.002]=7000000 )
DEF_CHUNK=5000; DEF_MAXN=3000000

OUT=${OUT:-/tmp/sccamp/corr}; mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"; : > "$RESULT"
echo "SURFACE d=11 CORRELATED(full-DEM basis='${BASIS:-none}') campaign rho=$RHO beta=$BSTART->$BSTOP sweeps=$SWEEPS chains=$CHAINS legs=$LEGS tgt=$TGT ps='${PS[*]}' gpus='${GPUARR[*]}' start $(date)" > "$PROG"

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
  circ="$TD/circuit=rotated_surface_code_memory_X,distance=11,rounds=11,error_rate=$p,noise_model=uniform_circuit,basis=CX.stim"
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
      s=$(( 90000000 + pidx*1000000 + round*10 + gpu ))
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
