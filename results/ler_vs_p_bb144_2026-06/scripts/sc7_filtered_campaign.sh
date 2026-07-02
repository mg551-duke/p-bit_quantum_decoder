#!/usr/bin/env bash
# FILTERED LER-vs-p campaign for the rotated SURFACE CODE d=7 memory_X circuits, decoded
# with --basis-filter X (192 det / 883 mech) -- the single-sector / "filtered" decode.
#
# OPTIMAL config (validated by the d=7 p=0.003 tuning + the p=0.002 decoder-BUDGET ladder,
# which showed the cheap baseline budget is optimal -- no gain from more chains/sweeps/
# legs/candidates; d=7 filtered sits at the min-weight distance-limited floor):
#   rho=0.25, beta 0.5->16, chains=128, sweeps=1000, max_legs=24, target_valid via tv1..5.
# Captures tv=1..5 from one trajectory/shot; ranks by tv5 (true/content) LER.
#
# Per p: run rounds on GPU 6, pool per-tv error counts, stop when tv5 reaches TGT errors
# OR the MAXN shot cap is hit (a capped point is a low-LER upper bound). Resume-safe per p.
#
# GPU 6 ONLY (GPUs 0-5 busy with the d=11 correlated campaign; GPU 7 in use by another
# user). Guard excludes only GPU 7.
# Override via env: DIST RHO BSTOP BSTART SWEEPS CHAINS LEGS BASIS TGT PS GPUS OUT.
# Detached launch:
#   OUT=/tmp/sccamp/d7filtX setsid bash sc7_filtered_campaign.sh > /tmp/sccamp/d7filtX_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/surface"

DIST=${DIST:-7}                   # surface-code distance (rounds=distance)
RHO=${RHO:-0.25}; BSTOP=${BSTOP:-16}; BSTART=${BSTART:-0.5}
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}
BASIS=${BASIS:-X}                 # --basis-filter sector (X for memory_X); empty = full/correlated
TGT=${TGT:-10}                    # tv5 content-error target per p
GPUS=${GPUS:-"6"}
# GUARD: never use GPU 7 (in use by another user). GPU 6 allowed per user instruction.
read -r -a _g <<< "$GPUS"; GPUARR=()
for x in "${_g[@]}"; do [ "$x" = "7" ] && continue; GPUARR+=("$x"); done
NG=${#GPUARR[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (7 excluded)"; exit 2; }

COMMON="--chains $CHAINS --sweeps $SWEEPS --rho-scales $RHO --device cuda --max-legs $LEGS \
        --selection min_weight --gamma 1.0 --beta-start $BSTART --beta-stop $BSTOP \
        ${BASIS:+--basis-filter $BASIS} --capture-targets 1,2,3,4,5"

read -r -a PS <<< "${PS:-0.005 0.004 0.003 0.002 0.001}"
# round granularity (shots/round) and per-p shot cap, sized for d=7 filtered LER
# (~5e-3 @ p=0.005 down to ~5e-5 @ p=0.001), so 10 tv5 errors land in ~1-3 rounds.
declare -A CHUNK=( [0.005]=3000 [0.004]=5000 [0.003]=10000 [0.002]=30000 [0.001]=100000 )
declare -A MAXN=(  [0.005]=60000 [0.004]=120000 [0.003]=300000 [0.002]=800000 [0.001]=3000000 )
DEF_CHUNK=20000; DEF_MAXN=2000000

OUT=${OUT:-/tmp/sccamp/d7filtX}; mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"; : > "$RESULT"
echo "SURFACE d=$DIST FILTERED-${BASIS:-none} campaign rho=$RHO beta=$BSTART->$BSTOP sweeps=$SWEEPS chains=$CHAINS legs=$LEGS tgt=$TGT ps='${PS[*]}' gpus='${GPUARR[*]}' start $(date)" > "$PROG"

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
  circ="$TD/circuit=rotated_surface_code_memory_X,distance=$DIST,rounds=$DIST,error_rate=$p,noise_model=uniform_circuit,basis=CX.stim"
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
      s=$(( 70000000 + pidx*1000000 + round*10 + gpu ))
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
