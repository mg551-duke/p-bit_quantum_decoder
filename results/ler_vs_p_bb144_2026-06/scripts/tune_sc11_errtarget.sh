#!/usr/bin/env bash
# Tuning grid for the rotated SURFACE CODE memory_X circuit-level problem at p=0.003
# (distance set via DIST env, default 11), run to a TRUE-LER ERROR TARGET per config
# (default 10 tv5 content errors)
# instead of fixed shots -- so every config gets a comparable-precision LER.
# Optional BASIS env: --basis-filter {X,Z} (filtered single-sector) or empty (full/correlated DEM).
# NOTE: filtering is about the detector/mechanism SET, NOT observable count -- applies even with 1 obs.
#
# Each config is sharded across ALL allowed GPUs; configs run SEQUENTIALLY so the
# per-config RESULT lines stream in.  Stop a config at TARGET tv5 errors OR the MAXN
# shot cap (a config that caps with <TARGET errors simply has a very low LER and is
# ranked best).  Resume-safe per config.  Rank afterwards with tune288_pool.py
# (true LER tv5 asc, then ms/shot, then invalid-rate).
#
# Detached launch:
#   OUT=/tmp/sctune/errtgt setsid bash tune_sc11_errtarget.sh > /tmp/sctune/errtgt_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD=/home/mert/CascadeProjects/relay/tests/testdata/surface
P=${P:-0.003}
DIST=${DIST:-11}           # surface-code distance (rounds=distance by convention)
SWEEPS=${SWEEPS:-1000}; CHAINS=${CHAINS:-128}; LEGS=${LEGS:-24}; BSTART=${BSTART:-0.5}
RHOS=${RHOS:-"0.15 0.20 0.25 0.30"}
BSTOPS=${BSTOPS:-"6 8 12 16"}
TARGET=${TARGET:-10}        # tv5 true-LER errors per config
CHUNK=${CHUNK:-5000}        # shots per GPU per round
MAXN=${MAXN:-1000000}       # per-config shot cap
BASIS=${BASIS:-}            # optional --basis-filter {X,Z}; empty = full/correlated DEM
GPUS=${GPUS:-"0 1 2 3 4 5"}
# GUARD: never use GPU 7 (in use by another user). GPU 6 allowed per user instruction.
read -r -a _g <<< "$GPUS"; G=()
for x in "${_g[@]}"; do [ "$x" = "7" ] && continue; G+=("$x"); done
NG=${#G[@]}; [ "$NG" -ge 1 ] || { echo "no allowed GPUs (7 excluded)"; exit 2; }
CIRC="$TD/circuit=rotated_surface_code_memory_X,distance=$DIST,rounds=$DIST,error_rate=$P,noise_model=uniform_circuit,basis=CX.stim"
[ -f "$CIRC" ] || { echo "MISSING circuit: $CIRC"; exit 2; }
OUT=${OUT:-/tmp/sctune/errtgt}; mkdir -p "$OUT"
PROG="$OUT/progress.txt"; RESULT="$OUT/RESULT.txt"; : > "$RESULT"
echo "tune_sc11 errtarget d=$DIST p=$P basis='${BASIS:-none}' target=$TARGET chunk=$CHUNK maxn=$MAXN sweeps=$SWEEPS chains=$CHAINS legs=$LEGS bstart=$BSTART rhos='$RHOS' bstops='$BSTOPS' gpus='${G[*]}' start $(date)" > "$PROG"

COMMON="--device cuda --chains $CHAINS --sweeps $SWEEPS --max-legs $LEGS --selection min_weight \
        --gamma 1.0 --beta-start $BSTART --capture-targets 1,2,3,4,5 ${BASIS:+--basis-filter $BASIS}"

pool_round() {  # $1=tag  $2=round  -> accumulate into E[] INV[] N
  local tag="$1" rnd="$2" gpu f
  for gpu in "${G[@]}"; do
    f="$OUT/${tag}_r${rnd}_g$gpu.log"
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

slot=0
for rho in $RHOS; do for bs in $BSTOPS; do
  tag="rho${rho}_bs${bs}"
  declare -A E=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  declare -A INV=( [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 )
  N=0; round=0
  # resume: re-pool any existing rounds for this config
  last=$(ls "$OUT"/${tag}_r*_g${G[0]}.log 2>/dev/null | sed -E 's/.*_r([0-9]+)_g[0-9]+\.log/\1/' | sort -n | tail -1)
  if [ -n "$last" ]; then for r in $(seq 1 "$last"); do pool_round "$tag" "$r"; done; round=$last; fi
  echo "=== $tag (rho=$rho beta=$BSTART->$bs) target tv5=$TARGET maxn=$MAXN @ $(date) ===" >> "$PROG"
  while [ "${E[5]}" -lt "$TARGET" ] && [ "$N" -lt "$MAXN" ]; do
    round=$((round+1))
    for gpu in "${G[@]}"; do
      s=$(( 2000 + slot*100000 + round*10 + gpu ))
      setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
        $PY experiments/bench_circuit_pbit.py $COMMON --beta-stop $bs --rho-scales $rho \
          --shots $CHUNK --seed $s --circuit \"$CIRC\"" \
        > "$OUT/${tag}_r${round}_g$gpu.log" 2>&1 &
    done
    wait
    pool_round "$tag" "$round"
    echo "$tag round=$round N=$N  tv1=${E[1]} tv2=${E[2]} tv3=${E[3]} tv4=${E[4]} tv5=${E[5]}  inv5=${INV[5]}  $(date)" >> "$PROG"
  done
  ms=$(grep -hoE '[0-9.]+ ms/shot' "$OUT/${tag}_r${round}_g${G[0]}.log" 2>/dev/null | head -1 | grep -oE '^[0-9.]+')
  for tv in 1 2 3 4 5; do
    "$PY" -c "import math;E=${E[$tv]};N=$N;ler=E/N if N else 0;se=(math.sqrt(E)/N) if (E and N) else 0;ul=((3.0 if E==0 else (E+1.64*math.sqrt(E)))/N) if N else 0;print(f'$tag tv=$tv  LER={ler:.3e} +-{se:.1e}  ul95={ul:.2e}  errors={E}/{N}  invalid=${INV[$tv]}  ms_shot=${ms:-nan}')" >> "$RESULT"
  done
  unset E INV
  slot=$((slot+1))
done; done
echo "ALLDONE @ $(date)" >> "$RESULT"
