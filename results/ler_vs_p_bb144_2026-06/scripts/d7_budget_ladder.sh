#!/usr/bin/env bash
# DECODER-BUDGET ladder for the rotated SURFACE CODE d=7 memory_X FILTERED-X problem at
# p=0.002. Takes the best-3 configs from the d=7 p=0.003 tuning (rho/beta-stop) and, for
# each, escalates the DECODER budget -- chains, sweeps, max_legs, and target-valid
# candidates (via --capture-targets) -- to find where the true (tv-max) LER PLATEAUS
# ("increase the budget until no gain"). NOT a statistics sweep: it isolates the compute
# budget's effect on accuracy.
#
# PAIRING: every (config,level) round uses the SAME seed sequence, so all levels decode
# IDENTICAL syndromes -> LER differences across levels are purely the budget effect
# (removes sampling noise; a real gain = higher budget fixes shots the baseline got wrong).
#
# GPU 6 ONLY (GPUs 0-5 busy with the d=11 correlated campaign; GPU 7 in use by another
# user). Resume-safe: a completed cell (log has a tv line) is skipped. Rounds keep GPU
# memory bounded; tv-ladder errors are pooled across rounds.
#
# Launch (detached):
#   setsid bash results/ler_vs_p_bb144_2026-06/scripts/d7_budget_ladder.sh > /tmp/sctune/d7budget/ladder_launch.log 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD=/home/mert/CascadeProjects/relay/tests/testdata/surface
P=${P:-0.002}; DIST=${DIST:-7}; GPU=${GPU:-6}
SEED0=${SEED0:-4242}                 # base seed (shared across levels => paired syndromes)
CHUNK=${CHUNK:-50000}                # shots/round (bounds GPU memory; d=7 filtered is tiny)
NROUNDS=${NROUNDS:-3}                 # => SHOTS = CHUNK*NROUNDS = 150k paired shots/cell
CIRC="$TD/circuit=rotated_surface_code_memory_X,distance=$DIST,rounds=$DIST,error_rate=$P,noise_model=uniform_circuit,basis=CX.stim"
[ -f "$CIRC" ] || { echo "MISSING circuit: $CIRC"; exit 2; }
OUT=${OUT:-/tmp/sctune/d7budget}; mkdir -p "$OUT"
RESULT="$OUT/ladder_RESULT.txt"; PROG="$OUT/ladder_progress.txt"
[ -f "$RESULT" ] || : > "$RESULT"
echo "d7 budget-ladder d=$DIST p=$P seed0=$SEED0 chunk=$CHUNK nrounds=$NROUNDS gpu=$GPU start $(date)" >> "$PROG"

# best-3 configs from d=7 p=0.003 tuning: "rho beta_stop"
CONFIGS=( "0.25 16" "0.15 8" "0.20 6" )
# budget levels: "name chains sweeps legs capture-targets"
LEVELS=(
  "L0 128  1000 24 1,2,3,4,5"
  "L1 256  2000 48 1,2,4,8"
  "L2 512  3000 96 1,2,4,8,16"
  "L3 1024 4000 96 1,2,4,8,16"
)

for c in "${CONFIGS[@]}"; do
  set -- $c; rho=$1; bs=$2
  for L in "${LEVELS[@]}"; do
    set -- $L; nm=$1; ch=$2; sw=$3; lg=$4; caps=$5
    tag="rho${rho}_bs${bs}_${nm}"
    done_marker="$OUT/${tag}.DONE"
    [ -f "$done_marker" ] && { echo "skip $tag (done)" >> "$PROG"; continue; }
    declare -A E=(); N=0
    maxtv=$(echo "$caps" | tr ',' '\n' | sort -n | tail -1)
    echo "=== $tag chains=$ch sweeps=$sw legs=$lg caps=$caps @ $(date) ===" >> "$PROG"
    for r in $(seq 1 "$NROUNDS"); do
      s=$(( SEED0 + r ))
      rlog="$OUT/${tag}_r${r}.log"
      if ! grep -q 'tv=' "$rlog" 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src $PY experiments/bench_circuit_pbit.py \
          --device cuda --chains "$ch" --sweeps "$sw" --max-legs "$lg" --selection min_weight \
          --gamma 1.0 --beta-start 0.5 --beta-stop "$bs" --rho-scales "$rho" --basis-filter X \
          --capture-targets "$caps" --shots "$CHUNK" --seed "$s" --circuit "$CIRC" \
          > "$rlog" 2>&1
      fi
      # pool this round's tv-ladder into E[] and N
      while read -r tvk er nn; do
        [ -n "$tvk" ] || continue
        E[$tvk]=$(( ${E[$tvk]:-0} + er ))
        [ "$tvk" = "$maxtv" ] && N=$(( N + nn ))
      done < <(tr -d '\r' < "$rlog" | awk '
        /tv=/ { tv="";er=0;nn=0;prevL=0
          for(i=1;i<=NF;i++){ if($i ~ /^tv=/)tv=substr($i,4)+0; if($i ~ /^LER=/)prevL=1;
            if($i ~ /^\([0-9]+\/[0-9]+\)$/ && prevL){split(substr($i,2,length($i)-2),a,"/");er=a[1];nn=a[2];prevL=0} }
          if(tv!="")print tv,er,nn }')
      echo "$tag round=$r N=$N tv${maxtv}=${E[$maxtv]:-0} $(date)" >> "$PROG"
    done
    ms=$(grep -hoE '[0-9.]+ ms/shot' "$OUT/${tag}_r1.log" 2>/dev/null | head -1 | grep -oE '^[0-9.]+')
    # emit tv-ladder for this cell (pooled)
    for tv in $(echo "$caps" | tr ',' ' '); do
      "$PY" -c "import math;E=${E[$tv]:-0};N=$N;ler=E/N if N else 0;se=(math.sqrt(E)/N) if (E and N) else 0;ul=((3.0 if E==0 else (E+1.64*math.sqrt(E)))/N) if N else 0;print(f'$tag chains=$ch sweeps=$sw legs=$lg tv=$tv LER={ler:.3e} +-{se:.1e} ul95={ul:.2e} err={E}/{N} ms=${ms:-nan}')" >> "$RESULT"
    done
    touch "$done_marker"
    unset E
  done
done
echo "ALLDONE @ $(date)" >> "$RESULT"
