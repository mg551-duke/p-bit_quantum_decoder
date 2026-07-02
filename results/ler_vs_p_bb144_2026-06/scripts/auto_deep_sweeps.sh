#!/usr/bin/env bash
# auto_deep_sweeps.sh -- AUTONOMOUS "push sweeps until the LER floor" driver for the
# filtered memory_Z bb144 [[144,12]] d=12 p-bit decoder.
#
# Paired / common-random-numbers design: EVERY sweep depth uses the SAME fixed seed,
# so all configs decode the identical error set and the only variable is annealing
# depth. This kills shot-noise in point-to-point comparison, which matters because
# returns diminish at high sweeps (adjacent points can differ by <10%).
#
# The driver runs unattended end to end:
#   1. launch a sweep ladder across the GPUs (one depth per GPU), wait for all,
#   2. pool tv=1..5 into an LER-vs-sweeps curve (deep_sweeps_report.py),
#   3. AUTO-EXTEND to deeper sweeps while the deepest point is still *significantly*
#      better than the previous one; otherwise STOP at the statistical plateau,
#   4. write the final curve + a RESULT file.
#
# Watch it:   tail -f /tmp/ftune/autodeep/DRIVER.log
# Stop it:    pkill -f auto_deep_sweeps   (and: pkill -f bench_circuit_pbit)
#
# Tunables (env): P SHOTS SEED CHAINS RHO GAMMA NGPU TV MAXSWEEPS MIN_REL SIG
#                 MAX_STAGES MAXWAIT LADDER
set -u

PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
SCRIPTS=$ROOT/results/ler_vs_p_bb144_2026-06/scripts
CIRCDIR=/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate

P=${P:-0.004}
SHOTS=${SHOTS:-16000}
SEED=${SEED:-7000}                 # fixed for ALL configs -> paired/CRN curve
CHAINS=${CHAINS:-128}
RHO=${RHO:-0.20}                   # phase-1/2 best rho_scale
GAMMA=${GAMMA:-1.0}
NGPU=${NGPU:-8}
TV=${TV:-5}                        # operating point the plateau is judged on
MAXSWEEPS=${MAXSWEEPS:-24000}      # hard cap on annealing depth
MIN_REL=${MIN_REL:-0.03}           # require >=3% relative LER drop to keep going
SIG=${SIG:-1.0}                    # ... and drop must exceed SIG * SE_of_difference
MAX_STAGES=${MAX_STAGES:-4}        # safety cap on extension rounds
MAXWAIT=${MAXWAIT:-72000}          # per-stage wait cap (s)
LADDER=${LADDER:-"1000 1500 2000 3000 4000 6000 8000 12000"}

COMMON="--selection min_weight --max-legs 24"
CAP="--capture-targets 1,2,3,4,5"
CIRC="$CIRCDIR/circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=$P,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"
OUT=/tmp/ftune/autodeep
mkdir -p "$OUT"
DRV="$OUT/DRIVER.log"
RES="$OUT/RESULT_DEEPSWEEPS.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }

[ -f "$CIRC" ] || { log "FATAL missing circuit: $CIRC"; exit 1; }
[ -f "$PY" ]   || { log "FATAL missing python: $PY"; exit 1; }

# Launch one sweep-depth run (detached), unless its log already finished.
launch_one() { # $1=sweeps  $2=gpu
  local s=$1 gpu=$2 lg="$OUT/s$1.log"
  if [ -f "$lg" ] && grep -q '^DONE$' "$lg"; then
    log "skip s$s (already DONE)"
    return
  fi
  setsid bash -c "export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu; cd $ROOT;
    echo '### name=s$s p=$P shots=$SHOTS seed=$SEED sweeps=$s rho=$RHO chains=$CHAINS gpu=$gpu';
    $PY experiments/bench_circuit_pbit.py --circuit \"$CIRC\" --basis-filter Z \
      --device cuda --shots $SHOTS --seed $SEED $COMMON $CAP \
      --chains $CHAINS --sweeps $s --rho-scales $RHO --gamma $GAMMA;
    echo DONE" > "$lg" 2>&1 &
  log "launched s$s on gpu $gpu (seed $SEED, shots $SHOTS)"
}

# Block until every sweep value in $@ has a DONE log (or MAXWAIT elapses).
wait_done() { # args: sweep values
  local want=("$@") t0 done s lg
  t0=$(date +%s)
  while :; do
    done=0
    for s in "${want[@]}"; do
      lg="$OUT/s$s.log"
      [ -f "$lg" ] && grep -q '^DONE$' "$lg" && done=$((done + 1))
    done
    log "  progress: $done/${#want[@]} sweep-points DONE"
    [ "$done" -ge "${#want[@]}" ] && return 0
    [ $(( $(date +%s) - t0 )) -gt "$MAXWAIT" ] && { log "  MAXWAIT hit; proceeding partial"; return 1; }
    sleep 60
  done
}

# Launch a whole batch across GPUs (round-robin) then wait for it.
run_batch() { # args: sweep values
  local vals=("$@") i=0 s
  for s in "${vals[@]}"; do
    launch_one "$s" "$(( i % NGPU ))"
    i=$((i + 1))
  done
  wait_done "${vals[@]}"
}

log "=== auto_deep_sweeps START  p=$P shots=$SHOTS seed=$SEED chains=$CHAINS rho=$RHO gamma=$GAMMA ==="
log "    cap=$MAXSWEEPS  min_rel=$MIN_REL  sig=$SIG  tv=$TV  ngpu=$NGPU"
log "    initial ladder: $LADDER"

tested=" "
ladder="$LADDER"
stage=0
final_decision=""

while [ "$stage" -lt "$MAX_STAGES" ]; do
  # drop any sweep values already tested this campaign
  batch=""
  for s in $ladder; do
    case "$tested" in *" $s "*) ;; *) batch="$batch $s"; tested="$tested$s " ;; esac
  done
  batch=$(echo "$batch" | xargs)
  [ -z "$batch" ] && { log "stage $stage: nothing new to run"; break; }

  log "--- stage $stage: running sweeps [$batch] ---"
  run_batch $batch

  decision=$("$PY" "$SCRIPTS/deep_sweeps_report.py" --out "$OUT" --tv "$TV" \
    --max-sweeps "$MAXSWEEPS" --min-rel "$MIN_REL" --sig "$SIG" 2>>"$DRV" \
    | tee -a "$DRV" | grep '^DECISION' | tail -n1)
  log "stage $stage decision -> ${decision:-<none>}"
  final_decision="$decision"

  set -- $decision
  if [ "${2:-}" = "EXTEND" ]; then
    shift 2
    ladder="$*"
    log "extending to deeper sweeps: $ladder"
    stage=$((stage + 1))
  else
    break
  fi
done

# ---- final report -----------------------------------------------------------
log "=== FINAL CURVE ==="
{
  echo "# auto_deep_sweeps RESULT  $(date)"
  echo "# circuit=memory_Z bb144 [[144,12]] d=12  p=$P  basis=Z(filtered)"
  echo "# decoder: pbit-cuda chains=$CHAINS rho_scale=$RHO gamma=$GAMMA legs<=24 selection=min_weight"
  echo "# paired seed=$SEED  shots/point=$SHOTS  plateau judged on tv=$TV"
  echo "# stop rule: extend while deepest LER drop > max($MIN_REL rel, ${SIG}sigma); cap=$MAXSWEEPS"
  echo
  "$PY" "$SCRIPTS/deep_sweeps_report.py" --out "$OUT" --tv "$TV" \
    --max-sweeps "$MAXSWEEPS" --min-rel "$MIN_REL" --sig "$SIG"
  echo
  echo "FINAL $final_decision"
} | tee "$RES" | tee -a "$DRV"

# stash the curve next to the campaign for keeping
cp "$OUT/curve.csv" "$SCRIPTS/../autodeep_curve.csv" 2>/dev/null || true
cp "$RES" "$SCRIPTS/../RESULT_DEEPSWEEPS.txt" 2>/dev/null || true
log "=== auto_deep_sweeps COMPLETE -> $RES ==="
