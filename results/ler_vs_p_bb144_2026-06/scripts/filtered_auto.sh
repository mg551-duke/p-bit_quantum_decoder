#!/usr/bin/env bash
# Self-driving filtered pipeline:
#   1. wait for the phase-3 sweeps tuning to finish (or 90-min timeout),
#   2. auto-pick the sweeps "knee" (filtered_pick_sweeps.py),
#   3. run the full filtered LER-vs-p campaign (filtered_campaign.sh, resume-safe),
#   4. generate the 3-way comparison plot (make_compare_plot.py),
#   5. archive results.
# Designed to be launched detached; all output -> $AUTO/auto.log. Re-runnable.
set -u
SC=/home/mert/p-bit_quantum_decoder/results/ler_vs_p_bb144_2026-06/scripts
RESDIR=/home/mert/p-bit_quantum_decoder/results/ler_vs_p_bb144_2026-06
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
PHASE3=/tmp/ftune/phase3
AUTO=/tmp/zauto; mkdir -p "$AUTO"
LOG="$AUTO/auto.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== filtered_auto start ==="

# 1. wait for phase-3 tuning (8 configs) -- timeout 90 min, then proceed with what we have
for i in $(seq 1 180); do
  n=$(grep -l DONE "$PHASE3"/*.log 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && { say "phase3 complete ($n/8)"; break; }
  [ "$i" -eq 180 ] && say "phase3 timeout at $n/8; proceeding with available data"
  sleep 30
done

# 2. pick sweeps knee (rationale -> log); pool full phase3 table for the record
$PY "$SC/filtered_tune_pool.py" "$PHASE3" 5 2>/dev/null | tee "$AUTO/phase3_pool.txt" >> "$LOG"
SW=$($PY "$SC/filtered_pick_sweeps.py" "$PHASE3" 5 2> "$AUTO/pick.txt")
cat "$AUTO/pick.txt" >> "$LOG"
case "$SW" in ''|*[!0-9]*) SW=1000; say "selector gave no int; defaulting SWEEPS=1000";; esac
say "SELECTED sweeps=$SW (rho=0.20 chains=128) for campaign"

# 3. run the campaign (loops p=0.005..0.002, resume-safe). This is the long pole.
say "launching filtered campaign (memory_Z basis Z) ..."
SWEEPS=$SW RHO=0.20 CHAINS=128 OUT=/tmp/zcamp GPUS="0 1 2 3 4 5 6 7" \
  bash "$SC/filtered_campaign.sh" >> "$LOG" 2>&1
say "campaign finished; RESULT.txt:"
cat /tmp/zcamp/RESULT.txt | tee -a "$LOG"

# 4. comparison plot (ours-filtered vs RelayBP-300sets vs ours-correlated)
$PY "$SC/make_compare_plot.py" /tmp/zcamp/RESULT.txt "$RESDIR/plots" >> "$LOG" 2>&1
say "comparison plot written to $RESDIR/plots/"

# 5. archive
cp /tmp/zcamp/RESULT.txt   "$RESDIR/FILTERED_RESULT.txt" 2>/dev/null
cp /tmp/zcamp/progress.txt "$RESDIR/logs/filtered_progress.txt" 2>/dev/null
cp "$AUTO/phase3_pool.txt" "$RESDIR/logs/filtered_phase3_pool.txt" 2>/dev/null
say "=== filtered_auto DONE (sweeps=$SW) ==="
touch "$AUTO/AUTODONE"
