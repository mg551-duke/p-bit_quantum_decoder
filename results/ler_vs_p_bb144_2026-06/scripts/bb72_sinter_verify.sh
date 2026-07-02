#!/usr/bin/env bash
# Verify the p-bit decoder via Stim/sinter (parallel CPU) reproduces the in-house
# GPU LER for BB [[72,12,6]] memory_Z, using the SAME decoder config as the
# campaigns (rho_scale=0.30, beta 0.5->6, sweeps=1000, chains=128, max_legs=24,
# target_valid=5 == the tv5 result).
#
# Track: FILTERED (BASIS=Z, default) or CORRELATED full-DEM (BASIS='' empty).
# Stim/sinter does the sampling + logical-error scoring; our decoder only decodes.
#
#   OUT=/tmp/bb72_sinter_filtZ.log setsid bash bb72_sinter_verify.sh > /tmp/bb72_sinter_filtZ_launch.log 2>&1 &
#   BASIS= OUT=/tmp/bb72_sinter_corr.log PS="0.005 0.004 0.003" setsid bash bb72_sinter_verify.sh > /tmp/... 2>&1 &
set -u
PY=/home/mert/p-bit_quantum_decoder/.venv-gpu/bin/python
ROOT=/home/mert/p-bit_quantum_decoder
TD="/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate"
SUF="distance=6,rounds=6,REP,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim"

BASIS=${BASIS-Z}         # '-' not ':-' so an explicitly empty BASIS (correlated) is honored
PROCS=${PROCS:-72}
OUT=${OUT:-/tmp/bb72_sinter_filtZ.log}
read -r -a PS <<< "${PS:-0.005 0.004 0.003}"
MULT=${MULT:-1}          # scale all shot counts (use MULT=2 for the lower-LER correlated track)
MAXERR=${MAXERR:-0}      # stop each point after this many logical errors (0 = only the shots cap)
# shots per p sized for ~250+ logical errors at the in-house filtered LER; correlated
# (lower LER) needs MULT>=2 to reach a comparable error count.
declare -A SH=( [0.005]=3000 [0.004]=6000 [0.003]=20000 [0.002]=120000 [0.001]=1500000 )

: > "$OUT"
label=$([ -n "$BASIS" ] && echo "FILTERED-$BASIS" || echo "CORRELATED(full-DEM)")
echo "BB72 sinter $label verify (parallel CPU) procs=$PROCS ps='${PS[*]}' start $(date)" >> "$OUT"
bf=(); [ -n "$BASIS" ] && bf=(--basis-filter "$BASIS")
me=(); [ "$MAXERR" -gt 0 ] && me=(--max-errors "$MAXERR")
for p in "${PS[@]}"; do
  circ="$TD/circuit=bicycle_bivariate_72_12_6_memory_Z,${SUF/REP/error_rate=$p}"
  if [ ! -f "$circ" ]; then echo "MISSING circuit p=$p" >> "$OUT"; continue; fi
  sh=$(( ${SH[$p]:-5000} * MULT ))
  echo "=== p=$p shots=$sh $(date) ===" >> "$OUT"
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/experiments/run_sinter_pbit.py" \
    --circuit "$circ" "${bf[@]}" "${me[@]}" \
    --shots "$sh" --procs "$PROCS" \
    --chains 128 --sweeps 1000 --beta-start 0.5 --beta-stop 6 \
    --rho-scale 0.30 --selection min_weight --max-legs 24 --target-valid 5 \
    >> "$OUT" 2>&1
done
echo "ALLDONE $(date)" >> "$OUT"
