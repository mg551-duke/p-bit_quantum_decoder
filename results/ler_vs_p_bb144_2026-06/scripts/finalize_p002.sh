#!/usr/bin/env bash
# Waits for the autonomous p=0.002 job to finish, then records its result into
# the artifact folder (copies result + progress, fills the summary.csv row).
set -u
D=/home/mert/p-bit_quantum_decoder/results/ler_vs_p_bb144_2026-06
RES=/tmp/p002_RESULT.txt
while [ ! -f "$RES" ]; do sleep 120; done
cp -f "$RES" "$D/p002_RESULT.txt"
cp -f /tmp/p002/progress.txt "$D/logs/p002_progress.txt" 2>/dev/null
line=$(grep -E '^errors=' "$RES")
E=$(echo "$line"  | grep -oE 'errors=[0-9]+'      | cut -d= -f2)
N=$(echo "$line"  | grep -oE 'shots=[0-9]+'       | cut -d= -f2)
MV=$(echo "$line" | grep -oE 'min_valid=[0-9.]+'  | cut -d= -f2)
LER=$(echo "$line" | sed -E 's/.*LER=([0-9.eE+-]+) \+- ([0-9.eE+-]+).*/\1/')
SE=$(echo "$line"  | sed -E 's/.*LER=([0-9.eE+-]+) \+- ([0-9.eE+-]+).*/\2/')
awk -v e="$E" -v n="$N" -v ler="$LER" -v se="$SE" -v mv="$MV" -F, 'BEGIN{OFS=","}
  $5=="0.002"{ $12=e; $13=n; $14=ler; $15=se; $16=mv; $21="done" } {print}' \
  "$D/summary.csv" > "$D/summary.csv.tmp" && mv "$D/summary.csv.tmp" "$D/summary.csv"
echo "$(date '+%F %T') p002 result recorded into artifact dir" >> /tmp/p002/progress.txt
