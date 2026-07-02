#!/usr/bin/env bash
# Pool campaign logs for a given tag prefix. Arg: TAG
TAG="$1"
files=$(ls /tmp/${TAG}_g*.log 2>/dev/null)
[ -z "$files" ] && { echo "no logs for $TAG"; exit 0; }
# errors/shots
cat $files | grep -oE '\([0-9]+/[0-9]+\)' | tr -d '()' | awk -F/ -v tag="$TAG" '
  {e+=$1; n+=$2}
  END{ if(n>0) printf "%-16s errors=%d shots=%d LER=%.3e +-%.2e\n", tag, e, n, e/n, sqrt((e/n)*(1-e/n)/n); else printf "%-16s (no results yet)\n", tag }'
# min validity across chunks (catches low-rho validity loss)
cat $files | grep -oE 'valid=[0-9.]+' | sed 's/valid=//' | sort -n | head -1 | awk -v tag="$TAG" '{printf "%-16s min_valid=%s\n", tag, $1}'
