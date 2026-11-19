#!/bin/sh
# Run every script once, without opening windows, saving the printed output next to the charts.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
for s in "$@"; do
  name=$(basename "$s" .py)
  start=$(date +%s)
  python -u -W ignore "$s" --no-show > "out_$name.txt" 2>&1
  echo "$s exit $? in $(( $(date +%s) - start ))s"
done
