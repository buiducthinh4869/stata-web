#!/bin/bash
# usage: tests/run_all.sh name.do ...   (runs each in a fresh workdir with example data)
cd "$(dirname "$0")/.."
for f in "$@"; do
  W=/tmp/stw_$(basename $f .do); rm -rf $W; mkdir -p $W; cp examples/data/*.dta $W/
  timeout 600 python3 tests/run_do.py examples/do/$f $W > /tmp/out_$(basename $f .do).txt 2>&1
  echo "== $f: $(tail -1 /tmp/out_$(basename $f .do).txt) | errs: $(grep -c '^r(' /tmp/out_$(basename $f .do).txt)"
done
