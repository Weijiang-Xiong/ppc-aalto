#!/bin/bash
set -e
cat > /box/cp.cc
chmod a-w /box/cp.cc

cd /program
/program/.ppc/grader.py --file "/box/cp.cc" --binary "/box/cp" --json "$@"
