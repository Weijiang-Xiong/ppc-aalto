#!/bin/bash
set -e
cat > /box/mf.cc
chmod a-w /box/mf.cc

cd /program
/program/.ppc/grader.py --file "/box/mf.cc" --binary "/box/mf" --json "$@"
