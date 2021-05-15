#!/bin/bash
set -e
cat > /box/so.cc
chmod a-w /box/so.cc

cd /program
/program/.ppc/grader.py --file "/box/so.cc" --binary "/box/so" --json "$@"
