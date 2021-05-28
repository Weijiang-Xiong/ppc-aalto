#!/bin/bash
set -e
cat > /box/is.cc
chmod a-w /box/is.cc

cd /program
/program/.ppc/grader.py --file "/box/is.cc" --binary "/box/is" --json "$@"
