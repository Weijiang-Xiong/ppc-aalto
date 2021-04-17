#!/bin/bash
set -e
cat > /box/prereq.cc
chmod a-w /box/prereq.cc

cd /program
/program/.ppc/grader.py --file "/box/prereq.cc" --binary "/box/prereq" --json "$@"
