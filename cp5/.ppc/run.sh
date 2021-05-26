#!/bin/bash
set -e
cat > /box/cp.cu
chmod a-w /box/cp.cu

# Fix nvprof
cat > "/etc/nsswitch.conf" <<EOT
passwd:         files systemd
EOT

cd /program
/program/.ppc/grader.py --file "/box/cp.cu" --binary "/box/cp" --json "$@"
