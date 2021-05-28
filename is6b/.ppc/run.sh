#!/bin/bash
set -e
cat > /box/is.cu
chmod a-w /box/is.cu

# Fix nvprof
cat > "/etc/nsswitch.conf" <<EOT
passwd:         files systemd
EOT

cd /program
/program/.ppc/grader.py --file "/box/is.cu" --binary "/box/is" --json "$@"
