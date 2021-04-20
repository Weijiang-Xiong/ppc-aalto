#!/usr/bin/env python3

import sys
if sys.version_info < (3, 6):
    sys.exit('Expected Python version 3.6 or later.')

from typing import List

import ppcgrader
import ppccp

if __name__ == "__main__":
    ppcgrader.cli(ppccp.Config(single_precision=False, openmp=False))
