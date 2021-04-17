#!/usr/bin/env python3

import sys
if sys.version_info < (3,6):
    sys.exit('Expected Python version 3.6 or later.')

from typing import List

import ppcgrader
import ppcprereq

if __name__ == "__main__":
    ppcgrader.cli(ppcprereq.Config())
