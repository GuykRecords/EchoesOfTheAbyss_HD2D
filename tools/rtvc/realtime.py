#!/usr/bin/env python3
"""Launcher so the documented command still works verbatim:

    python realtime.py --engine passthrough --in-device 23 --out-device 22 ...

The implementation lives in the ``rtvc`` package next to this file; this shim
only makes sure that package is importable when the script is run directly
from any working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtvc.realtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
