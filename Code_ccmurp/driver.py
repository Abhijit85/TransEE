#!/usr/bin/python3
"""
Dedicated CCMuRP entrypoint.

This launcher keeps CCMuRP runs isolated under Code_ccmurp/ while reusing
core training logic and dependencies from Code/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_CODE_DIR = os.path.join(_REPO_ROOT, 'Code')

# Force imports (model/dataloader/driver) to resolve from Code/.
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from driver import parse_args, main  # type: ignore


if __name__ == '__main__':
    args = parse_args()
    if getattr(args, 'model', None) not in {'MuRP', 'CCMuRP', 'BKRelatE', 'ARelatE', 'CCRelatE', 'RelatE'}:
        raise ValueError('Code_ccmurp/driver.py is intended for MuRP/CCMuRP-family runs.')
    main(args)
