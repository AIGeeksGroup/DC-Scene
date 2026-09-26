#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
(cd third_party/pointnet2 && python -m pip install . --no-build-isolation)
(cd utils && python cython_compile.py build_ext --inplace)
