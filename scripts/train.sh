#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/3dcoca_scanrefer.json}"
if (( $# > 0 )); then shift; fi
python main.py --config "$config" --mode train "$@"
