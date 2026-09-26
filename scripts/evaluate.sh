#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# < 2 )); then
  echo "Usage: bash scripts/evaluate.sh CONFIG CHECKPOINT [extra flags]" >&2
  exit 2
fi
config="$1"; checkpoint="$2"
shift 2
python main.py --config "$config" --mode evaluate --checkpoint "$checkpoint" "$@"
