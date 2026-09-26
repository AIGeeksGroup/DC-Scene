#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# < 3 )); then
  echo "Usage: bash scripts/score.sh CONFIG CHECKPOINT OUTPUT [extra flags]" >&2
  exit 2
fi
config="$1"; checkpoint="$2"; output="$3"
shift 3
python main.py --config "$config" --mode score --checkpoint "$checkpoint" --output "$output" "$@"
