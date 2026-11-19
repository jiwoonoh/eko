#!/usr/bin/env bash
set -euo pipefail

for experiment in 12-lead single-lead-I single-lead-II; do
  uv run python -m benchmark.score \
    --config "benchmark/experiments/${experiment}-gpt.json" \
    "$@"
done
