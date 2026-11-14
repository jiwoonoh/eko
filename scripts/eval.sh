#!/usr/bin/env bash
set -euo pipefail

for leads in single-lead-II-gpt; do
  uv run python -m benchmark.score \
    --config "benchmark/experiments/${leads}.json" \
    "$@"
done
