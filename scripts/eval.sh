#!/usr/bin/env bash
set -euo pipefail

uv run python -m benchmark.score \
  --config benchmark/experiments/single-lead-II-gpt.json \
  "$@"
