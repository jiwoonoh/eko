#!/usr/bin/env bash
set -euo pipefail

exec uv run python -m benchmark.run \
  --config benchmark/experiments/single-lead-II-qwen3.5-9b.json \
  --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
  --ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
  "$@"
