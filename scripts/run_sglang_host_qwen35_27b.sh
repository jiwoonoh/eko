#!/usr/bin/env bash

if [ -f "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh"
fi

# Exempt loopback from the company proxy.
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

export FLASHINFER_USE_CUDA_NORM=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-3}"

MODEL=${MODEL:-Qwen/Qwen3.5-27B}
PORT=${PORT:-30005}

# Fallback if CUDA graphs or FlashInfer fail:
# sglang serve --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --disable-cuda-graph --disable-piecewise-cuda-graph --sampling-backend pytorch --attention-backend triton --disable-overlap-schedule

MULTINODE=${NNODES:+--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $DIST_INIT_ADDR}

sglang serve \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size "${TP:-1}" --dp-size "${DP:-1}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_BS:-256}" \
    ${MEM_FRAC:+--mem-fraction-static $MEM_FRAC} $MULTINODE
