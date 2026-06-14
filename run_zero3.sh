#!/usr/bin/env bash
# Tek node / 2x L4 GPU uzerinde DeepSpeed ZeRO-3 (afterOptimization) benchmark.
set -e

export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
# Hangi config? Disaridan DEEPSPEED_CFG ve RUN_MODE export ederek sec.
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-$WORKSPACE_DIR/configs/config_auto.json}"
export RUN_MODE="${RUN_MODE:-ZERO3_AUTO}"
# export MODEL_ID="Qwen/Qwen2.5-0.5B"   # varsayilan zaten bu

cd "$WORKSPACE_DIR"
mkdir -p results logs configs

deepspeed --num_gpus 2 afterOptimization.py
