#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-}"

if [[ -z "${DRAFT_MODEL}" ]]; then
  echo "Set DRAFT_MODEL to a local FlexDraft checkpoint path when the checkpoint is available." >&2
  exit 1
fi

python scripts/inference.py \
  --model-name-or-path "${TARGET_MODEL}" \
  --draft-name-or-path "${DRAFT_MODEL}" \
  --block-size 16 \
  --dataset gsm8k \
  --max-samples 5 \
  --max-new-tokens 256 \
  --temperature 0 \
  --draft-confidence-threshold 0.01 \
  --pruning-strategy cumulative_product
