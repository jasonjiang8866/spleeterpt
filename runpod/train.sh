#!/usr/bin/env bash
set -euo pipefail

# Run this only after creating a GPU Pod and attaching persistent storage.
# All paths can be overridden. No API keys or cloud provisioning are used.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to your training JSONL}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST to your validation JSONL}"
: "${RUN_DIR:?Set RUN_DIR to a directory on your persistent volume}"
config_path="${CONFIG:-configs/2stems.json}"
precision="${PRECISION:-fp32}"
batch_size="${BATCH_SIZE:-1}"

uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
uv run --no-sync spleeterpt train \
    --config "$config_path" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --output "$RUN_DIR" \
    --device cuda --precision "$precision" --batch-size "$batch_size" \
    --workers "${WORKERS:-4}" --epochs "${EPOCHS:-100}" "$@"
