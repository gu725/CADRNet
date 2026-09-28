#!/usr/bin/env bash

# Batch training commands for the five datasets used by CADRNet.
#
# Expected dataset layout:
#   $DATASET_ROOT/LEVIR-CD/train.txt
#   $DATASET_ROOT/LEVIR-CD/val.txt
#   ...
#
# Each txt line should contain:
#   path/to/time1_image path/to/time2_image path/to/mask
#
# Example:
#   DATASET_ROOT=/home/kairui/datasets bash scripts/train_all_datasets.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/home/kairui/datasets}"
WORK_DIR="${WORK_DIR:-work_dirs_cadr_gdcr_le}"
BACKBONE="${BACKBONE:-convnext_base}"
EXCHANGE="${EXCHANGE:-le}"
MAX_STEPS="${MAX_STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0003}"
MIN_LR="${MIN_LR:-0.00003}"
WARMUP="${WARMUP:-3000}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GDCR_LOSS_WEIGHT="${GDCR_LOSS_WEIGHT:-0.05}"
CDAM_LOSS_WEIGHT="${CDAM_LOSS_WEIGHT:-1.0}"

run_experiment() {
    echo
    echo "============================================="
    echo "[Run] $*"
    echo "============================================="
    if "$@"; then
        echo "[OK] finished"
    else
        echo "[ERROR] failed, continue to next experiment"
    fi
}

train_dataset() {
    local name="$1"
    local src_size="$2"
    local crop_size="$3"
    local dataset_path="${DATASET_ROOT}/${name}"
    local exp_name="${name}_${SEED}_CADRNet_${BACKBONE}_${EXCHANGE}_cdam_gdcr_msprior_loss${GDCR_LOSS_WEIGHT}"

    run_experiment "${PYTHON_BIN}" tools/train.py \
        --dataset "${dataset_path}" \
        --backbone "${BACKBONE}" \
        --exchange "${EXCHANGE}" \
        --decoder-scale single \
        --crop-size "${crop_size}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --max-steps "${MAX_STEPS}" \
        --lr "${LR}" \
        --min-lr "${MIN_LR}" \
        --warmup "${WARMUP}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --work-dir "${WORK_DIR}" \
        --exp-name "${exp_name}" \
        --cdam-loss-weight "${CDAM_LOSS_WEIGHT}" \
        --gdcr-loss-weight "${GDCR_LOSS_WEIGHT}"
}

echo "[Config] DATASET_ROOT=${DATASET_ROOT}"
echo "[Config] WORK_DIR=${WORK_DIR}"
echo "[Config] BACKBONE=${BACKBONE}, EXCHANGE=${EXCHANGE}, DEVICE=${DEVICE}"

train_dataset "LEVIR-CD" 1024 256
train_dataset "MSRS-CD" 1024 256
train_dataset "S2Looking" 1024 256
train_dataset "WHU-CD" 256 256
train_dataset "SYSU-CD" 256 256
