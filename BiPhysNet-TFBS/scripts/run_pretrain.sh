#!/usr/bin/env bash
set -euo pipefail

# Safe defaults for multi-GPU environments that may have NCCL/NVML issues.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_NVML_DISABLE=${NCCL_NVML_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data/scorpio_42d_dataset_v2"}
MODEL_PATH=${MODEL_PATH:-"${PROJECT_ROOT}/pretrained/MetaBERTa-local"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs"}
EXP_NAME=${EXP_NAME:-"pretrain_ITC_MLM_42d"}

TRAIN_CSV=${TRAIN_CSV:-"train_subset.csv"}
VALID_CSV=${VALID_CSV:-"valid_subset.csv"}
TRAIN_NPY=${TRAIN_NPY:-"${DATA_DIR}/train_subset_features_42d.npy"}
VALID_NPY=${VALID_NPY:-"${DATA_DIR}/valid_subset_features_42d.npy"}

python -m biphysnet.train_pretrain \
  --input_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}/phase1" \
  --exp_name "${EXP_NAME}" \
  --pretrained_model_name "${MODEL_PATH}" \
  --train_csv "${TRAIN_CSV}" \
  --valid_csv "${VALID_CSV}" \
  --train_features_npy "${TRAIN_NPY}" \
  --valid_features_npy "${VALID_NPY}" \
  --feature_dim 42 \
  --kmer_len 6 \
  --max_len 128 \
  --batch_size 128 \
  --accumulate_grad_batches 1 \
  --num_workers 16 \
  --num_epochs 30 \
  --learning_rate 2e-4 \
  --weight_decay 1e-5 \
  --warmup_steps 100 \
  --accelerator gpu \
  --devices "0,1" \
  --mlm_lambda 1.0
