#!/usr/bin/env bash
set -euo pipefail

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
PHASE1_CKPT=${PHASE1_CKPT:-"FROM_SCRATCH"}
EXP_NAME=${EXP_NAME:-"finetune_42d"}

python -m biphysnet.train_finetune \
  --input_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}/phase2" \
  --exp_name "${EXP_NAME}" \
  --phase1_checkpoint "${PHASE1_CKPT}" \
  --pretrained_model_name "${MODEL_PATH}" \
  --train_csv "${TRAIN_CSV:-train_subset.csv}" \
  --valid_csv "${VALID_CSV:-valid_subset.csv}" \
  --test_csv "${TEST_CSV:-test_subset.csv}" \
  --md_ep_train_file "${TRAIN_NPY_NAME:-train_subset_features_42d.npy}" \
  --md_ep_valid_file "${VALID_NPY_NAME:-valid_subset_features_42d.npy}" \
  --md_ep_test_file "${TEST_NPY_NAME:-test_subset_features_42d.npy}" \
  --extra_feature_dim 42 \
  --freeze_layers 0 \
  --kmer_len 6 \
  --max_len 128 \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 3e-5 \
  --head_lr_mult 3.0 \
  --devices "0,1" \
  --accumulate_grad_batches 4 \
  --num_workers 2 \
  --precision 16 \
  --consistency_weight 0.2 \
  --use_focal \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
