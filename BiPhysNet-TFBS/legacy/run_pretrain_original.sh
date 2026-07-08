#!/bin/bash
set -e

# ================= 1. 环境变量 (Gloo + 显存优化) =================
export NCCL_DEBUG=INFO
export NCCL_NVML_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= 2. 路径配置 =================
DATA_DIR_PHASE1="/home/wjw/resnetcopy/SimCLR-style/TransferDataSet/scorpio_42d_dataset_v2"
TRAIN_NPY_PATH="${DATA_DIR_PHASE1}/train_subset_features_42d.npy"
VALID_NPY_PATH="${DATA_DIR_PHASE1}/valid_subset_features_42d.npy"
TRAIN_CSV_NAME="train_subset.csv"
VALID_CSV_NAME="valid_subset.csv"

# [保持使用 MetaBERTa] 因为它其实比 DNA_bert_6 轻 (55M vs 109M)
MODEL_PATH="/home/wjw/resnetcopy/SimCLR-style/MetaBERTa-local"

OUTPUT_DIR="./training_output_42d_deepsea"
PHASE1_EXP_NAME="pretrain_ITC_MLM_SPEEDUP"
PHASE1_OUT_DIR="$OUTPUT_DIR/phase1"
PYTHON_SCRIPT="/home/wjw/resnetcopy/SimCLR-style/ph1_pretrain_42.py"

echo "--- [Stage 1] Start Training (SPEED MODE: Batch 256) ---"

# ================= 3. 运行 (暴力提速参数) =================
# Batch Size: 256 (3090 显存足够大，128长度随便跑)
# Workers: 16 (加快 CPU 读取数据)
# Accumulate: 1 (实时更新，不需要等待)

python "$PYTHON_SCRIPT" \
    --input_dir "$DATA_DIR_PHASE1" \
    --output_dir "$PHASE1_OUT_DIR" \
    --exp_name "$PHASE1_EXP_NAME" \
    --pretrained_model_name "$MODEL_PATH" \
    --train_features_npy "$TRAIN_NPY_PATH" \
    --valid_features_npy "$VALID_NPY_PATH" \
    \
    --train_csv "$TRAIN_CSV_NAME" \
    --valid_csv "$VALID_CSV_NAME" \
    \
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

echo "✅ Phase 1 Done."