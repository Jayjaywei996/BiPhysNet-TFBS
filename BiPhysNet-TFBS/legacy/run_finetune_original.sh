#!/bin/bash
set -e

# ================= 1. 环境变量 =================
export NCCL_DEBUG=INFO
export NCCL_NVML_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= 2. 路径配置 =================
DATA_DIR="/home/wjw/resnetcopy/SimCLR-style/TransferDataSet/scorpio_42d_dataset_v2"
FINETUNE_SCRIPT="/home/wjw/resnetcopy/SimCLR-style/ph2_42.py"
MODEL_PATH="/home/wjw/resnetcopy/SimCLR-style/MetaBERTa-local"

# [!!! 关键修改 1：指向刚才中断前最新的存档 !!!]
# 请务必检查路径是否正确，这是您 Phase 2 跑到一半生成的
# 使用 find 命令找到的那个路径填在这里：
PHASE1_CKPT="/home/wjw/resnetcopy/SimCLR-style/training_output_42d_deepsea/phase1/pretrain_ITC_MLM_SPEEDUP/pretrain-epoch=08-val_loss=0.485.ckpt"

OUTPUT_DIR="./training_output_42d_deepsea/phase2"
# 改名，表示这是续跑的
EXP_NAME="finetune_genome_subset_42d_RESUME" 

TRAIN_CSV="train_subset.csv"
TRAIN_NPY="train_subset_features_42d.npy"
VALID_CSV="valid_subset.csv"
VALID_NPY="valid_subset_features_42d.npy"
TEST_CSV="/home/wjw/resnetcopy/SimCLR-style/TransferDataSet/scorpio_42d_dataset_v2/test_subset.csv"
TEST_NPY="/home/wjw/resnetcopy/SimCLR-style/TransferDataSet/scorpio_42d_dataset_v2/test_subset_features_42d.npy"

echo "--- [Stage 2] Resuming Finetune (Safe Mode) ---"

# 双重检查 Checkpoint 是否存在
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "❌ 错误: 找不到续命存档: $PHASE1_CKPT"
    echo "请运行: find ./training_output_42d_deepsea/phase2 -name 'last.ckpt'"
    echo "然后把找到的路径填入脚本的 PHASE1_CKPT 变量中！"
    exit 1
fi

python "$FINETUNE_SCRIPT" \
    --input_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --phase1_checkpoint "$PHASE1_CKPT" \
    --pretrained_model_name "$MODEL_PATH" \
    \
    --train_csv "$TRAIN_CSV" \
    --valid_csv "$VALID_CSV" \
    --test_csv "$TEST_CSV" \
    \
    --md_ep_train_file "$TRAIN_NPY" \
    --md_ep_valid_file "$VALID_NPY" \
    --md_ep_test_file "$TEST_NPY" \
    \
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

echo "✅ Resume Done."