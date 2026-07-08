$ErrorActionPreference = "Stop"

$env:NCCL_DEBUG = if ($env:NCCL_DEBUG) { $env:NCCL_DEBUG } else { "INFO" }
$env:NCCL_NVML_DISABLE = if ($env:NCCL_NVML_DISABLE) { $env:NCCL_NVML_DISABLE } else { "1" }
$env:NCCL_IB_DISABLE = if ($env:NCCL_IB_DISABLE) { $env:NCCL_IB_DISABLE } else { "1" }
$env:NCCL_P2P_DISABLE = if ($env:NCCL_P2P_DISABLE) { $env:NCCL_P2P_DISABLE } else { "1" }
$env:TORCH_NCCL_BLOCKING_WAIT = if ($env:TORCH_NCCL_BLOCKING_WAIT) { $env:TORCH_NCCL_BLOCKING_WAIT } else { "1" }
$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "expandable_segments:True" }

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PYTHONPATH = "$PROJECT_ROOT\src;$env:PYTHONPATH"

$DATA_DIR = if ($env:DATA_DIR) { $env:DATA_DIR } else { "$PROJECT_ROOT\data\scorpio_42d_dataset_v2" }
$MODEL_PATH = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "$PROJECT_ROOT\pretrained\MetaBERTa-local" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "$PROJECT_ROOT\outputs" }
$PHASE1_CKPT = if ($env:PHASE1_CKPT) { $env:PHASE1_CKPT } else { "FROM_SCRATCH" }
$EXP_NAME = if ($env:EXP_NAME) { $env:EXP_NAME } else { "finetune_42d" }

$TRAIN_CSV = if ($env:TRAIN_CSV) { $env:TRAIN_CSV } else { "train_subset.csv" }
$VALID_CSV = if ($env:VALID_CSV) { $env:VALID_CSV } else { "valid_subset.csv" }
$TEST_CSV = if ($env:TEST_CSV) { $env:TEST_CSV } else { "test_subset.csv" }
$TRAIN_NPY_NAME = if ($env:TRAIN_NPY_NAME) { $env:TRAIN_NPY_NAME } else { "train_subset_features_42d.npy" }
$VALID_NPY_NAME = if ($env:VALID_NPY_NAME) { $env:VALID_NPY_NAME } else { "valid_subset_features_42d.npy" }
$TEST_NPY_NAME = if ($env:TEST_NPY_NAME) { $env:TEST_NPY_NAME } else { "test_subset_features_42d.npy" }

python -m biphysnet.train_finetune `
  --input_dir "$DATA_DIR" `
  --output_dir "$OUTPUT_DIR\phase2" `
  --exp_name "$EXP_NAME" `
  --phase1_checkpoint "$PHASE1_CKPT" `
  --pretrained_model_name "$MODEL_PATH" `
  --train_csv "$TRAIN_CSV" `
  --valid_csv "$VALID_CSV" `
  --test_csv "$TEST_CSV" `
  --md_ep_train_file "$TRAIN_NPY_NAME" `
  --md_ep_valid_file "$VALID_NPY_NAME" `
  --md_ep_test_file "$TEST_NPY_NAME" `
  --extra_feature_dim 42 `
  --freeze_layers 0 `
  --kmer_len 6 `
  --max_len 128 `
  --batch_size 64 `
  --num_epochs 30 `
  --learning_rate 3e-5 `
  --head_lr_mult 3.0 `
  --devices "0,1" `
  --accumulate_grad_batches 4 `
  --num_workers 2 `
  --precision 16 `
  --consistency_weight 0.2 `
  --use_focal `
  --focal_alpha 0.25 `
  --focal_gamma 2.0
