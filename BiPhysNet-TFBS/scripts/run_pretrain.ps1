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
$EXP_NAME = if ($env:EXP_NAME) { $env:EXP_NAME } else { "pretrain_ITC_MLM_42d" }

$TRAIN_CSV = if ($env:TRAIN_CSV) { $env:TRAIN_CSV } else { "train_subset.csv" }
$VALID_CSV = if ($env:VALID_CSV) { $env:VALID_CSV } else { "valid_subset.csv" }
$TRAIN_NPY = if ($env:TRAIN_NPY) { $env:TRAIN_NPY } else { "$DATA_DIR\train_subset_features_42d.npy" }
$VALID_NPY = if ($env:VALID_NPY) { $env:VALID_NPY } else { "$DATA_DIR\valid_subset_features_42d.npy" }

python -m biphysnet.train_pretrain `
  --input_dir "$DATA_DIR" `
  --output_dir "$OUTPUT_DIR\phase1" `
  --exp_name "$EXP_NAME" `
  --pretrained_model_name "$MODEL_PATH" `
  --train_csv "$TRAIN_CSV" `
  --valid_csv "$VALID_CSV" `
  --train_features_npy "$TRAIN_NPY" `
  --valid_features_npy "$VALID_NPY" `
  --feature_dim 42 `
  --kmer_len 6 `
  --max_len 128 `
  --batch_size 128 `
  --accumulate_grad_batches 1 `
  --num_workers 16 `
  --num_epochs 30 `
  --learning_rate 2e-4 `
  --weight_decay 1e-5 `
  --warmup_steps 100 `
  --accelerator gpu `
  --devices "0,1" `
  --mlm_lambda 1.0
