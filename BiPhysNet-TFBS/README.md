# BiPhysNet-TFBS

**BiPhysNet-TFBS** is a research-oriented deep learning project for transcription factor binding site (TFBS) prediction. It integrates DNA sequence representations with 42-dimensional DNA biophysical features through cross-modal pretraining and adaptive sequence-structure gating.

This repository is organized as a **GitHub-ready research prototype**: it contains modular source code, reproducible launch scripts, example data formats, method documentation, benchmark notes, and lightweight unit tests. Large datasets, pretrained language-model weights, feature matrices, and checkpoints are intentionally excluded.

## Project Overview

TFBS prediction is often dominated by sequence similarity. However, DNA binding also depends on local structural and thermodynamic properties. BiPhysNet-TFBS models two complementary views of each DNA segment:

1. a **sequence branch**, based on overlapping k-mer tokens and a MetaBERTa-compatible DNA language model;
2. a **biophysical branch**, based on 42-dimensional DNA composition, shape-related, and thermodynamic features.

The project follows a two-stage pipeline:

- **Stage 1: Cross-modal pretraining** with masked language modeling (MLM) and sequence-structure contrastive alignment (ITC).
- **Stage 2: TFBS fine-tuning** with adaptive gating to combine sequence and biophysical representations for binary classification.

## Key Features

- DNA k-mer tokenization and sequence augmentation utilities.
- 42D biophysical feature branch for structure-aware modeling.
- MLM + ITC cross-modal pretraining objective.
- Adaptive sequence-structure gating in downstream fine-tuning.
- PyTorch Lightning implementation with multi-GPU launch scripts.
- Windows PowerShell and Linux/Git Bash entry points.
- Example CSV files and data-format documentation.
- Lightweight tests for tokenizer and loss functions.
- Original experiment scripts preserved in `legacy/` for traceability.

## Method Pipeline

```text
DNA sequence
├── k-mer tokenizer ──> MetaBERTa encoder ──> sequence embedding
└── 42D biophysical feature extraction ──> physical encoder ──> physical embedding

Stage 1: MLM + ITC cross-modal pretraining
Stage 2: adaptive sequence-structure gating
Output : TFBS / non-TFBS prediction
```

A Mermaid version of the workflow is available in [`docs/figures/pipeline.md`](docs/figures/pipeline.md).

## Repository Structure

```text
BiPhysNet-TFBS/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── configs/
│   ├── pretrain.yaml
│   └── finetune.yaml
├── scripts/
│   ├── run_pretrain.sh
│   ├── run_finetune.sh
│   ├── run_pretrain.ps1
│   └── run_finetune.ps1
├── src/
│   └── biphysnet/
│       ├── tokenizer.py
│       ├── losses.py
│       ├── modules.py
│       ├── data_pretrain.py
│       ├── data_finetune.py
│       ├── model_pretrain.py
│       ├── model_finetune.py
│       ├── train_pretrain.py
│       └── train_finetune.py
├── examples/
├── docs/
├── results/
├── tests/
├── tools/
└── legacy/
```

## Installation

```bash
conda create -n biphysnet python=3.10
conda activate biphysnet
pip install -r requirements.txt
pip install -e .
```

For local development without installation, set `PYTHONPATH` manually:

```powershell
# PowerShell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

```bash
# Linux / Git Bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Data Format

### Fine-tuning CSV

Fine-tuning CSV files should contain at least:

```csv
sequence,label,cell_type
ATCGATCGATCG,1,A549
GCGCGCATATA,0,A549
```

Required columns:

- `sequence`: DNA sequence string.
- `label`: binary label, where `1` indicates TFBS and `0` indicates non-TFBS.

Optional column:

- `cell_type`: cell type or sample group used for grouped evaluation.

### Pretraining CSV

Pretraining only requires:

```csv
sequence
ATCGATCGATCG
GCGCGCATATA
```

### 42D feature arrays

The 42D feature files should be stored as NumPy arrays with shape:

```text
num_samples × 42
```

The row order in the `.npy` file must match the row order in the corresponding CSV file.

## Quick Start

This repository does not include real DeepSEA data, MetaBERTa weights, `.npy` feature matrices, or checkpoints. The fastest project-level check is therefore:

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests
```

To generate small toy 42D feature arrays for local format checks:

```bash
python tools/make_toy_features.py
```

This creates toy `.npy` files under `examples/generated_features/`. These files are for format validation only and are ignored by Git.

## Stage 1: Cross-modal Pretraining

### PowerShell

```powershell
cd E:\BiPhysNet-TFBS
$env:DATA_DIR="E:\path\to\scorpio_42d_dataset_v2"
$env:MODEL_PATH="E:\path\to\MetaBERTa-local"
.\scripts\run_pretrain.ps1
```

### Linux / Git Bash

```bash
cd /path/to/BiPhysNet-TFBS
DATA_DIR=/path/to/scorpio_42d_dataset_v2 \
MODEL_PATH=/path/to/MetaBERTa-local \
bash scripts/run_pretrain.sh
```

## Stage 2: TFBS Fine-tuning

### PowerShell

```powershell
cd E:\BiPhysNet-TFBS
$env:DATA_DIR="E:\path\to\scorpio_42d_dataset_v2"
$env:MODEL_PATH="E:\path\to\MetaBERTa-local"
$env:PHASE1_CKPT="E:\path\to\pretrain.ckpt"
.\scripts\run_finetune.ps1
```

### Linux / Git Bash

```bash
cd /path/to/BiPhysNet-TFBS
DATA_DIR=/path/to/scorpio_42d_dataset_v2 \
MODEL_PATH=/path/to/MetaBERTa-local \
PHASE1_CKPT=/path/to/pretrain.ckpt \
bash scripts/run_finetune.sh
```

## Evaluation Metrics

The fine-tuning module supports common binary classification metrics:

- AUROC
- AUPR
- Accuracy
- F1 score
- MCC
- Sensitivity / Recall
- Specificity

## Results Summary

A concise benchmark note is available in [`results/benchmark_summary.md`](results/benchmark_summary.md).

The reported values should be treated as **internal DeepSEA-style benchmark results** and should be independently reproduced before making external SOTA claims.

## Notes on Data and Model Weights

The following files are intentionally not included in this repository:

- DeepSEA raw data or processed full benchmark files.
- MetaBERTa / DNA language-model weights.
- 42D `.npy` feature matrices.
- Model checkpoints such as `.ckpt`, `.pt`, or `.pth`.
- Full training logs, TensorBoard logs, and W&B runs.
- Unpublished manuscript or patent draft PDFs.

Use environment variables such as `DATA_DIR`, `MODEL_PATH`, `PHASE1_CKPT`, and `OUTPUT_DIR` to point to local resources.

## Limitations

- This repository is a research prototype rather than a production package.
- Full reproduction requires external datasets and pretrained model weights.
- Benchmark values depend on the exact dataset split, feature-generation pipeline, and checkpoint selection.
- The toy examples are only for validating file formats and code organization.

## Resume Description

**Chinese:**

> 构建 BiPhysNet-TFBS 转录因子结合位点预测框架，融合 MetaBERTa DNA 序列语义表示与 42 维 DNA 生物物理特征；设计 MLM + ITC 跨模态预训练实现序列与结构表征对齐，并在下游分类中引入自适应门控融合机制用于序列-结构一致性校验和假阳性抑制。基于 PyTorch Lightning 完成模块化训练代码、双阶段训练脚本、评估指标、示例数据、结果说明与单元测试。

**English:**

> Developed BiPhysNet-TFBS, a PyTorch Lightning framework for TFBS prediction that integrates MetaBERTa-based DNA sequence embeddings with 42-dimensional biophysical features. Designed a two-stage pipeline with MLM and contrastive pretraining, followed by adaptive sequence-structure gating for downstream binary classification. Modularized the codebase with reproducible scripts, configuration files, examples, benchmark notes, and unit tests.

## Citation

See [`CITATION.cff`](CITATION.cff).
