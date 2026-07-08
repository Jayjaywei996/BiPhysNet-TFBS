# Benchmark Summary

This file records the current internal benchmark summary for BiPhysNet-TFBS.

> Important: these numbers are from internal DeepSEA-style benchmark experiments and should be independently reproduced before making external SOTA claims.

## Reported Internal Results

| Model | AUROC | Specificity | Notes |
|---|---:|---:|---|
| MetaBERTa baseline | 0.908 | 0.764 | Sequence-only baseline reported in the project draft |
| BiPhysNet-TFBS | 0.921 | 0.910 | Sequence + 42D physical features with adaptive gating |

## Interpretation

The main observed improvement is in **specificity**, suggesting that the biophysical branch and gating mechanism may help reduce false-positive predictions caused by sequence similarity alone.

## Reproduction Notes

To make these results fully reproducible, record the following for each future run:

- dataset version and split strategy;
- feature-generation script and feature table version;
- pretrained model checkpoint;
- exact training command;
- random seed;
- software and CUDA versions;
- final validation and test metrics;
- checkpoint used for final evaluation.

Recommended future output files:

```text
results/
├── benchmark_summary.md
├── run_YYYYMMDD_metrics.csv
├── run_YYYYMMDD_command.txt
└── run_YYYYMMDD_notes.md
```
