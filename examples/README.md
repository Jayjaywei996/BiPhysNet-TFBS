# Examples

This directory provides tiny toy CSV files showing the expected input format.

## Files

- `toy_train.csv`: toy training CSV.
- `toy_valid.csv`: toy validation CSV.
- `toy_test.csv`: toy test CSV.

## Fine-tuning CSV format

```csv
sequence,label,cell_type
ATCGATCGATCGATCGATCGATCGATCGATCG,1,A549
GCGCGCGCATATATATATATATATATATATA,0,A549
```

## Pretraining CSV format

Pretraining requires only the `sequence` column. The toy fine-tuning CSVs include `label` and `cell_type`, but the pretraining data loader only reads `sequence`.

## 42D feature arrays

Real runs require `.npy` files with shape:

```text
num_samples × 42
```

To generate tiny placeholder arrays for local format checks:

```bash
python tools/make_toy_features.py
```

The generated `.npy` files are stored in `examples/generated_features/` and are ignored by Git. They are not biologically meaningful.
