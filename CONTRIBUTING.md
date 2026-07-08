# Contributing

Issues and suggestions are welcome.

## Guidelines

- Keep model logic reproducible and traceable.
- Do not commit large datasets, `.npy` feature matrices, checkpoints, or pretrained weights.
- Document any change that affects training behavior or reported metrics.
- Prefer small, focused pull requests.
- Run the project checks before submitting changes:

```bash
python -m py_compile src/biphysnet/*.py
python -m pytest tests
```
