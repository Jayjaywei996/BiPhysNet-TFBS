"""Generate tiny deterministic 42D toy feature arrays for format checks.

The generated arrays are not biologically meaningful. They are only used to
validate file paths and data-loading behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
OUT_DIR = EXAMPLES_DIR / "generated_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_features(csv_name: str, out_name: str) -> None:
    df = pd.read_csv(EXAMPLES_DIR / csv_name)
    rng = np.random.default_rng(42)
    features = rng.normal(loc=0.0, scale=1.0, size=(len(df), 42)).astype("float32")
    np.save(OUT_DIR / out_name, features)
    print(f"Wrote {OUT_DIR / out_name}: {features.shape}")


if __name__ == "__main__":
    make_features("toy_train.csv", "toy_train_features_42d.npy")
    make_features("toy_valid.csv", "toy_valid_features_42d.npy")
    make_features("toy_test.csv", "toy_test_features_42d.npy")
