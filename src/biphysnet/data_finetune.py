"""Data module for downstream TFBS fine-tuning."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset(Dataset):
    """TFBS binary classification dataset with optional 42D features."""

    def __init__(self, csv_file, tokenizer, md_ep_file=None):
        self.tokenizer = tokenizer
        self.md_ep_features = None
        self.cell_types = None

        print(f"--- Loading: {csv_file}")
        df = pd.read_csv(csv_file, header=0, sep=",", on_bad_lines="warn", engine="python", quoting=3, dtype=str)
        df = df.dropna(subset=["sequence", "label"])
        self.sequences = df["sequence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.cell_types = df["cell_type"].astype(str).tolist() if "cell_type" in df.columns else ["ALL"] * len(self.labels)

        if md_ep_file and os.path.exists(md_ep_file):
            print(f"--- Loading Features: {md_ep_file}")
            self.md_ep_features = np.load(md_ep_file).astype(np.float32)
            if len(self.md_ep_features) != len(self.sequences):
                raise ValueError(f"Mismatch: {len(self.md_ep_features)} vs {len(self.sequences)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.labels):
            raise IndexError
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cell = self.cell_types[idx]
        try:
            ids = torch.as_tensor(self.tokenizer.encode(str(seq)), dtype=torch.long)
        except Exception:
            return None, None, None
        if self.md_ep_features is not None:
            return (ids, torch.tensor(self.md_ep_features[idx], dtype=torch.float32)), label, cell
        return ids, label, cell


def classification_collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    if not batch:
        return None
    if isinstance(batch[0][0], tuple):
        ids = torch.nn.utils.rnn.pad_sequence([x[0][0] for x in batch], batch_first=True, padding_value=0)
        feats = torch.stack([x[0][1] for x in batch])
        labels = torch.stack([x[1] for x in batch])
        cells = [x[2] for x in batch]
        return (ids, feats), labels, cells

    ids = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    labels = torch.stack([x[1] for x in batch])
    cells = [x[2] for x in batch]
    return ids, labels, cells


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        base = Path(self.args.input_dir)
        cargs = {"tokenizer": self.tokenizer}
        t_csv = base / self.args.train_csv
        v_csv = base / self.args.valid_csv
        te_csv = base / self.args.test_csv
        t_npy = base / self.args.md_ep_train_file if self.args.md_ep_train_file else None
        v_npy = base / self.args.md_ep_valid_file if self.args.md_ep_valid_file else None
        te_npy = base / self.args.md_ep_test_file if self.args.md_ep_test_file else None

        if stage == "fit" or stage is None:
            self.train_dataset = ClassificationDataset(t_csv, md_ep_file=t_npy, **cargs)
            self.val_dataset = ClassificationDataset(v_csv, md_ep_file=v_npy, **cargs)
        if stage == "test" or stage is None:
            self.test_dataset = ClassificationDataset(te_csv, md_ep_file=te_npy, **cargs)

        if hasattr(self, "train_dataset"):
            lbls = np.array(self.train_dataset.labels)
            pos = lbls.sum()
            neg = len(lbls) - pos
            setattr(self.args, "pos_weight", float(neg) / pos if pos > 0 else 1.0)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=classification_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=classification_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=classification_collate_fn,
            pin_memory=True,
        )
