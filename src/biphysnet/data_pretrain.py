"""Data module for BiPhysNet cross-modal pretraining."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class CrossModalDataset(Dataset):
    """Pairs DNA sequences with precomputed 42D biophysical features."""

    def __init__(self, csv_file, feature_file, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size

        print("--- [Dataset] ---")
        print(f"  Seq: {csv_file}")
        print(f"  Feat: {feature_file}")

        self.data = pd.read_csv(
            csv_file,
            header=0,
            usecols=["sequence"],
            sep=",",
            on_bad_lines="warn",
            engine="python",
            quoting=3,
            dtype=str,
        )
        self.data = self.data.dropna(subset=["sequence"])
        self.sequences = self.data["sequence"].astype(str).tolist()

        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Missing: {feature_file}")
        self.features = np.load(feature_file).astype(np.float32)

        if len(self.sequences) != len(self.features):
            raise ValueError(f"Mismatch: Seq={len(self.sequences)}, Feat={len(self.features)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if idx >= len(self.sequences):
            raise IndexError
        seq_str = str(self.sequences[idx])
        feat_vec = torch.tensor(self.features[idx], dtype=torch.float32)

        try:
            token_ids = torch.as_tensor(self.tokenizer.encode(seq_str), dtype=torch.long)
        except Exception:
            return None

        mlm_input_ids = token_ids.clone()
        mlm_labels = torch.full_like(token_ids, -100)
        valid_indices = torch.where(mlm_input_ids > 4)[0]

        if len(valid_indices) > 0:
            num_mask = max(1, int(len(valid_indices) * self.mlm_probability))
            mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_mask]]
            mlm_labels[mask_indices] = mlm_input_ids[mask_indices]
            for idx_mask in mask_indices:
                prob = torch.rand(1).item()
                if prob < 0.8:
                    mlm_input_ids[idx_mask] = self.mask_token_id
                elif prob < 0.9:
                    mlm_input_ids[idx_mask] = torch.randint(5, self.vocab_size, (1,)).item()

        return token_ids, feat_vec, mlm_input_ids, mlm_labels


def cross_modal_collate_fn(batch):
    batch = [x for x in batch if x is not None and x[0] is not None]
    if not batch:
        return None
    seqs = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    feats = torch.stack([x[1] for x in batch])
    mlm_in = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)
    mlm_lbl = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=-100)
    return seqs, feats, mlm_in, mlm_lbl


class CrossModalDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        del stage
        t_csv = Path(self.args.input_dir) / self.args.train_csv
        v_csv = Path(self.args.input_dir) / self.args.valid_csv
        self.train_dataset = CrossModalDataset(t_csv, self.args.train_features_npy, self.tokenizer)
        self.val_dataset = CrossModalDataset(v_csv, self.args.valid_features_npy, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=cross_modal_collate_fn,
            persistent_workers=(self.args.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=cross_modal_collate_fn,
            persistent_workers=(self.args.num_workers > 0),
        )
