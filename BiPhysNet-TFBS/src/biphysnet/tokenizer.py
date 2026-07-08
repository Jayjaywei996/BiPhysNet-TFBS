"""Tokenization and DNA sequence augmentation utilities for BiPhysNet-TFBS."""
from __future__ import annotations

import random
from typing import Dict

import pandas as pd
import torch


class KmerTokenizer:
    """A lightweight overlapping k-mer tokenizer for DNA sequences."""

    def __init__(self, kmerlen: int = 6, overlapping: bool = True, maxlen: int = 512):
        self.k = kmerlen
        self.overlapping = overlapping
        self.maxlen = maxlen
        self.vocab = self._build_vocab(kmerlen)
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4

    def _build_vocab(self, k: int) -> Dict[str, int]:
        bases = ["A", "T", "C", "G"]
        vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
        idx = 5
        queue = [""]
        generated = {""}

        for _ in range(k + 1):
            next_queue = []
            for current_mer in queue:
                if len(current_mer) == k:
                    if current_mer not in vocab:
                        vocab[current_mer] = idx
                        idx += 1
                elif len(current_mer) < k:
                    for base in bases:
                        next_mer = current_mer + base
                        if next_mer not in generated:
                            generated.add(next_mer)
                            next_queue.append(next_mer)
            queue = next_queue

        for current_mer in queue:
            if len(current_mer) == k and current_mer not in vocab:
                vocab[current_mer] = idx
                idx += 1
        return vocab

    def kmer_tokenize(self, sequence: str) -> list[int]:
        sequence = sequence.upper().strip()
        tokens = []
        step = 1 if self.overlapping else self.k
        seq_len = len(sequence)
        for i in range(0, seq_len - self.k + 1, step):
            kmer = sequence[i : i + self.k]
            tokens.append(self.vocab.get(kmer, self.unk_token_id))
        return tokens

    def encode(self, sequence: str) -> torch.Tensor:
        if not sequence or pd.isna(sequence):
            return torch.full((self.maxlen,), self.pad_token_id, dtype=torch.long)
        token_ids = self.kmer_tokenize(str(sequence))
        if len(token_ids) > self.maxlen - 2:
            token_ids = token_ids[: self.maxlen - 2]
        final_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        padding_len = self.maxlen - len(final_ids)
        final_ids += [self.pad_token_id] * padding_len
        return torch.tensor(final_ids, dtype=torch.long)


def reverse_complement(seq: str) -> str:
    if not isinstance(seq, str):
        seq = str(seq)
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    seq = seq.upper().strip()
    return "".join(complement.get(base, base) for base in reversed(seq))


def random_kmer_mask(token_ids: torch.Tensor, mask_token_id: int, k: int, p: float = 0.15) -> torch.Tensor:
    del k  # kept for backward-compatible API
    masked_ids = token_ids.clone()
    indices = torch.where(token_ids > 4)[0]
    if len(indices) == 0:
        return masked_ids
    num_to_mask = int(len(indices) * p)
    if num_to_mask == 0:
        return masked_ids
    num_to_mask = min(num_to_mask, len(indices))
    mask_indices = indices[torch.randperm(len(indices))[:num_to_mask]]
    masked_ids[mask_indices] = mask_token_id
    return masked_ids


def random_nucleotide_dropout(sequence: str, p: float = 0.1) -> str:
    if not isinstance(sequence, str):
        sequence = str(sequence)
    seq_list = list(sequence)
    dropped_seq = [base for base in seq_list if random.random() > p]
    return "".join(dropped_seq)


def get_augmentations(sequence: str, tokenizer: KmerTokenizer, k_mask: int = 3, p_mask: float = 0.15, p_dropout: float = 0.1) -> torch.Tensor:
    if not sequence:
        raise ValueError("Cannot augment empty sequence")
    aug_type = random.choice(["mask", "dropout", "revcomp"])
    if aug_type == "mask":
        token_ids = tokenizer.encode(sequence)
        augmented_ids = random_kmer_mask(token_ids, tokenizer.mask_token_id, k=k_mask, p=p_mask)
    elif aug_type == "dropout":
        aug_seq = random_nucleotide_dropout(sequence, p=p_dropout)
        if not aug_seq:
            aug_seq = sequence
        augmented_ids = tokenizer.encode(aug_seq)
    else:
        aug_seq = reverse_complement(sequence)
        augmented_ids = tokenizer.encode(aug_seq)
    return augmented_ids
