"""
contrastive_utils.py

包含 Scorpio 预训练和微调所需的核心工具函数，严格按照 SimCLR/NT-Xent 流程设计。
[!! 完整最终版 - 包含 NTXentLoss 动态批次修复 !!]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import pandas as pd # <--- [!! 修复 !!] 补上这个 import

# --- 1. K-mer Tokenizer ---
class KmerTokenizer:
    def __init__(self, kmerlen=6, overlapping=True, maxlen=512): # 默认 maxlen 改为 512
        self.k = kmerlen
        self.overlapping = overlapping
        self.maxlen = maxlen
        self.vocab = self._build_vocab(kmerlen)
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0; self.unk_token_id = 1; self.cls_token_id = 2;
        self.sep_token_id = 3; self.mask_token_id = 4

    def _build_vocab(self, k):
        bases = ['A', 'T', 'C', 'G']; vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3, '<MASK>': 4}; idx = 5
        queue = [""]; generated = {''}
        # 使用迭代代替递归以避免深度限制
        for i in range(k + 1):
            next_queue = []
            for current_mer in queue:
                if len(current_mer) == k:
                    if current_mer not in vocab: vocab[current_mer] = idx; idx += 1
                elif len(current_mer) < k:
                    for base in bases:
                        next_mer = current_mer + base
                        if next_mer not in generated:
                             generated.add(next_mer)
                             next_queue.append(next_mer)
            queue = next_queue
        # 处理最后一轮生成的 k-mers
        for current_mer in queue:
             if len(current_mer) == k and current_mer not in vocab: vocab[current_mer] = idx; idx +=1
        return vocab


    def kmer_tokenize(self, sequence):
        sequence = sequence.upper().strip(); tokens = []
        step = 1 if self.overlapping else self.k
        seq_len = len(sequence)
        for i in range(0, seq_len - self.k + 1, step):
            kmer = sequence[i:i+self.k]; tokens.append(self.vocab.get(kmer, self.unk_token_id))
        return tokens

    def encode(self, sequence):
        if not sequence or pd.isna(sequence): # <--- 'pd' 在这里使用
            return torch.full((self.maxlen,), self.pad_token_id, dtype=torch.long)
        token_ids = self.kmer_tokenize(sequence)
        if len(token_ids) > self.maxlen - 2: token_ids = token_ids[:self.maxlen - 2]
        final_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        padding_len = self.maxlen - len(final_ids); final_ids += [self.pad_token_id] * padding_len
        return torch.tensor(final_ids, dtype=torch.long)

# --- 2. DNA 数据增强 ---
def reverse_complement(seq):
    if not isinstance(seq, str): seq = str(seq)
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}; seq = seq.upper().strip()
    return "".join(complement.get(base, base) for base in reversed(seq))

def random_kmer_mask(token_ids, mask_token_id, k, p=0.15):
    masked_ids = token_ids.clone(); indices = torch.where(token_ids > 4)[0]
    if len(indices) == 0: return masked_ids
    num_to_mask = int(len(indices) * p);
    if num_to_mask == 0: return masked_ids
    num_to_mask = min(num_to_mask, len(indices))
    mask_indices = indices[torch.randperm(len(indices))[:num_to_mask]]
    masked_ids[mask_indices] = mask_token_id; return masked_ids


def random_nucleotide_dropout(sequence, p=0.1):
    if not isinstance(sequence, str): sequence = str(sequence)
    seq_list = list(sequence); dropped_seq = [base for base in seq_list if random.random() > p]
    return "".join(dropped_seq)

def get_augmentations(sequence, tokenizer, k_mask=3, p_mask=0.15, p_dropout=0.1):
    if not sequence: raise ValueError("Cannot augment empty sequence")
    aug_type = random.choice(['mask', 'dropout', 'revcomp'])
    if aug_type == 'mask':
        token_ids = tokenizer.encode(sequence)
        augmented_ids = random_kmer_mask(token_ids, tokenizer.mask_token_id, k=k_mask, p=p_mask)
    elif aug_type == 'dropout':
        aug_seq = random_nucleotide_dropout(sequence, p=p_dropout)
        if not aug_seq: aug_seq = sequence
        augmented_ids = tokenizer.encode(aug_seq)
    else: # revcomp
        aug_seq = reverse_complement(sequence)
        augmented_ids = tokenizer.encode(aug_seq)
    return augmented_ids

# --- 3. NT-Xent 损失函数 ---
class NTXentLoss(nn.Module):
    """
    NT-Xent Loss [!! 已修复 !!] 使用动态批次大小。
    """
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = max(temperature, 1e-8)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, actual_batch_size):
        mask = torch.ones((actual_batch_size * 2, actual_batch_size * 2), dtype=torch.bool)
        mask = mask.fill_diagonal_(False)
        for i in range(actual_batch_size):
            mask[i, actual_batch_size + i] = False
            mask[actual_batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        actual_batch_size = z_i.shape[0]
        if actual_batch_size <= 1:
             return torch.tensor(0.0, device=z_i.device, requires_grad=True)
        z_i = F.normalize(z_i, dim=1); z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))
        l_pos = torch.diag(similarity_matrix, actual_batch_size)
        r_pos = torch.diag(similarity_matrix, -actual_batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * actual_batch_size, 1)
        mask = self.mask_correlated_samples(actual_batch_size).to(representations.device)
        if not mask.any():
             negatives = torch.empty(2 * actual_batch_size, 0, device=representations.device)
        else:
             negatives = similarity_matrix[mask].view(2 * actual_batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1);
        logits /= self.temperature
        labels = torch.zeros(2 * actual_batch_size).to(representations.device).long()
        loss = self.criterion(logits, labels);
        loss /= (2 * actual_batch_size)
        return loss