#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ph1_pretrain_42.py (最终修复版)
修复内容：
1. 内置 NTXentLoss，修复 'invalid device ordinal' 多卡报错。
2. 保持 DDP (Gloo) 后端，防止 NVML 报错。
"""
import os
import sys

# [!! 环境变量 !!]
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_NVML_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

import argparse
import warnings
import pathlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# 尝试导入 Tokenizer (只导入 Tokenizer，不导入 Loss)
try:
    from contrastive_utils import KmerTokenizer
except ImportError:
    print("错误: 找不到 contrastive_utils.py")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# [!! 新增 !!] 本地定义的稳健 Loss 函数
# 替代 contrastive_utils 里有 bug 的版本
# ==========================================
class SafeNTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SafeNTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (Batch, Dim)
        """
        batch_size = z_i.size(0)
        
        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 计算相似度 (Batch, Batch)
        # sim_ij[k, l] 是 z_i[k] 和 z_j[l] 的相似度
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        
        # 标签：对角线是正样本 (0->0, 1->1, ...)
        # [!! 关键修复 !!] 必须使用 z_i.device 确保 label 在同一张卡上
        labels = torch.arange(batch_size, dtype=torch.long, device=z_i.device)
        
        # 双向计算 CrossEntropy
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2

# ==========================================
# 1. 基础组件
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, hidden_size)
        self.attention_query = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, hidden_states, attention_mask, output_attentions=False): 
        uit = torch.tanh(self.attention_weights(hidden_states))
        scores = torch.matmul(uit, self.attention_query)
        if attention_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, mask_value)
        weights = F.softmax(scores, dim=1)
        context_vector = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        if output_attentions: return context_vector, weights
        else: return context_vector

# ==========================================
# 2. Data Module
# ==========================================
class CrossModalDataset(Dataset):
    def __init__(self, csv_file, feature_file, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        print(f"--- [Dataset] ---")
        print(f"  Seq: {csv_file}")
        print(f"  Feat: {feature_file}")

        try:
            self.data = pd.read_csv(csv_file, header=0, usecols=['sequence'], sep=',', on_bad_lines='warn', engine='python', quoting=3, dtype=str)
            self.data = self.data.dropna(subset=['sequence'])
            self.sequences = self.data['sequence'].astype(str).tolist()
            
            if not os.path.exists(feature_file): raise FileNotFoundError(f"Missing: {feature_file}")
            self.features = np.load(feature_file).astype(np.float32)
            
            if len(self.sequences) != len(self.features):
                raise ValueError(f"Mismatch: Seq={len(self.sequences)}, Feat={len(self.features)}")
        except Exception as e: print(f"Data Error: {e}"); raise

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        if idx >= len(self.sequences): raise IndexError
        seq_str = str(self.sequences[idx])
        feat_vec = torch.tensor(self.features[idx], dtype=torch.float32)

        try: token_ids = torch.tensor(self.tokenizer.encode(seq_str), dtype=torch.long)
        except: return None

        mlm_input_ids = token_ids.clone()
        mlm_labels = torch.full_like(token_ids, -100) 
        valid_indices = torch.where(mlm_input_ids > 4)[0]
        
        if len(valid_indices) > 0:
            num_mask = max(1, int(len(valid_indices) * self.mlm_probability))
            mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_mask]]
            mlm_labels[mask_indices] = mlm_input_ids[mask_indices]
            for idx_mask in mask_indices:
                prob = torch.rand(1).item()
                if prob < 0.8: mlm_input_ids[idx_mask] = self.mask_token_id
                elif prob < 0.9: mlm_input_ids[idx_mask] = torch.randint(5, self.vocab_size, (1,)).item()
        
        return token_ids, feat_vec, mlm_input_ids, mlm_labels

def cross_modal_collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    if not batch: return None
    seqs = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    feats = torch.stack([x[1] for x in batch])
    mlm_in = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)
    mlm_lbl = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=-100)
    return seqs, feats, mlm_in, mlm_lbl

class CrossModalDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args; self.tokenizer = tokenizer

    def setup(self, stage=None):
        t_csv = Path(self.args.input_dir) / self.args.train_csv
        v_csv = Path(self.args.input_dir) / self.args.valid_csv
        self.train_dataset = CrossModalDataset(t_csv, self.args.train_features_npy, self.tokenizer)
        self.val_dataset = CrossModalDataset(v_csv, self.args.valid_features_npy, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, collate_fn=cross_modal_collate_fn, persistent_workers=(self.args.num_workers > 0)) 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, collate_fn=cross_modal_collate_fn, persistent_workers=(self.args.num_workers > 0)) 

# ==========================================
# 3. Model
# ==========================================
class CrossModalPretrainModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict): self.save_hyperparameters(args)
        else: self.save_hyperparameters(args)
        
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model_name)
        h_dim = self.encoder.config.hidden_size
        
        self.attention_pooling = AttentionPooling(h_dim) 
        self.seq_projection_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, self.hparams.projection_dim))

        self.feature_embedding = nn.Linear(1, h_dim)
        self.feature_cls_token = nn.Parameter(torch.randn(1, 1, h_dim))
        self.feature_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(h_dim, 8, h_dim*4, 0.1, 'gelu', batch_first=True), 2)
        self.feature_projection_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, self.hparams.projection_dim))
        
        self.mlm_transform = nn.Sequential(nn.Linear(h_dim, h_dim), nn.GELU(), nn.LayerNorm(h_dim))
        self.mlm_head = nn.Linear(h_dim, self.encoder.config.vocab_size)
        
        # [!! 使用本地定义的 SafeNTXentLoss !!]
        self.itc_criterion = SafeNTXentLoss(temperature=self.hparams.temperature)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100) 

    def encode_sequence(self, ids):
        mask = (ids != 0).int()
        out = self.encoder(ids, attention_mask=mask).last_hidden_state
        ctx = self.attention_pooling(out, mask)
        return self.seq_projection_head(ctx)

    def encode_feature(self, feats):
        toks = self.feature_embedding(feats.unsqueeze(-1))
        cls = self.feature_cls_token.expand(toks.size(0), -1, -1)
        out = self.feature_encoder(torch.cat([cls, toks], dim=1))[:, 0, :]
        return self.feature_projection_head(out)

    def training_step(self, batch, batch_idx):
        if not batch: return None
        seq_itc, feat, mlm_in, mlm_lbl = batch
        
        z_seq = self.encode_sequence(seq_itc)
        z_feat = self.encode_feature(feat)
        loss_itc = self.itc_criterion(z_seq, z_feat)
        
        mask = (mlm_in != 0).int()
        out = self.encoder(mlm_in, attention_mask=mask).last_hidden_state
        logits = self.mlm_head(self.mlm_transform(out))
        loss_mlm = self.mlm_criterion(logits.view(-1, self.encoder.config.vocab_size), mlm_lbl.view(-1))
        
        total = loss_itc + (self.hparams.mlm_lambda * loss_mlm)
        
        self.log('train_loss', total, prog_bar=True)
        self.log('train_itc', loss_itc, prog_bar=False)
        self.log('train_mlm', loss_mlm, prog_bar=False)
        return total

    def validation_step(self, batch, batch_idx):
        if not batch: return
        seq_itc, feat, mlm_in, mlm_lbl = batch
        z_seq = self.encode_sequence(seq_itc)
        z_feat = self.encode_feature(feat)
        loss_itc = self.itc_criterion(z_seq, z_feat)
        mask = (mlm_in != 0).int()
        out = self.encoder(mlm_in, attention_mask=mask).last_hidden_state
        logits = self.mlm_head(self.mlm_transform(out))
        loss_mlm = self.mlm_criterion(logits.view(-1, self.encoder.config.vocab_size), mlm_lbl.view(-1))
        total = loss_itc + (self.hparams.mlm_lambda * loss_mlm)
        self.log('val_loss', total, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        sch = get_cosine_schedule_with_warmup(opt, self.hparams.warmup_steps, 100000)
        return [opt], [{"scheduler": sch, "interval": "step"}]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--train_csv', default="train.csv")
    parser.add_argument('--valid_csv', default="valid.csv")
    parser.add_argument('--train_features_npy', required=True)
    parser.add_argument('--valid_features_npy', required=True)
    parser.add_argument('--output_dir', default="scorpio_pretrain_output")
    parser.add_argument('--exp_name', default="pretrain")
    parser.add_argument('--pretrained_model_name', default="MsAlEhR/MetaBERTa-bigbird-gene")
    parser.add_argument('--resume_checkpoint', default=None)
    parser.add_argument('--feature_dim', type=int, default=42)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--kmer_len', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--mlm_lambda', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default='0')
    
    args = parser.parse_args()
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    print(f"=== Phase 1 Pretrain (Safe Loss + Gloo) ===")
    
    tokenizer = KmerTokenizer(kmerlen=args.kmer_len, maxlen=args.max_len)
    data_module = CrossModalDataModule(args, tokenizer)
    model = CrossModalPretrainModel(args)

    ckpt = ModelCheckpoint(dirpath=Path(args.output_dir)/args.exp_name, filename='pretrain-{epoch:02d}-{val_loss:.3f}', monitor='val_loss', mode='min', save_last=True, save_top_k=2)
    
    strat = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True) if args.devices and ',' in args.devices else 'auto'
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs, accelerator=args.accelerator, devices=args.devices, strategy=strat, 
        callbacks=[ckpt, LearningRateMonitor(logging_interval='step')], 
        logger=pl.loggers.TensorBoardLogger(args.output_dir, name=args.exp_name), 
        precision=16, accumulate_grad_batches=args.accumulate_grad_batches, enable_checkpointing=True, log_every_n_steps=50
    )

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        trainer.fit(model, data_module, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, data_module)

    print(f"Best: {ckpt.best_model_path}")

if __name__ == '__main__':
    main()