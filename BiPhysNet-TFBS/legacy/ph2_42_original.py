#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ph2_42.py (论文逻辑对齐版)
修改内容：
1. 融合逻辑改为 (1-alpha)*seq + alpha*phys，支持“结构否决”。
2. 门控输入改为原始模态特征，相似度计算基于 f_out。
3. 一致性损失去除 .detach()，实现双向梯度校准。
"""
import os
import sys
import argparse
import pathlib
from pathlib import Path
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torchmetrics import MetricCollection, AUROC, Accuracy, F1Score, MatthewsCorrCoef, AveragePrecision, Recall, Specificity
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoModel, AutoConfig

# [Fix PyTorch 2.6]
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath])
except AttributeError:
    pass

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_NVML_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.append(script_dir)

try:
    from ph1_pretrain_42 import CrossModalPretrainModel
except ImportError:
    class CrossModalPretrainModel(nn.Module): pass

try:
    from contrastive_utils import KmerTokenizer
except ImportError:
    print("错误: 找不到 contrastive_utils.py")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ================= 基础组件 =================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, hidden_size)
        self.attention_query = nn.Parameter(torch.randn(hidden_size))
    def forward(self, hidden_states, attention_mask):
        uit = torch.tanh(self.attention_weights(hidden_states))
        scores = torch.matmul(uit, self.attention_query)
        if attention_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, mask_value)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        targets = targets.type_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        at = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = at * (1 - pt).pow(self.gamma) * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def bernoulli_kl(p, q, eps=1e-8):
    p = p.clamp(eps, 1-eps); q = q.clamp(eps, 1-eps)
    return p * torch.log(p/q) + (1-p) * torch.log((1-p)/(1-q))

def make_se_layer(dim, reduction=4):
    mid = max(8, dim // reduction)
    return nn.Sequential(nn.Linear(dim, mid), nn.ReLU(), nn.Linear(mid, dim), nn.Sigmoid())

# ================= 数据处理 (保持原样) =================
class ClassificationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, md_ep_file=None):
        self.tokenizer = tokenizer
        self.md_ep_features = None
        self.cell_types = None
        try:
            print(f"--- Loading: {csv_file}")
            df = pd.read_csv(csv_file, header=0, sep=',', on_bad_lines='warn', engine='python', quoting=3, dtype=str)
            df = df.dropna(subset=['sequence', 'label'])
            self.sequences = df['sequence'].astype(str).tolist()
            self.labels = df['label'].astype(int).tolist()
            self.cell_types = df['cell_type'].astype(str).tolist() if 'cell_type' in df.columns else ['ALL']*len(self.labels)
            
            if md_ep_file and os.path.exists(md_ep_file):
                print(f"--- Loading Features: {md_ep_file}")
                self.md_ep_features = np.load(md_ep_file).astype(np.float32)
                if len(self.md_ep_features) != len(self.sequences):
                    raise ValueError(f"Mismatch: {len(self.md_ep_features)} vs {len(self.sequences)}")
        except Exception as e: print(f"Error: {e}"); raise

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        if idx >= len(self.labels): raise IndexError
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cell = self.cell_types[idx]
        try: ids = torch.as_tensor(self.tokenizer.encode(str(seq)), dtype=torch.long)
        except: return None, None, None
        if self.md_ep_features is not None:
            return (ids, torch.tensor(self.md_ep_features[idx], dtype=torch.float32)), label, cell
        return ids, label, cell

def classification_collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    if not batch: return None
    if isinstance(batch[0][0], tuple):
        ids = torch.nn.utils.rnn.pad_sequence([x[0][0] for x in batch], batch_first=True, padding_value=0)
        feats = torch.stack([x[0][1] for x in batch])
        labels = torch.stack([x[1] for x in batch])
        cells = [x[2] for x in batch]
        return (ids, feats), labels, cells
    else:
        ids = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
        labels = torch.stack([x[1] for x in batch])
        cells = [x[2] for x in batch]
        return ids, labels, cells

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args; self.tokenizer = tokenizer

    def setup(self, stage=None):
        base = Path(self.args.input_dir)
        cargs = {'tokenizer': self.tokenizer}
        t_csv = base / self.args.train_csv
        v_csv = base / self.args.valid_csv
        te_csv = base / self.args.test_csv
        t_npy = base/self.args.md_ep_train_file if self.args.md_ep_train_file else None
        v_npy = base/self.args.md_ep_valid_file if self.args.md_ep_valid_file else None
        te_npy = base/self.args.md_ep_test_file if self.args.md_ep_test_file else None

        if stage == 'fit' or stage is None:
            self.train_dataset = ClassificationDataset(t_csv, md_ep_file=t_npy, **cargs)
            self.val_dataset = ClassificationDataset(v_csv, md_ep_file=v_npy, **cargs)
        if stage == 'test' or stage is None:
            self.test_dataset = ClassificationDataset(te_csv, md_ep_file=te_npy, **cargs)

        if hasattr(self, 'train_dataset'):
            lbls = np.array(self.train_dataset.labels)
            pos = lbls.sum(); neg = len(lbls) - pos
            setattr(self.args, 'pos_weight', float(neg)/pos if pos>0 else 1.0)

    def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=classification_collate_fn, pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=classification_collate_fn, pin_memory=True)
    def test_dataloader(self): return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=classification_collate_fn, pin_memory=True)

# ================= 核心模型 (修改逻辑点) =================
class ScorpioFinetuneModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if not isinstance(args, argparse.Namespace): args = argparse.Namespace(**args)
        self.save_hyperparameters(args)

        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model_name, config=config)
        self.phase1_attention_pooling = AttentionPooling(self.encoder.config.hidden_size)
        
        dim = self.encoder.config.hidden_size
        feat_dim = getattr(self.hparams, 'extra_feature_dim', 42)
        self.use_md_ep = (self.hparams.md_ep_train_file is not None)

        if self.use_md_ep:
            self.feature_embedding = nn.Linear(1, dim)
            self.feature_cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.feature_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, dim*4, 0.1, 'gelu', batch_first=True), 2)

        self.load_pretrained_modules(self.hparams.phase1_checkpoint)

        self.seq_se = make_se_layer(dim)
        self.feat_se = make_se_layer(feat_dim) if self.use_md_ep else None
        self.fused_se = make_se_layer(dim, reduction=8)
        
        self.fusion_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, dim*4, 0.1, 'gelu', batch_first=True), 2)
        # 门控输入维度为 dim*2 (seq + phys)
        self.gating_network = nn.Sequential(nn.Linear(dim*2, dim//4), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim//4, 1))
        
        self.classification_head = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim//2, 1))

        if self.hparams.use_focal:
            self.criterion = FocalLoss(alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(getattr(self.hparams, 'pos_weight', 1.0))))
        self.consistency_fn_mse = nn.MSELoss()

        metrics = {
            'AUC': AUROC(task="binary", sync_on_compute=False), 
            'AUPR': AveragePrecision(task="binary", sync_on_compute=False), 
            'Accuracy': Accuracy(task="binary", sync_on_compute=False), 
            'F1': F1Score(task="binary", sync_on_compute=False), 
            'MCC': MatthewsCorrCoef(task="binary", sync_on_compute=False),
            'Sens': Recall(task="binary", sync_on_compute=False), 
            'Spec': Specificity(task="binary", sync_on_compute=False)
        }
        self.train_metrics = MetricCollection(metrics.copy(), prefix='train_')
        self.val_metrics = MetricCollection(metrics.copy(), prefix='val_')
        self.test_metrics = MetricCollection(metrics.copy(), prefix='test_')
        self._test_store = defaultdict(lambda: {"preds": [], "labels": []})

    def load_pretrained_modules(self, path):
        if path == "FROM_SCRATCH" or path is None: return
        print(f">>> Loading Phase 1 Weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded successfully")
        except Exception as e: print(f"⚠️ Error loading weights: {e}")

    def forward(self, inputs):
        ids, feats = None, None
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 2: ids, feats = inputs
            else: ids = inputs[0]
        else: ids = inputs

        if ids is None: raise ValueError("CRITICAL: ids is None!")
        if not isinstance(ids, torch.Tensor): ids = torch.as_tensor(ids, device=self.device)
        
        mask = torch.ne(ids, 0).long()
        seq_out = self.encoder(ids, attention_mask=mask).last_hidden_state
        seq_pool = self.seq_se(self.phase1_attention_pooling(seq_out, mask))
        logits_seq = self.classification_head(seq_pool)

        if self.use_md_ep and feats is not None:
            if not isinstance(feats, torch.Tensor): feats = torch.as_tensor(feats, device=self.device)
            # 1. 物理分支特征提取 (h_phys)
            feats_cal = feats * self.feat_se(feats)
            f_tokens = self.feature_embedding(feats_cal.unsqueeze(-1))
            cls = self.feature_cls_token.expand(f_tokens.size(0), -1, -1)
            f_out = self.feature_encoder(torch.cat([cls, f_tokens], dim=1))[:, 0, :]
            logits_feat = self.classification_head(f_out)

            # 2. 跨模态交互 (交互后的物理特征)
            fused_in = torch.cat([f_out.unsqueeze(1), seq_out], dim=1)
            f_mask = torch.cat([torch.ones(f_out.size(0), 1, device=ids.device), mask], dim=1)
            feat_prime = self.fusion_encoder(fused_in, src_key_padding_mask=(f_mask==0))[:, 0, :]

            # 3. 自适应门控计算 (对齐论文公式)
            # 基于原始模态特征计算相似度先验
            norm_seq = F.normalize(seq_pool, dim=-1)
            norm_phys = F.normalize(f_out, dim=-1)
            sim = (norm_seq * norm_phys).sum(dim=-1, keepdim=True) # 余弦相似度 s
            
            # 门控输入：原始特征拼接
            gate_in = torch.cat([seq_pool, f_out], dim=1)
            raw_a = self.gating_network(gate_in)
            alpha = torch.sigmoid(0.7 * raw_a + 1.5 * sim) # alpha = sigma(0.7*MLP + 1.5*s)
            
            # 4. 门控加权平均 (实现结构否决)
            # h_final = (1-alpha)*h_seq + alpha*h_phys_prime
            ctx = (1.0 - alpha) * seq_pool + alpha * feat_prime
            
            logits_fused = self.classification_head(ctx * self.fused_se(ctx))
            return logits_fused.squeeze(1), logits_seq.squeeze(1), logits_feat.squeeze(1)
        
        return logits_seq.squeeze(1), logits_seq.squeeze(1), None

    def _step(self, batch, mode):
        if not batch: return None
        try:
            if isinstance(batch[0], (tuple, list)): (ids, feats), labels, cells = batch
            else: ids, labels, cells = batch; feats = None
        except: ids, labels, cells = batch[0], batch[1], batch[2]; feats = None

        if mode == 'train' and feats is not None:
            if self.hparams.feature_noise_std > 0: feats += torch.randn_like(feats)*self.hparams.feature_noise_std
            if self.hparams.feature_dropout > 0: feats *= (torch.rand_like(feats)>self.hparams.feature_dropout).float()

        l_fused, l_seq, l_feat = self((ids, feats))
        loss = self.criterion(l_fused, labels.float())
        
        if l_feat is not None and self.hparams.consistency_weight > 0:
            # [关键修改] 去掉 .detach()，实现真正的双塔相互校准 (Mutual Calibration)
            ps, pf = torch.sigmoid(l_seq), torch.sigmoid(l_feat)
            loss_c = 0.5 * bernoulli_kl(ps, pf).mean() + 0.5 * self.consistency_fn_mse(ps, pf)
            loss += self.hparams.consistency_weight * loss_c

        self.log(f'{mode}_loss', loss, on_step=(mode=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        probs = torch.sigmoid(l_fused)
        getattr(self, f"{mode}_metrics").update(probs, labels.int())
        
        if mode == 'test':
            p, l = probs.detach().cpu().numpy(), labels.detach().cpu().numpy()
            for c, pv, lv in zip(cells, p, l):
                self._test_store[c]['preds'].append(float(pv))
                self._test_store[c]['labels'].append(int(lv))
        return loss

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): self._step(b, 'val')
    def test_step(self, b, i): self._step(b, 'test')
    
    def on_train_epoch_end(self): self._log_m(self.train_metrics)
    def on_validation_epoch_end(self): self._log_m(self.val_metrics)
    def _log_m(self, m):
        try: self.log_dict(m.compute(), prog_bar=True, sync_dist=True)
        except: pass
        m.reset()

    def on_test_epoch_end(self):
        print("\n" + "="*40)
        print("=== Final Test Results (BiPhysNet Logic) ===")
        print("="*40)
        res = self.test_metrics.compute()
        for k, v in res.items(): print(f"{k:<15}: {v:.4f}")
        print("="*40 + "\n")
        self.log_dict(res)
        self._test_store.clear()

    def configure_optimizers(self):
        params = list(self.named_parameters())
        grouped = [
            {'params': [p for n, p in params if 'encoder.' in n or 'feature_' in n], 'lr': self.hparams.learning_rate},
            {'params': [p for n, p in params if not ('encoder.' in n or 'feature_' in n)], 'lr': self.hparams.learning_rate * self.hparams.head_lr_mult}
        ]
        opt = AdamW(grouped, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingWarmRestarts(opt, T_0=self.hparams.lr_t0)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]

# ================= Main (保持原样) =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default="output")
    parser.add_argument('--exp_name', default="finetune")
    parser.add_argument('--phase1_checkpoint', required=True)
    parser.add_argument('--pretrained_model_name', default="armheb/DNA_bert_6")
    parser.add_argument('--train_csv', default="train.csv")
    parser.add_argument('--valid_csv', default="valid.csv")
    parser.add_argument('--test_csv', default="test.csv")
    parser.add_argument('--md_ep_train_file', default=None)
    parser.add_argument('--md_ep_valid_file', default=None)
    parser.add_argument('--md_ep_test_file', default=None)
    parser.add_argument('--extra_feature_dim', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--head_lr_mult', type=float, default=3.0)
    parser.add_argument('--lr_t0', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--consistency_weight', type=float, default=0.2)
    parser.add_argument('--feature_noise_std', type=float, default=0.02)
    parser.add_argument('--feature_dropout', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default='0')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--kmer_len', type=int, default=6)
    
    args = parser.parse_args()
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    
    tokenizer = KmerTokenizer(kmerlen=args.kmer_len, maxlen=args.max_len)
    dm = ClassificationDataModule(args, tokenizer)
    model = ScorpioFinetuneModel(args)
    
    ckpt = ModelCheckpoint(dirpath=Path(args.output_dir)/args.exp_name, filename='best-{epoch:02d}-{val_AUC:.4f}', monitor='val_AUC', mode='max', save_last=True)
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices if args.devices!='-1' else -1, 
        strategy="auto", max_epochs=args.num_epochs, precision=16, 
        accumulate_grad_batches=args.accumulate_grad_batches, 
        callbacks=[ckpt, EarlyStopping(monitor='val_AUC', patience=args.early_stopping_patience, mode='max'), LearningRateMonitor()]
    )
    
    if not args.do_test:
        trainer.fit(model, dm)
        dm.setup('test')
        trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt.best_model_path)
    else:
        dm.setup('test')
        trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=args.phase1_checkpoint)

if __name__ == '__main__':
    main()