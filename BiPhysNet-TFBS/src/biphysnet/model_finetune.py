"""Phase 2 adaptive-gating TFBS fine-tuning model."""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, MatthewsCorrCoef, MetricCollection, Recall, Specificity
from transformers import AutoConfig, AutoModel

from biphysnet.losses import FocalLoss, bernoulli_kl
from biphysnet.modules import AttentionPooling, make_se_layer

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath])
except AttributeError:
    pass


class ScorpioFinetuneModel(pl.LightningModule):
    """Adaptive sequence-structure gating model for TFBS prediction."""

    def __init__(self, args):
        super().__init__()
        if not isinstance(args, argparse.Namespace):
            args = argparse.Namespace(**args)
        self.save_hyperparameters(args)

        config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model_name, config=config)
        self.phase1_attention_pooling = AttentionPooling(self.encoder.config.hidden_size)

        dim = self.encoder.config.hidden_size
        feat_dim = getattr(self.hparams, "extra_feature_dim", 42)
        self.use_md_ep = self.hparams.md_ep_train_file is not None

        if self.use_md_ep:
            self.feature_embedding = nn.Linear(1, dim)
            self.feature_cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.feature_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, 8, dim * 4, 0.1, "gelu", batch_first=True),
                2,
            )

        self.load_pretrained_modules(self.hparams.phase1_checkpoint)

        self.seq_se = make_se_layer(dim)
        self.feat_se = make_se_layer(feat_dim) if self.use_md_ep else None
        self.fused_se = make_se_layer(dim, reduction=8)

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 8, dim * 4, 0.1, "gelu", batch_first=True),
            2,
        )
        self.gating_network = nn.Sequential(nn.Linear(dim * 2, dim // 4), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim // 4, 1))
        self.classification_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim // 2, 1))

        if self.hparams.use_focal:
            self.criterion = FocalLoss(alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(getattr(self.hparams, "pos_weight", 1.0))))
        self.consistency_fn_mse = nn.MSELoss()

        metrics = {
            "AUC": AUROC(task="binary", sync_on_compute=False),
            "AUPR": AveragePrecision(task="binary", sync_on_compute=False),
            "Accuracy": Accuracy(task="binary", sync_on_compute=False),
            "F1": F1Score(task="binary", sync_on_compute=False),
            "MCC": MatthewsCorrCoef(task="binary", sync_on_compute=False),
            "Sens": Recall(task="binary", sync_on_compute=False),
            "Spec": Specificity(task="binary", sync_on_compute=False),
        }
        self.train_metrics = MetricCollection(metrics.copy(), prefix="train_")
        self.val_metrics = MetricCollection(metrics.copy(), prefix="val_")
        self.test_metrics = MetricCollection(metrics.copy(), prefix="test_")
        self._test_store = defaultdict(lambda: {"preds": [], "labels": []})

    def load_pretrained_modules(self, path):
        if path == "FROM_SCRATCH" or path is None:
            return
        print(f">>> Loading Phase 1 Weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded successfully")
        except Exception as exc:
            print(f"⚠️ Error loading weights: {exc}")

    def forward(self, inputs):
        ids, feats = None, None
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 2:
                ids, feats = inputs
            else:
                ids = inputs[0]
        else:
            ids = inputs

        if ids is None:
            raise ValueError("CRITICAL: ids is None!")
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=self.device)

        mask = torch.ne(ids, 0).long()
        seq_out = self.encoder(ids, attention_mask=mask).last_hidden_state
        seq_pool = self.seq_se(self.phase1_attention_pooling(seq_out, mask))
        logits_seq = self.classification_head(seq_pool)

        if self.use_md_ep and feats is not None:
            if not isinstance(feats, torch.Tensor):
                feats = torch.as_tensor(feats, device=self.device)
            feats_cal = feats * self.feat_se(feats)
            f_tokens = self.feature_embedding(feats_cal.unsqueeze(-1))
            cls = self.feature_cls_token.expand(f_tokens.size(0), -1, -1)
            f_out = self.feature_encoder(torch.cat([cls, f_tokens], dim=1))[:, 0, :]
            logits_feat = self.classification_head(f_out)

            fused_in = torch.cat([f_out.unsqueeze(1), seq_out], dim=1)
            f_mask = torch.cat([torch.ones(f_out.size(0), 1, device=ids.device), mask], dim=1)
            feat_prime = self.fusion_encoder(fused_in, src_key_padding_mask=(f_mask == 0))[:, 0, :]

            norm_seq = F.normalize(seq_pool, dim=-1)
            norm_phys = F.normalize(f_out, dim=-1)
            sim = (norm_seq * norm_phys).sum(dim=-1, keepdim=True)
            gate_in = torch.cat([seq_pool, f_out], dim=1)
            raw_a = self.gating_network(gate_in)
            alpha = torch.sigmoid(0.7 * raw_a + 1.5 * sim)

            ctx = (1.0 - alpha) * seq_pool + alpha * feat_prime
            logits_fused = self.classification_head(ctx * self.fused_se(ctx))
            return logits_fused.squeeze(1), logits_seq.squeeze(1), logits_feat.squeeze(1)

        return logits_seq.squeeze(1), logits_seq.squeeze(1), None

    def _step(self, batch, mode):
        if not batch:
            return None
        try:
            if isinstance(batch[0], (tuple, list)):
                (ids, feats), labels, cells = batch
            else:
                ids, labels, cells = batch
                feats = None
        except Exception:
            ids, labels, cells = batch[0], batch[1], batch[2]
            feats = None

        if mode == "train" and feats is not None:
            if self.hparams.feature_noise_std > 0:
                feats += torch.randn_like(feats) * self.hparams.feature_noise_std
            if self.hparams.feature_dropout > 0:
                feats *= (torch.rand_like(feats) > self.hparams.feature_dropout).float()

        l_fused, l_seq, l_feat = self((ids, feats))
        loss = self.criterion(l_fused, labels.float())

        if l_feat is not None and self.hparams.consistency_weight > 0:
            ps, pf = torch.sigmoid(l_seq), torch.sigmoid(l_feat)
            loss_c = 0.5 * bernoulli_kl(ps, pf).mean() + 0.5 * self.consistency_fn_mse(ps, pf)
            loss += self.hparams.consistency_weight * loss_c

        self.log(f"{mode}_loss", loss, on_step=(mode == "train"), on_epoch=True, prog_bar=True, sync_dist=True)
        probs = torch.sigmoid(l_fused)
        getattr(self, f"{mode}_metrics").update(probs, labels.int())

        if mode == "test":
            p, l = probs.detach().cpu().numpy(), labels.detach().cpu().numpy()
            for c, pv, lv in zip(cells, p, l):
                self._test_store[c]["preds"].append(float(pv))
                self._test_store[c]["labels"].append(int(lv))
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        self._step(batch, "test")

    def on_train_epoch_end(self):
        self._log_m(self.train_metrics)

    def on_validation_epoch_end(self):
        self._log_m(self.val_metrics)

    def _log_m(self, metrics):
        try:
            self.log_dict(metrics.compute(), prog_bar=True, sync_dist=True)
        except Exception:
            pass
        metrics.reset()

    def on_test_epoch_end(self):
        print("\n" + "=" * 40)
        print("=== Final Test Results (BiPhysNet Logic) ===")
        print("=" * 40)
        res = self.test_metrics.compute()
        for k, v in res.items():
            print(f"{k:<15}: {v:.4f}")
        print("=" * 40 + "\n")
        self.log_dict(res)
        self._test_store.clear()

    def configure_optimizers(self):
        params = list(self.named_parameters())
        grouped = [
            {"params": [p for n, p in params if "encoder." in n or "feature_" in n], "lr": self.hparams.learning_rate},
            {"params": [p for n, p in params if not ("encoder." in n or "feature_" in n)], "lr": self.hparams.learning_rate * self.hparams.head_lr_mult},
        ]
        opt = AdamW(grouped, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingWarmRestarts(opt, T_0=self.hparams.lr_t0)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]
