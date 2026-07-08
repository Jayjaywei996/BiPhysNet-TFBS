"""Phase 1 cross-modal pretraining model for BiPhysNet-TFBS."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_cosine_schedule_with_warmup
import pytorch_lightning as pl

from biphysnet.losses import SafeNTXentLoss
from biphysnet.modules import AttentionPooling


class CrossModalPretrainModel(pl.LightningModule):
    """MLM + contrastive alignment between sequence and 42D feature branches."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args if isinstance(args, dict) else vars(args))

        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model_name)
        h_dim = self.encoder.config.hidden_size

        self.attention_pooling = AttentionPooling(h_dim)
        self.seq_projection_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, self.hparams.projection_dim))

        self.feature_embedding = nn.Linear(1, h_dim)
        self.feature_cls_token = nn.Parameter(torch.randn(1, 1, h_dim))
        self.feature_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h_dim, 8, h_dim * 4, 0.1, "gelu", batch_first=True),
            2,
        )
        self.feature_projection_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, self.hparams.projection_dim))

        self.mlm_transform = nn.Sequential(nn.Linear(h_dim, h_dim), nn.GELU(), nn.LayerNorm(h_dim))
        self.mlm_head = nn.Linear(h_dim, self.encoder.config.vocab_size)

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

    def _shared_step(self, batch):
        seq_itc, feat, mlm_in, mlm_lbl = batch
        z_seq = self.encode_sequence(seq_itc)
        z_feat = self.encode_feature(feat)
        loss_itc = self.itc_criterion(z_seq, z_feat)

        mask = (mlm_in != 0).int()
        out = self.encoder(mlm_in, attention_mask=mask).last_hidden_state
        logits = self.mlm_head(self.mlm_transform(out))
        loss_mlm = self.mlm_criterion(logits.view(-1, self.encoder.config.vocab_size), mlm_lbl.view(-1))
        total = loss_itc + (self.hparams.mlm_lambda * loss_mlm)
        return total, loss_itc, loss_mlm

    def training_step(self, batch, batch_idx):
        del batch_idx
        if not batch:
            return None
        total, loss_itc, loss_mlm = self._shared_step(batch)
        self.log("train_loss", total, prog_bar=True)
        self.log("train_itc", loss_itc, prog_bar=False)
        self.log("train_mlm", loss_mlm, prog_bar=False)
        return total

    def validation_step(self, batch, batch_idx):
        del batch_idx
        if not batch:
            return None
        total, _, _ = self._shared_step(batch)
        self.log("val_loss", total, prog_bar=True, sync_dist=True)
        return total

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        sch = get_cosine_schedule_with_warmup(opt, self.hparams.warmup_steps, 100000)
        return [opt], [{"scheduler": sch, "interval": "step"}]
