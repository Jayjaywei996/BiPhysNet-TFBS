"""Command-line entry point for Phase 2 BiPhysNet fine-tuning."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from biphysnet.data_finetune import ClassificationDataModule
from biphysnet.model_finetune import ScorpioFinetuneModel
from biphysnet.tokenizer import KmerTokenizer

warnings.filterwarnings("ignore")

os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_NVML_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BiPhysNet Phase 2: TFBS fine-tuning")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/phase2")
    parser.add_argument("--exp_name", default="finetune")
    parser.add_argument("--phase1_checkpoint", required=True)
    parser.add_argument("--pretrained_model_name", default="armheb/DNA_bert_6")
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--valid_csv", default="valid.csv")
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--md_ep_train_file", default=None)
    parser.add_argument("--md_ep_valid_file", default=None)
    parser.add_argument("--md_ep_test_file", default=None)
    parser.add_argument("--extra_feature_dim", type=int, default=42)
    parser.add_argument("--freeze_layers", type=int, default=0, help="Reserved for compatibility with older scripts")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--head_lr_mult", type=float, default=3.0)
    parser.add_argument("--lr_t0", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    parser.add_argument("--feature_noise_std", type=float, default=0.02)
    parser.add_argument("--feature_dropout", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--precision", default="16")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--kmer_len", type=int, default=6)
    return parser


def main():
    args = build_parser().parse_args()
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    tokenizer = KmerTokenizer(kmerlen=args.kmer_len, maxlen=args.max_len)
    dm = ClassificationDataModule(args, tokenizer)
    model = ScorpioFinetuneModel(args)

    ckpt = ModelCheckpoint(
        dirpath=Path(args.output_dir) / args.exp_name,
        filename="best-{epoch:02d}-{val_AUC:.4f}",
        monitor="val_AUC",
        mode="max",
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices if args.devices != "-1" else -1,
        strategy="auto",
        max_epochs=args.num_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt, EarlyStopping(monitor="val_AUC", patience=args.early_stopping_patience, mode="max"), LearningRateMonitor()],
    )

    if not args.do_test:
        trainer.fit(model, dm)
        dm.setup("test")
        trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt.best_model_path)
    else:
        dm.setup("test")
        trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=args.phase1_checkpoint)


if __name__ == "__main__":
    main()
