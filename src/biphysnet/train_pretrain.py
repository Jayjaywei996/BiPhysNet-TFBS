"""Command-line entry point for Phase 1 BiPhysNet pretraining."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from biphysnet.data_pretrain import CrossModalDataModule
from biphysnet.model_pretrain import CrossModalPretrainModel
from biphysnet.tokenizer import KmerTokenizer

warnings.filterwarnings("ignore")

# Safe defaults for environments where NCCL/NVML causes issues.
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_NVML_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BiPhysNet Phase 1: MLM + ITC pretraining")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--valid_csv", default="valid.csv")
    parser.add_argument("--train_features_npy", required=True)
    parser.add_argument("--valid_features_npy", required=True)
    parser.add_argument("--output_dir", default="outputs/phase1")
    parser.add_argument("--exp_name", default="pretrain")
    parser.add_argument("--pretrained_model_name", default="MsAlEhR/MetaBERTa-bigbird-gene")
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--feature_dim", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--kmer_len", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--mlm_lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="0")
    return parser


def main():
    args = build_parser().parse_args()
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    print("=== Phase 1 Pretrain: MLM + ITC ===")

    tokenizer = KmerTokenizer(kmerlen=args.kmer_len, maxlen=args.max_len)
    data_module = CrossModalDataModule(args, tokenizer)
    model = CrossModalPretrainModel(args)

    ckpt = ModelCheckpoint(
        dirpath=Path(args.output_dir) / args.exp_name,
        filename="pretrain-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=2,
    )
    strat = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True) if args.devices and "," in str(args.devices) else "auto"

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strat,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
        logger=pl.loggers.TensorBoardLogger(args.output_dir, name=args.exp_name),
        precision=16,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=True,
        log_every_n_steps=50,
    )

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        trainer.fit(model, data_module, ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, data_module)

    print(f"Best: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()
