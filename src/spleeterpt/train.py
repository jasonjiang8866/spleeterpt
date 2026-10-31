"""Single-device training, validation, AMP, and resumable checkpoints."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .audio import stft
from .checkpoint import FORMAT_VERSION, atomic_save, read_checkpoint
from .config import ModelConfig
from .data import StemDataset, manifest_fingerprint
from .model import Spleeter


def spectral_loss(model, mixture, targets, config, *, precision, device):
    # STFT stays float32; only convolutions use mixed precision.
    mix = stft(mixture.float(), config).abs()[..., : config.frequency_bins]
    target = stft(targets.float(), config).abs()[..., : config.frequency_bins]
    with torch.autocast(
        device_type=device.type,
        enabled=precision != "fp32",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    ):
        predicted = model(mix)
    # Upstream sums per-instrument means, rather than averaging across instruments.
    return F.l1_loss(predicted.float(), target) * len(config.instruments)


def run_training(args):
    for name in ("epochs", "batch_size", "crops_per_track", "validation_crops", "save_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("invalid workers, learning rate, or gradient clipping")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.resume and args.init:
        raise ValueError("choose --resume or --init, not both")
    config = ModelConfig.load(args.config)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("initial training implementation supports cpu or cuda")
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("AMP training requires CUDA; use --precision fp32 on CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; check the installed PyTorch wheel and GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use fp16 or fp32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    train = StemDataset(
        args.train_manifest,
        config,
        training=True,
        samples_per_track=args.crops_per_track,
        seed=args.seed,
    )
    validation = StemDataset(
        args.val_manifest,
        config,
        training=False,
        samples_per_track=args.validation_crops,
        seed=args.seed,
    )
    if set(train.tracks) & set(validation.tracks):
        raise ValueError("training and validation manifests overlap")
    signature = {
        "train": manifest_fingerprint(train.tracks),
        "validation": manifest_fingerprint(validation.tracks),
        "batch_size": args.batch_size,
        "crops_per_track": args.crops_per_track,
        "validation_crops": args.validation_crops,
        "seed": args.seed,
        "precision": args.precision,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "deterministic": args.deterministic,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("output directory is not empty; use --resume or a new run directory")
    model = Spleeter(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    epoch, next_batch, step, best = 0, 0, 0, math.inf
    if args.init:
        initial = read_checkpoint(args.init)
        if ModelConfig(**initial["config"]) != config:
            raise ValueError("initial checkpoint configuration differs")
        model.load_state_dict(initial["model"], strict=True)
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        if ModelConfig(**checkpoint["config"]) != config or checkpoint["signature"] != signature:
            raise ValueError(
                "resume configuration/data differs; use --init for a new fine-tuning run"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch, next_batch, step, best = (
            checkpoint[k] for k in ("epoch", "next_batch", "step", "best")
        )
        torch.set_rng_state(checkpoint["rng_cpu"])
        if device.type == "cuda":
            if len(checkpoint["rng_cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count changed; use --init for a new run")
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])

    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    def save(name, resume_epoch, resume_batch):
        atomic_save(
            {
                "format_version": FORMAT_VERSION,
                "config": config.to_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "signature": signature,
                "epoch": resume_epoch,
                "next_batch": resume_batch,
                "step": step,
                "best": best,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            output / name,
        )

        if name == "last.pt" and getattr(args, "backup_dir", None):
            from .backup import publish

            publish(output, args.backup_dir, step)

    def log(record):
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    pin = device.type == "cuda"
    # Separate generators keep DataLoader bookkeeping from changing dropout's RNG.
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.max_steps is not None and step >= args.max_steps:
        return {"step": step, "best_validation_loss": best}
    for current_epoch in range(epoch, args.epochs):
        train.epoch = current_epoch
        loader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin,
            generator=torch.Generator().manual_seed(args.seed + current_epoch),
        )
        model.train()
        for batch_index, (mixture, targets) in enumerate(loader):
            if current_epoch == epoch and batch_index < next_batch:
                continue
            mixture, targets = mixture.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = spectral_loss(
                model, mixture, targets, config, precision=args.precision, device=device
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # GradScaler handles fp16 overflow by skipping the update; fp32/bf16 fail fast.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=args.precision != "fp16"
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            log(
                {
                    "kind": "train",
                    "epoch": current_epoch,
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else None,
                    "update_skipped": scaler.get_scale() < old_scale,
                }
            )
            if step % args.save_every == 0:
                save("last.pt", current_epoch, batch_index + 1)
            if args.max_steps is not None and step >= args.max_steps:
                save("last.pt", current_epoch, batch_index + 1)
                return {"step": step, "best_validation_loss": best}
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for mixture, targets in validation_loader:
                loss = spectral_loss(
                    model,
                    mixture.to(device),
                    targets.to(device),
                    config,
                    precision=args.precision,
                    device=device,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite validation loss")
                total += loss.item() * mixture.shape[0]
                count += mixture.shape[0]
        value = total / count
        log({"kind": "validation", "epoch": current_epoch, "step": step, "loss": value})
        if value < best:
            best = value
            save("best.pt", current_epoch + 1, 0)
        save("last.pt", current_epoch + 1, 0)
        next_batch = 0
    return {"step": step, "best_validation_loss": best}
