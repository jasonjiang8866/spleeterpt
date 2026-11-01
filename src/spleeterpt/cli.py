"""Command-line entry points. All training is explicitly started by the caller."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from .audio import read_audio, separate
from .checkpoint import load_model
from .config import ModelConfig
from .data import make_manifests
from .train import run_training


def parser():
    root = argparse.ArgumentParser(prog="spleeterpt")
    commands = root.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="write an upstream-compatible model preset")
    config.add_argument("--stems", type=int, choices=[2, 4, 5], default=2)
    config.add_argument("--output", required=True)
    manifest = commands.add_parser("manifest", help="split a training pool by track")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--validation-tracks", type=int, required=True)
    manifest.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--val-manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--crops-per-track", type=int, default=16)
    train.add_argument("--validation-crops", type=int, default=4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--save-every", type=int, default=150)
    train.add_argument(
        "--backup-dir", help="publish last three checkpoint generations on persistent storage"
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    train.add_argument("--resume")
    train.add_argument("--init", help="initialize model weights only, with a fresh optimizer")
    train.add_argument(
        "--max-steps", type=int, help="absolute total batch limit, useful for smoke tests"
    )
    train.add_argument("--deterministic", action="store_true")
    train.add_argument(
        "--freeze-bn-stats",
        action="store_true",
        help="use fixed BatchNorm running statistics while training affine parameters and dropout",
    )
    infer = commands.add_parser("separate")
    infer.add_argument("input")
    weights = infer.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint")
    weights.add_argument(
        "--model", choices=["2stems", "4stems"], help="download pinned official conversion"
    )
    infer.add_argument("--cache-dir", help="cache directory for --model")
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    infer.add_argument(
        "--bounded-memory",
        action="store_true",
        help="stream file blocks at the model sample rate into a new output directory",
    )
    download = commands.add_parser("download", help="fetch and verify released inference weights")
    download.add_argument("--stems", type=int, choices=[2, 4], required=True)
    download.add_argument("--cache-dir")
    convert = commands.add_parser(
        "convert", help="convert an official TF1 checkpoint (optional TF dependency)"
    )
    convert.add_argument(
        "--checkpoint", required=True, help="prefix such as weights/2stems/model, without .index"
    )
    convert.add_argument("--stems", type=int, choices=[2, 4, 5], required=True)
    convert.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "config":
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            stream.write(json.dumps(ModelConfig.preset(args.stems).to_dict(), indent=2) + "\n")
    elif args.command == "manifest":
        for path in make_manifests(args.root, args.output, args.validation_tracks, args.seed):
            print(path)
    elif args.command == "train":
        run_training(args)
    elif args.command == "convert":
        from .convert import convert_checkpoint

        count = convert_checkpoint(args.checkpoint, args.stems, args.output)
        print(f"Converted {count} tensors to {args.output}")
    elif args.command == "download":
        from .pretrained import download_model

        print(download_model(f"{args.stems}stems", args.cache_dir))
    elif args.command == "separate":
        checkpoint = args.checkpoint
        if args.model:
            from .pretrained import download_model

            checkpoint = download_model(args.model, args.cache_dir)
        model = load_model(checkpoint, args.device)
        if args.bounded_memory:
            from .streaming import separate_file

            separate_file(model, args.input, args.output)
            return
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if any((output / f"{name}.wav").exists() for name in model.config.instruments):
            raise FileExistsError("output stems already exist; choose another directory")
        audio = read_audio(args.input, model.config.sample_rate)
        for name, stem in separate(model, audio).items():
            sf.write(
                output / f"{name}.wav", stem.numpy().T, model.config.sample_rate, subtype="FLOAT"
            )


if __name__ == "__main__":
    main()
