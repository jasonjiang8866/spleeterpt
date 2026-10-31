"""Committed checkpoint generations on a separately mounted persistent filesystem."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

FILES = {"last.pt", "best.pt", "metrics.jsonl", "config.json", "arguments.json"}


def digest(path):
    with Path(path).open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def sync_dir(path):
    if not hasattr(os, "O_DIRECTORY"):
        return  # Directory fsync is available on the Linux training platform.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def synced_copy(source, target):
    with Path(source).open("rb") as src, Path(target).open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def publish(output, root, step):
    """Publish only after all files reach storage; retain the last three commits."""
    output, root = Path(output), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{step:09d}-{uuid.uuid4().hex}"
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        files = {}
        for filename in sorted(FILES):
            source = output / filename
            if source.is_file():
                synced_copy(source, pending / filename)
                files[filename] = {
                    "bytes": source.stat().st_size,
                    "sha256": digest(pending / filename),
                }
        if not {"last.pt", "metrics.jsonl"} <= files.keys():
            raise ValueError("Checkpoint and metrics are required")
        manifest = {"format": 1, "step": step, "generation": name, "files": files}
        with (pending / "manifest.json").open("x") as f:
            json.dump(manifest, f)
            f.flush()
            os.fsync(f.fileno())
        sync_dir(pending)
        os.rename(pending, root / name)
        sync_dir(root)
        pointer = root / (".latest-" + uuid.uuid4().hex)
        with pointer.open("x") as f:
            json.dump({"generation": name}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pointer, root / "latest.json")
        sync_dir(root)
        # Only prune generations created by this module, after the new pointer is durable.
        generations = sorted(
            (p for p in root.glob("step-*") if re.fullmatch(r"step-\d{9}-[a-f0-9]{32}", p.name)),
            key=lambda p: p.stat().st_mtime_ns,
        )
        for old in generations[:-3]:
            if old.name != name:
                shutil.rmtree(old)
        print(f"Durable checkpoint committed: step {step}", flush=True)
        return manifest
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def validate(root):
    root = Path(root)
    name = json.loads((root / "latest.json").read_text())["generation"]
    if not isinstance(name, str) or not re.fullmatch(r"step-\d{9}-[a-f0-9]{32}", name):
        raise ValueError("Invalid backup generation")
    folder = root / name
    if folder.is_symlink():
        raise ValueError("Backup generation must not be a link")
    manifest = json.loads((folder / "manifest.json").read_text())
    files = manifest["files"]
    if (
        manifest.get("format") != 1
        or manifest.get("generation") != name
        or not {"last.pt", "metrics.jsonl"} <= files.keys() <= FILES
    ):
        raise ValueError("Invalid backup inventory")
    for n, info in files.items():
        p = folder / n
        if (
            p.is_symlink()
            or not p.is_file()
            or p.stat().st_size != info["bytes"]
            or digest(p) != info["sha256"]
        ):
            raise ValueError("Backup checksum mismatch")
    return folder, manifest


def restore(root, output):
    """Restore the committed metrics prefix too; never append a lost attempt's tail."""
    folder, manifest = validate(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Restore requires a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".restore-", dir=output.parent))
    try:
        for n in manifest["files"]:
            synced_copy(folder / n, pending / n)
        sync_dir(pending)
        os.rename(pending, output)
        sync_dir(output.parent)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return output / "last.pt", manifest
