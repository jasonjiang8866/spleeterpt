"""Pinned, checksum-verified inference weights; TensorFlow is never imported here."""

import hashlib
import json
import os
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse


def catalog():
    return json.loads(files("spleeterpt").joinpath("pretrained.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_download(url, destination, *, sha256, size):
    """Bounded HTTPS download, validated cache hits and atomic publication.

    Interrupted/corrupt responses never replace an existing file. Concurrent
    identical requests use distinct temporary files and publish the same bytes.
    """
    destination = Path(destination)
    if urlparse(url).scheme != "https" or size <= 0:
        raise ValueError("Expected HTTPS and positive asset size")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Invalid SHA-256")
    if destination.is_symlink():
        raise ValueError("Refusing a symlink cache entry")
    if (
        destination.is_file()
        and destination.stat().st_size == size
        and digest(destination) == sha256
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            request = urllib.request.Request(url, headers={"User-Agent": "spleeterpt-pretrained/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Refusing non-HTTPS redirect")
                total, checksum = 0, hashlib.sha256()
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Download exceeds expected asset size")
                    checksum.update(block)
                    stream.write(block)
            if total != size or checksum.hexdigest() != sha256:
                raise ValueError("Downloaded asset size/SHA-256 mismatch")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def download_model(name, cache_dir=None):
    metadata = catalog()
    if name not in metadata["models"]:
        raise ValueError(f"Unknown released model: {name}")
    item = metadata["models"][name]
    base = (
        Path(cache_dir)
        if cache_dir
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spleeterpt"
    )
    destination = base / metadata["release"] / item["filename"]
    path = verified_download(item["url"], destination, sha256=item["sha256"], size=item["bytes"])
    notice = files("spleeterpt").joinpath("pretrained-notice.txt").read_text()
    (destination.parent / "Deezer-Spleeter-MIT.txt").write_text(notice)
    return path
