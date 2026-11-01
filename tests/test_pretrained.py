"""No-network download integrity, cache behavior, CLI and release metadata checks."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from spleeterpt import pretrained
from spleeterpt.checkpoint import FORMAT_VERSION, atomic_save
from spleeterpt.cli import main, parser
from spleeterpt.model import Spleeter

ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    def geturl(self):
        return "https://release-assets.githubusercontent.com/test"


@pytest.fixture
def asset(tmp_path):
    payload = b"verified weights fixture"
    return payload, {
        "url": "https://github.com/example/release/weights.pt",
        "destination": tmp_path / "model.pt",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def test_download_atomic_cache_and_corruption_repair(asset, monkeypatch):
    payload, args = asset
    calls = []

    def request(*a, **kw):
        calls.append(1)
        return Response(payload)

    monkeypatch.setattr(pretrained.urllib.request, "urlopen", request)
    path = pretrained.verified_download(**args)
    assert path.read_bytes() == payload
    assert pretrained.verified_download(**args) == path
    assert len(calls) == 1
    path.write_bytes(b"x" * len(payload))
    pretrained.verified_download(**args)
    assert path.read_bytes() == payload and len(calls) == 2
    assert sorted(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("response", [b"short", b"x" * 24, b"oversized" * 10])
def test_invalid_download_preserves_existing_file(asset, monkeypatch, response):
    _, args = asset
    path = args["destination"]
    path.write_bytes(b"previous file")
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", lambda *a, **kw: Response(response))
    with pytest.raises(ValueError):
        pretrained.verified_download(**args)
    assert path.read_bytes() == b"previous file"
    assert sorted(path.parent.iterdir()) == [path]


def test_network_failure_cleans_temporary_file(asset, monkeypatch):
    _, args = asset

    def failed(*a, **kw):
        raise TimeoutError("test timeout")

    monkeypatch.setattr(pretrained.urllib.request, "urlopen", failed)
    with pytest.raises(TimeoutError):
        pretrained.verified_download(**args)
    assert not list(args["destination"].parent.iterdir())


def test_refuses_insecure_redirect(asset, monkeypatch):
    payload, args = asset
    response = Response(payload)
    monkeypatch.setattr(response, "geturl", lambda: "http://example.test/weights")
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="non-HTTPS"):
        pretrained.verified_download(**args)
    assert not args["destination"].exists()


def test_model_cache_notice_and_cli(asset, monkeypatch, capsys):
    payload, args = asset
    fixture = {
        "release": "weights-test",
        "models": {
            "4stems": {
                "filename": "4stems.pt",
                "url": args["url"],
                "sha256": args["sha256"],
                "bytes": args["size"],
            }
        },
    }
    monkeypatch.setattr(pretrained, "catalog", lambda: fixture)
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", lambda *a, **kw: Response(payload))
    cache = args["destination"].parent
    main(["download", "--stems", "4", "--cache-dir", str(cache)])
    path = Path(capsys.readouterr().out.strip())
    assert path == cache / "weights-test/4stems.pt"
    assert path.read_bytes() == payload
    assert (path.parent / "Deezer-Spleeter-MIT.txt").read_bytes() == (
        ROOT / "licenses/Deezer-Spleeter-MIT.txt"
    ).read_bytes()
    with pytest.raises(ValueError, match="Unknown"):
        pretrained.download_model("5stems", cache)


def test_cli_model_inference_uses_downloaded_checkpoint(
    audio_dataset, tiny_config, monkeypatch, tmp_path
):
    path = tmp_path / "tiny.pt"
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "config": tiny_config.to_dict(),
            "model": Spleeter(tiny_config).state_dict(),
        },
        path,
    )
    calls = []

    def download(name, cache):
        calls.append((name, cache))
        return path

    monkeypatch.setattr(pretrained, "download_model", download)
    output = tmp_path / "output"
    main(
        [
            "separate",
            str(audio_dataset[3][2] / "mixture.wav"),
            "--model",
            "2stems",
            "--cache-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    assert calls == [("2stems", str(tmp_path))]
    assert all((output / f"{s}.wav").is_file() for s in tiny_config.instruments)
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["separate", "song.wav", "--model", "2stems", "--checkpoint", "x.pt", "--output", "out"]
        )


def test_catalog_pins_original_assets_and_notice():
    catalog = pretrained.catalog()
    source = json.loads((ROOT / "runpod/quality-assets.json").read_text())["assets"]
    assert set(catalog["models"]) == {"2stems", "4stems"}
    assert catalog["release"] == "weights-v1.0.0"
    for name, item in catalog["models"].items():
        assert item["upstream_archive"] == source[name + ".tar.gz"]
        assert item["filename"] == name + ".pt"
        assert item["url"] == (
            "https://github.com/jasonjiang8866/spleeterpt/releases/download/"
            + catalog["release"]
            + "/"
            + name
            + ".pt"
        )
        assert len(item["sha256"]) == 64 and item["bytes"] > 1_000_000
    assert (ROOT / "src/spleeterpt/pretrained-notice.txt").read_bytes() == (
        ROOT / "licenses/Deezer-Spleeter-MIT.txt"
    ).read_bytes()
