import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roformer_kim import checked_download, separate


class Scale(torch.nn.Module):
    def forward(self, x):
        return x * 0.375


@pytest.mark.parametrize("length", [1, 49, 50, 51, 99, 100, 101, 199, 211])
def test_overlap_add_keeps_every_boundary_sample_and_residual(length):
    wave = torch.randn(2, length, generator=torch.Generator().manual_seed(17))
    result = separate(Scale(), wave, chunk_size=100)
    assert set(result) == {"vocals", "accompaniment"}
    torch.testing.assert_close(result["vocals"], wave * 0.375, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(result["vocals"] + result["accompaniment"], wave)


def test_nonfinite_prediction_is_not_silently_replaced():
    class Broken(torch.nn.Module):
        def forward(self, x):
            return x * float("nan")

    with pytest.raises(ValueError, match="finiteness"):
        separate(Broken(), torch.ones(2, 101), chunk_size=100)


def test_short_prediction_is_rejected_instead_of_truncating_reference():
    class Short(torch.nn.Module):
        def forward(self, x):
            return x[..., :-1]

    with pytest.raises(ValueError, match="length"):
        separate(Short(), torch.ones(2, 101), chunk_size=100)


def test_download_rejects_corruption_and_does_not_publish_partial(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "download"
    source.write_bytes(b"incorrect")
    expected = hashlib.sha256(b"correct").hexdigest()
    with pytest.raises(ValueError, match="byte count"):
        checked_download(source.as_uri(), target, len(b"correct"), expected)
    assert not target.exists() and not target.with_suffix(".part").exists()
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached asset"):
        checked_download(source.as_uri(), target, len(b"correct"), expected)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker measurement")
def test_rss_is_process_specific_after_exec():
    code = """
import json, resource
from roformer_kim import peak_rss_bytes
print(json.dumps({'hwm': peak_rss_bytes(), 'rusage': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))
"""
    launcher = """
import subprocess,sys
memory = bytearray(128 * 1024**2)
subprocess.run([sys.executable, '-c', sys.argv[1]], check=True)
"""
    import json
    import os

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "scripts"))
    result = json.loads(subprocess.check_output([sys.executable, "-c", launcher, code], env=env))
    assert result["hwm"] < 64 * 1024**2
    assert result["rusage"] >= 128 * 1024**2
