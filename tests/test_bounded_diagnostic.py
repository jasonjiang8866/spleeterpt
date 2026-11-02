import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

from spleeterpt import audio, streaming
from spleeterpt.checkpoint import FORMAT_VERSION, atomic_save
from spleeterpt.model import Spleeter

spec = importlib.util.spec_from_file_location(
    "diagnose_bounded", Path(__file__).resolve().parents[1] / "scripts/diagnose_bounded.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_cached_and_fp64_controls_restore_frontends(tiny_config, tmp_path):
    checkpoint = tmp_path / "model.pt"
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "config": tiny_config.to_dict(),
            "model": Spleeter(tiny_config).state_dict(),
        },
        checkpoint,
    )
    source = tmp_path / "audio.wav"
    sf.write(source, np.random.default_rng(42).normal(0, 0.1, (8000, 2)), 8000, subtype="FLOAT")
    original = audio.stft
    report = diagnostic.diagnose(checkpoint, source, "cpu", seconds=1, fp64_control=True)
    assert audio.stft is original and streaming.stft is original
    assert report["duration_seconds"] == 1
    assert len(report["stft_block_max_complex_errors"]) > 1
    for key in ("normal_bounded", "cached_full_stft_bounded"):
        assert all(m["max_abs_error"] < 2e-6 for m in report[key].values())
    assert all(
        m["max_abs_error"] < 2e-6 for m in report["fp64_fft_control"]["bounded_vs_full"].values()
    )
