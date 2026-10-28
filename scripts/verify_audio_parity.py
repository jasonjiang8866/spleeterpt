"""Compare complete TF1 and PyTorch waveform pipelines with identical decoded audio.

Reference SNR measures agreement with TensorFlow, NOT source-separation SDR against
ground-truth stems. Requires official v1.4.0 .meta checkpoints and optional TF.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model


def agreement(reference, candidate):
    """No gain fitting or time alignment; report length mismatch separately."""
    length = min(reference.shape[-1], candidate.shape[-1])
    ref = np.asarray(reference[..., :length], dtype=np.float64)
    got = np.asarray(candidate[..., :length], dtype=np.float64)
    error = got - ref
    signal_power = float(np.mean(ref**2))
    error_power = float(np.mean(error**2))
    return {
        "reference_samples": reference.shape[-1],
        "candidate_samples": candidate.shape[-1],
        "max_absolute_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(error_power)),
        "relative_l2_error": float(np.sqrt(error_power / signal_power)) if signal_power else None,
        "reference_snr_db": float(10 * np.log10(signal_power / error_power))
        if signal_power and error_power
        else None,
    }


class OfficialGraph:
    """Current upstream audio methods + restored original pretrained network graph.

    The v1.4.0 .meta audio frontend predates upstream's leading n_fft padding fix.
    Use actual upstream EstimatorSpecBuilder methods, not that obsolete frontend.
    Only replace the builder's model_outputs with the restored pretrained graph.
    """

    def __init__(self, prefix, instruments, upstream_root):
        package = types.ModuleType("spleeter")
        package.__path__ = [str(upstream_root.resolve() / "spleeter")]
        sys.modules["spleeter"] = package
        module = importlib.import_module("spleeter.model")
        params_path = upstream_root / "spleeter" / "resources" / f"{len(instruments)}stems.json"
        params = json.loads(params_path.read_text())
        if params["instrument_list"] != list(instruments):
            raise ValueError("upstream instrument order differs from checkpoint")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.waveform = tf.compat.v1.placeholder(tf.float32, [None, 2], name="input_audio")
            builder = module.EstimatorSpecBuilder({"waveform": self.waveform}, params)
            magnitude = builder.spectrogram_feature
            # Restore the serialized network, bypassing only its obsolete frontend.
            saver = tf.compat.v1.train.import_meta_graph(
                str(prefix) + ".meta",
                import_scope="pretrained",
                input_map={"strided_slice_3:0": magnitude},
            )
            builder._model_outputs = {
                f"{name}_spectrogram": self.graph.get_tensor_by_name(
                    f"pretrained/{name}_spectrogram/mul:0"
                )
                for name in instruments
            }
            self.outputs = builder.outputs
        self.session = tf.compat.v1.Session(
            graph=self.graph,
            config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=2, inter_op_parallelism_threads=2
            ),
        )
        with self.graph.as_default():
            saver.restore(self.session, str(prefix))

    def separate(self, waveform):
        outputs = self.session.run(self.outputs, feed_dict={self.waveform: waveform.numpy().T})
        return {name: value.T for name, value in outputs.items()}

    def close(self):
        self.session.close()


def audio_cases(audio, config, include_edges):
    cases = {"music_original": audio}
    if include_edges:
        # Same music repeated, not a second independent recording. Crosses multiple
        # model windows and deliberately ends off the STFT hop grid.
        length = config.hop_length * config.segment_frames * 2 + 137
        repeats = (length + audio.shape[-1] - 1) // audio.shape[-1]
        cases["music_multiwindow"] = audio.repeat(1, repeats)[:, :length]
        cases["music_short"] = audio[:, :2000]
        cases["silence"] = torch.zeros(2, 4097)
        impulse = torch.zeros(2, 8201)
        impulse[:, [0, 4096, 8200]] = torch.tensor([[0.2], [-0.15]])
        cases["boundary_impulses"] = impulse
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tf-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-edges", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    config = model.config
    audio = read_audio(args.input, config.sample_rate)
    oracle = OfficialGraph(args.tf_checkpoint, config.instruments, args.upstream_root)
    report = {
        "input": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "sample_rate": config.sample_rate,
        "instruments": list(config.instruments),
        "reference": "upstream EstimatorSpecBuilder audio pipeline + original pretrained network graph",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "metric_note": "Reference is TF output, not isolated ground truth; this is not separation SDR.",
        "environment": {"torch": torch.__version__, "tensorflow": tf.__version__, "device": "cpu"},
        "cases": {},
    }
    passed = True
    try:
        for label, waveform in audio_cases(audio, config, args.include_edges).items():
            expected = oracle.separate(waveform)
            actual = separate(model, waveform)
            result = {}
            for name in config.instruments:
                ref, got = expected[name], actual[name].numpy()
                metrics = agreement(ref, got)
                valid = ref.shape == got.shape and np.allclose(got, ref, atol=1e-5, rtol=1e-4)
                metrics["allclose_atol_1e-5_rtol_1e-4"] = bool(valid)
                passed &= valid
                result[name] = metrics
            report["cases"][label] = result
            print(json.dumps({"case": label, "results": result}), flush=True)
    finally:
        oracle.close()
    report["passed"] = bool(passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("FAIL: waveform parity; inspect report")
    print("PASS: complete waveform pipeline, exact length and numerical agreement")


if __name__ == "__main__":
    main()
