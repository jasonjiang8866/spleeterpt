"""Compare a converted model against the original checkpoint's TF1 meta graph.

Requires optional TensorFlow, the official .meta/.index/.data files, and a converted
.pt checkpoint. Uses full 512x1024 spectrograms. Does not download or train anything.
"""

import argparse
import json

import numpy as np
import tensorflow as tf
import torch

from spleeterpt.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    config = model.config
    rng = np.random.default_rng(901)
    magnitude = rng.uniform(0, 20, (1, config.segment_frames, config.frequency_bins, 2)).astype(
        "float32"
    )
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(args.tf_checkpoint + ".meta")
    feature = graph.get_operation_by_name("conv2d/Conv2D").inputs[0]
    outputs = [graph.get_tensor_by_name(f"{name}_spectrogram/mul:0") for name in config.instruments]
    session_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=2, inter_op_parallelism_threads=2
    )
    with tf.compat.v1.Session(graph=graph, config=session_config) as session:
        saver.restore(session, args.tf_checkpoint)
        expected = np.stack(session.run(outputs, feed_dict={feature: magnitude}), axis=1)
    with torch.inference_mode():
        actual = (
            model(torch.from_numpy(magnitude.transpose(0, 3, 1, 2)))
            .numpy()
            .transpose(0, 1, 3, 4, 2)
        )
    for i, name in enumerate(config.instruments):
        delta = actual[:, i] - expected[:, i]
        print(
            json.dumps(
                {
                    "stem": name,
                    "max_absolute_error": float(np.abs(delta).max()),
                    "rmse": float(np.sqrt(np.mean(delta**2))),
                }
            )
        )
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=3e-4)
    print("PASS: converted PyTorch model matches the original TF1 inference graph")


if __name__ == "__main__":
    main()
