import numpy as np
import pytest
import torch

from spleeterpt.convert import import_weights, layer_name
from spleeterpt.model import Spleeter


class Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_variable_to_shape_map(self):
        return {name: value.shape for name, value in self.tensors.items()}

    def get_tensor(self, name):
        return self.tensors[name]


def export_test_tensors(model):
    tensors = {}

    def conv(layer, base, index):
        name = layer_name(base, index)
        tensors[f"{name}/kernel"] = layer.weight.detach().numpy().transpose(2, 3, 1, 0).copy()
        tensors[f"{name}/bias"] = layer.bias.detach().numpy().copy()

    def norm(layer, index):
        name = layer_name("batch_normalization", index)
        for target, source in (
            ("weight", "gamma"),
            ("bias", "beta"),
            ("running_mean", "moving_mean"),
            ("running_var", "moving_variance"),
        ):
            tensors[f"{name}/{source}"] = getattr(layer, target).detach().numpy().copy()

    for stem, net in enumerate(model.unets.values()):
        for i, layer in enumerate(net.encoder):
            conv(layer, "conv2d", 7 * stem + i)
        for i, layer in enumerate(net.encoder_norm):
            norm(layer, 12 * stem + i)
        for i, layer in enumerate(net.decoder):
            conv(layer, "conv2d_transpose", 6 * stem + i)
            norm(net.decoder_norm[i], 12 * stem + 6 + i)
        conv(net.output, "conv2d", 7 * stem + 6)
    return tensors


def test_import_preserves_parameters(tiny_config):
    source, restored = Spleeter(tiny_config), Spleeter(tiny_config)
    assert import_weights(Reader(export_test_tensors(source)), restored) == 140
    for name, parameter in source.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("error", ["missing", "shape", "nonfinite", "extra_stem"])
def test_import_rejects_incompatible_checkpoint(tiny_config, error):
    model = Spleeter(tiny_config)
    tensors = export_test_tensors(model)
    if error == "missing":
        del tensors["conv2d/kernel"]
    elif error == "shape":
        tensors["conv2d/kernel"] = np.zeros((1, 1, 1, 1), dtype="float32")
    elif error == "nonfinite":
        tensors["conv2d/kernel"].fill(np.nan)
    else:
        tensors["conv2d_14/kernel"] = np.ones((5, 5, 2, 16), dtype="float32")
    with pytest.raises(ValueError):
        import_weights(Reader(tensors), model)
