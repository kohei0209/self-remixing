import my_torch_utils as utils
import torch
import torch.nn as nn

from .conformer import ConformerSeparator
from .tfgridnetv2 import TFGridNetV2


def build_model(model_name, conf):
    models = globals()
    model = models[model_name](**conf)

    return model


class Separator(nn.Module):
    r"""Wrapper class for separation models.

    Args:
        config: Dict
            Dictionary containing the configuration.
    """

    def __init__(self, config):
        super().__init__()
        self.model = build_model(config["separator"], config["separator_conf"])
        if config["frontend"] == "stft":
            self.enc = utils.STFTEncoder(**config["frontend_conf"])
            self.dec = utils.STFTDecoder(**config["frontend_conf"])
        else:
            self.enc = self.dec = None
        self.model_name = config["separator"]

    def forward(self, mix):
        n_batch, n_samples = mix.shape[0], mix.shape[-1]

        # stft
        if self.enc is not None:
            ilens = torch.tile(torch.LongTensor([n_samples]), (n_batch,))
            X = self.enc(mix, ilens)[0]
            n_frames, n_freqs = X.shape[-2:]
        else:
            X = mix

        # separation
        Y = self.model(X)

        # istft
        if self.dec is not None:
            y = self.dec(Y.view(-1, n_frames, n_freqs), ilens)[0]
            y = torch.nn.functional.pad(y, (0, n_samples - y.shape[-1]))
            y = y.view(n_batch, -1, n_samples)
        else:
            y = Y
            y = torch.nn.functional.pad(y, (0, n_samples - y.shape[-1]))

        return y
