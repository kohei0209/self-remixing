import random

import torch


def get_start_and_end(mix_len, tgt_len, offsets, org_len, margen=4000):
    """
    Function to obtain start and end time where two sources are included.
    This function assumes the existance of only two sources.

    Parameters
    ----------
    mix_len: int
        Length of mixture
    tgt_len: int
        Desired length
    offsets: list[int]
        List of two offset values
    org_len: list[int]
        Original length of two sources
    """

    if mix_len - tgt_len > 0:
        l = max(offsets)
        r = l + min(org_len)

        if random.random() > 0.5:
            start = random.randint(l + margen, r - margen)
            end = start + tgt_len
            if end > mix_len:
                start -= end - mix_len
                end = mix_len
        else:
            end = random.randint(l + margen, r - margen)
            start = end - tgt_len
            if start < 0:
                start = 0
                end = tgt_len
    else:
        start = 0
        end = tgt_len

    return start, end


def zeropad_sources(source, target_length, span):
    """
    source: shape: (..., n_chan, n_samples) or (..., n_samples)
        The source to be zero-padded
    target_length: int
        The desired audio length
    span: int
        Length to span zeros used in padding
    """

    org_shape = source.shape[:-1]
    n_samples = source.shape[-1]

    zeros = torch.zeros(
        (org_shape + (target_length - n_samples,)),
        dtype=source.dtype,
        device=source.device,
    )
    source = torch.cat((zeros[..., :span], source, zeros[..., span:]), dim=-1)
    return source
