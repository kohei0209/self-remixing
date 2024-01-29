import re
from pathlib import Path

import torch


def search_epochs_with_best_criteria(result, num_epochs, criteria, max_epoch=None):
    if criteria == "loss" or criteria == "wer":
        coef = 1
    elif "sisdr" in criteria:
        coef = -1
    else:
        print("Multiply -1 to specified criterion, OK??")
        coef = -1

    search_range = len(result) if max_epoch is None else min(max_epoch, len(result))
    losses = {}
    for i in range(search_range - 1, -1, -1):
        losses[result[i]["epoch"]] = coef * result[i][criteria]

    l = sorted(losses.items(), key=lambda x: x[1])
    losses.clear()
    losses.update(l)

    epochs = list(losses.keys())[:num_epochs]
    return epochs


def search_epochs_with_best_criteria2(model_dir, result, num_epochs, criteria):
    if criteria == "loss" or criteria == "wer":
        coef = 1
    elif "sisdr" in criteria:
        coef = -1
    else:
        print("Multiply -1 to specified criterion, OK??")
        coef = -1

    epochs = []
    p_tmp = Path(model_dir).glob("epoch*")
    for p in p_tmp:
        epoch = int(re.sub(f"\D", "", p.name))
        epochs.append(epoch)

    losses = {}
    for epoch in epochs:
        losses[epoch] = coef * result[epoch - 1][criteria]

    losses_tmp = sorted(losses.items(), key=lambda x: x[1])
    losses.clear()
    losses.update(losses_tmp)

    epochs = list(losses.keys())[:num_epochs]
    return epochs


def grad_norm(module: torch.nn.Module):
    """
    Helper function that computes the gradient norm
    """
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def most_energetic(X, n_src, src_dim=1):
    """
    This function selects most energetic sources from input.

    Parameters
    ----------
    X: torch.Tensor, shape(..., n_src_current, n_time)
        Input time-domain signal with ```n_src_current``` sources
    n_src: int,
        Number of sources we want to select

    Returns
    ----------
    X: torch.Tensor, shape(..., n_src, n_time)
        Selected ```n_src``` sources
    """
    assert src_dim == 1, "We assume src_dim is 1"
    # input tensor dimension
    ndim = X.ndim
    # compute power
    power = X.abs().square().mean(dim=tuple(range(2, ndim)), keepdim=True)
    # get topk index
    _, idx = torch.topk(power, n_src, dim=src_dim)
    # sort the index
    idx, _ = torch.sort(idx, dim=src_dim, descending=False)
    # make idx have same shape as X
    _, idx = torch.broadcast_tensors(X[:, :n_src], idx)
    # get most energetic tensors
    X = torch.gather(X, src_dim, idx)
    return X


def mixture_consistency(
    y: torch.Tensor,
    org_mix: torch.Tensor,
    est_mix: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if est_mix is None:
        est_mix = torch.sum(y, dim=1)
    y = y + ((org_mix - est_mix) / y.shape[1])[:, None]

    return y
