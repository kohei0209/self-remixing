from typing import Optional, Tuple

import losses
import my_torch_utils as utils
import torch

from .efficient_mixit_wrapper import EfficentMixITWrapper
from .mixit_wrapper import MixITLossWrapper


class MixITLoss(torch.nn.Module):
    r"""Mixture invariant training loss.

    Args:
        separator: torch.nn.Module,
            Separation model.
        loss_func: callable,
            Loss function.
        generalized: bool,
            Whether to use generalized mixit loss.
            See MixITLossWrapper for details.
            Efficient mixit loss always uses generalized loss.
        ensure_mixconsis: bool,
            Whether to ensure mixture consistency.
        normalize: bool,
            Whether to normalize the variance of mixture of mixtuers.
        efficient_mixit: bool,
            Whether to use efficient (least-square) mixit loss.
        sparsity_loss_weight: float,
            Weight for sparsity loss.

    References
        [1] Scott Wisdom et al. "Unsupervised sound separation using
        mixtures of mixtures." arXiv:2006.12701 (2020).
        [2] Scott Wisdom et al. "Sparse, efficient, and semantic mixture invariant
        training: Taming in-the-wild unsupervised sound separation." WASPAA, 2021.
    """

    def __init__(
        self,
        separator,
        loss_func,
        generalized: Optional[bool] = True,
        ensure_mixconsis: Optional[bool] = False,
        normalize: Optional[bool] = False,
        efficient_mixit: Optional[bool] = False,
        sparsity_loss_weight: Optional[float] = 0.0,
    ):
        super().__init__()

        self.separator = separator
        self.ensure_mixconsis = ensure_mixconsis
        self.normalize = normalize
        if self.normalize:
            print("\nstandardize MoM!!!!\n")

        if efficient_mixit:
            self.mixit_loss = EfficentMixITWrapper(loss_func)
        else:
            self.mixit_loss = MixITLossWrapper(
                loss_func,
                generalized=generalized,
            )
        self.sparsity_loss_weight = sparsity_loss_weight

    def forward(
        self,
        mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""MixIT loss forward

        Args:
            mix: torch.Tensor, shape (n_batch, n_samples).
                Mixtures in time-domain.
                n_batch must be even number to make mixture of mixtures.

        Returns:
            y: torch.Tensor, shape (n_batch//2, n_src*2, n_time).
                Separated sources in time-domain.
            loss: torch.Tensor, shape (1, ).
                Mean MixIT loss value.
        """
        # we only consider batch size is even number
        n_batch = mix.shape[0]
        assert n_batch % 2 == 0, mix.shape

        # mom: mixture of mixtures, shape(n_batch//2, n_time)
        mix = torch.stack((mix[: n_batch // 2], mix[n_batch // 2 :]), dim=-2)
        mom = mix.sum(dim=-2)

        if self.normalize:
            std = torch.std(mom, dim=-1, keepdim=True)
            mom /= std
            mix /= std[..., None]

        # separation,  y: shape(n_batch//2, n_src*2, n_time)
        y = self.separator(mom)

        m = min(y.shape[-1], mix.shape[-1])
        y, mom, mix = y[..., :m], mom[..., :m], mix[..., :m]

        if self.ensure_mixconsis:
            y = utils.mixture_consistency(y, mom)
        loss = self.mixit_loss(y, mix)

        if self.sparsity_loss_weight > 0:
            sparsity_loss = losses.sparsity_loss(y)
            loss += self.sparsity_loss_weight * sparsity_loss

        return y, loss
