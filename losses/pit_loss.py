from typing import Callable, Optional, Tuple

import my_torch_utils as utils
import torch

from .pit_wrapper import PITLossWrapper


class PITLoss(torch.nn.Module):
    def __init__(
        self,
        separator: torch.nn.Module,
        loss_func: Callable,
        pit_from: Optional[str] = None,
        ensure_mixture_consistency: Optional[bool] = False,
        noise_loss_ratio: Optional[float] = 0.2,
    ):
        super().__init__()

        self.separator = separator
        self.ensure_mixture_consistency = ensure_mixture_consistency
        self.noise_loss_ratio = noise_loss_ratio

        if pit_from is None:
            # when given ```loss_func``` is already PIT loss
            self.pit_loss = loss_func
        else:
            self.pit_loss = PITLossWrapper(loss_func, pit_from=pit_from)

    def forward(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y, *_ = self.separator(mix)

        m = min(y.shape[-1], ref.shape[-1])
        y, ref, mix = y[..., :m], ref[..., :m], mix[..., :m]

        y_return = y.clone().detach()

        if self.ensure_mixture_consistency:
            y = utils.mixture_consistency(y, mix)

        if noise is None:
            loss = self.pit_loss(y, ref)
        else:
            loss = self.pit_loss(
                y[..., :-1, :], ref
            ) + self.noise_loss_ratio * self.pit_loss(y[..., [-1], :], noise[..., :m])

        return y_return, loss.mean()


class PITLossWithZeroReference(torch.nn.Module):
    def __init__(
        self,
        separator: torch.nn.Module,
        loss_func: Callable,
        ensure_mixture_consistency: Optional[bool] = False,
        pit_from: Optional[str] = None,
        standardization: Optional[bool] = False,
    ):
        super().__init__()

        self.separator = separator
        self.ensure_mixture_consistency = ensure_mixture_consistency
        self.loss_func = loss_func
        self.standardization = standardization
        if self.standardization:
            print("\n\nStandardization in PIT Loss Wrapper\n\n")

        # print("Use the loss cited from Mr.Tzinis's repository")
        # self.loss_func = utils.PermInvariantSNRwithZeroRefs(
        #    n_sources=4,
        #    zero_mean=False,
        #    backward_loss=True,
        #    inactivity_threshold=-40.,
        # )

    def forward(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        n_refs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.standardization:
            std = mix.std(dim=-1, keepdim=True) + 1e-8
            mean = mix.mean(dim=-1, keepdim=True)
            network_input = (mix - mean) / std
            ref = ref - ref.mean(dim=-1, keepdim=True)
        else:
            network_input = mix

        y, *_ = self.separator(network_input)

        # unscale the output
        if self.standardization:
            y = y * std[..., None, :]

        y_return = y.clone().detach()

        m = min(y.shape[-1], ref.shape[-1])
        y, ref, mix = y[..., :m], ref[..., :m], mix[..., :m]

        if self.ensure_mixture_consistency:
            y = utils.mixture_consistency(y, mix)

        loss = self.loss_func(y, ref, mix)
        # loss, perm = self.loss_func(y, ref, return_best_permutation=True)

        return y_return, loss
