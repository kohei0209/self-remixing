from typing import Callable, Dict, Optional, Tuple

import my_torch_utils as utils
import torch

from .pit_wrapper import PITLossWrapper


class SelfRemixingLoss(torch.nn.Module):
    r"""A class to compute Self-Remixing's loss [1].
    Can also compute RemixIT's loss (unofficial inplementation) [2].

    As shown in [3], RemixIT and Self-Remixing work without any pre-training.

    Final loss value is computed as
    remixit_loss_weight*RemixITLoss + selfremixing_loss_weight*SelfRemixingLoss

    Args:
        solver_separator: torch.nn.Module
            Solver's separator (student in RemixIT, f_S in [1]).
            Receiving time-domain mixtures as inputs,
            the separator should return time-domain separated signals.
        shuffler_separator: torch.nn.Module
            Shuffler's separator (teacher in RemixIT, f_T in [1])
        loss_func: Callable
            A loss function to compute loss.
            Expected input shape is (n_batch, n_src, n_samples).
        remixit_loss_weight: Optional, float
            The weight of RemixIT's loss.
        selfremixing_loss_weight: Optional, float
            The weight of SelfRemixing's loss.
        constrained_batch_shuffle: Optional, bool
            Whether to appky constrained batch shuffling.
        channel_shuffle: Optional, bool
            Whether to apply channel shuffling.
        channel_shuffle_except_last: bool.
            Whether to apply channel shuffling except the last channel.
            Useful when we know the last channel is noise
            (e.g., semi-supervised speech separation).
        nsrc_to_remix: Optional, int
            Number of sources to be remixed.
            If set to None, all separated sources are used for remixing.
            Useful when the model has more outputs than the number of sources
            (e.g., MixIT pre-trained model).
        solver_mixconsis: Optional, bool
            If True, ensure mixture consistency on solver's outputs.
        shuffler_mixconsis: Optional, bool
            If True, ensure mixture consistency on shuffler's outputs.
        normalize: Optional, bool
            If True, normalize the variance of mixtures.
            Note both initial mixtures and pseudo-mixtures are normalized.
        pit_from: Optional, str
            How to compute PIT loss (see PITLossWrapper for details).
            If specified, loss_func is wrapped by PITLossWrapper.
            If loss_func is already in form of PIT, please set pit_from to None.

    References
    [1] K. Saijo and T. Ogawa, "Self-Remixing: Unsupervised Speech Separation
        VIA Separation and Remixing," ICASSP 2023.

    [2] Tzinis, E., Adi, Y., Ithapu, V. K., Xu, B., Smaragdis, P., & Kumar, A.,
        "RemixIT: Continual self-training of speech enhancement models via bootstrapped
        remixing," IEEE Journal of Selected Topics in Signal Processing.

    [3] K. Saijo and T. Ogawa, "Remixing-based Unsupervised Source Separation
        from Scratch," Interspeech 2023.
    """

    def __init__(
        self,
        solver_separator: torch.nn.Module,
        shuffler_separator: torch.nn.Module,
        loss_func: Callable,
        remixit_loss_weight: Optional[float] = 0.0,
        selfremixing_loss_weight: Optional[float] = 1.0,
        constrained_batch_shuffle: Optional[bool] = False,
        channel_shuffle: Optional[bool] = False,
        channel_shuffle_except_last: Optional[bool] = False,
        nsrc_to_remix: Optional[int] = None,
        solver_mixconsis: Optional[bool] = False,
        shuffler_mixconsis: Optional[bool] = True,
        normalize: Optional[bool] = False,
        pit_from: Optional[str] = None,
    ):
        super().__init__()

        assert remixit_loss_weight >= 0
        assert selfremixing_loss_weight >= 0

        self.remix = utils.Remixing(
            constrained_batch_shuffle=constrained_batch_shuffle,
            channel_shuffle=channel_shuffle,
            channel_shuffle_except_last=channel_shuffle_except_last,
        )

        self.solver_separator = solver_separator
        self.shuffler_separator = shuffler_separator
        self.nsrc_to_remix = nsrc_to_remix
        self.remixit_loss_weight = remixit_loss_weight
        self.selfremixing_loss_weight = selfremixing_loss_weight
        self.solver_mixconsis = solver_mixconsis
        self.shuffler_mixconsis = shuffler_mixconsis
        self.normalize = normalize

        # apply PIT wrapper if necessary
        if pit_from is None:
            self.loss_func = loss_func
        else:
            self.loss_func = PITLossWrapper(loss_func, pit_from=pit_from)

    def forward(
        self,
        mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Self-Remixing's loss forward

        Args
            mix: torch.Tensor
                The input mixture in time-domain,
                shape (..., n_samples)
        Returns
            y_return: torch.Tensor, shape (..., n_src, n_samples)
                The separated signals from shuffler
            loss: torch.Tensor, (1, )
                Computed loss value
        """

        assert mix.shape[0] % 2 == 0

        # shuffler separation and remixing
        pseudo_mix, pseudo_ref, perms, y_return = self.shuffler_process(mix)
        # solver separation and loss computation
        loss = self.solver_process(mix, pseudo_mix, pseudo_ref, perms)

        return y_return, loss.mean()

    @torch.no_grad()
    def shuffler_process(self, mix):
        # valiance normalization
        if self.normalize:
            mix_mean = mix.mean(dim=-1, keepdim=True)
            mix_std = mix.std(dim=-1, keepdim=True) + 1e-8
            mix = (mix - mix_mean) / mix_std

        pseudo_ref = self.shuffler_separator(mix)

        # y_return is returned from forward
        y_return = pseudo_ref.clone().detach()
        if self.normalize:
            y_return = y_return * (mix_std[..., None, :]).clone().detach()

        # adjust length just in case
        m = min(pseudo_ref.shape[-1], mix.shape[-1])
        pseudo_ref, mix = pseudo_ref[..., :m], mix[..., :m]

        # select sources with highest powers for remixing
        pseudo_ref = utils.source_selection(
            pseudo_ref, nsrc_to_remix=self.nsrc_to_remix
        )

        # hold mixture consistency if specified
        if self.shuffler_mixconsis:
            pseudo_ref = utils.mixture_consistency(pseudo_ref, mix)

        # remix shuffler's outputs
        pseudo_mix, pseudo_ref, perms = self.remix.remix(pseudo_ref)
        return pseudo_mix, pseudo_ref, perms, y_return

    def solver_process(self, org_mix, pseudo_mix, pseudo_ref, perms):
        # valiance normalization
        if self.normalize:
            pseudo_mix_std = pseudo_mix.std(dim=-1, keepdim=True) + 1e-8
            pseudo_mix_mean = pseudo_mix.mean(dim=-1, keepdim=True)
            pseudo_ref_mean = pseudo_ref.mean(dim=-1, keepdim=True)
            pseudo_mix = (pseudo_mix - pseudo_mix_mean) / pseudo_mix_std
            pseudo_ref = (pseudo_ref - pseudo_ref_mean) / pseudo_mix_std[..., None]

        # solver's separation
        y = self.solver_separator(pseudo_mix)

        # select same number of most energetic sources as pseudo_references
        if y.shape[-2] > pseudo_ref.shape[-2]:
            y = utils.most_energetic(y, n_src=pseudo_ref.shape[-2])

        assert y.shape == pseudo_ref.shape, (y.shape, pseudo_ref.shape)

        # ensure consistency b/w solver's outputs and pseudo mixture
        if self.solver_mixconsis:
            y = utils.mixture_consistency(y, pseudo_mix)

        # compute RemixIT loss and permute sources back to the original order
        remixit_loss, y = self.loss_func(
            y, pseudo_ref, return_est=True, return_mean=False
        )
        loss = self.remixit_loss_weight * remixit_loss

        # compute Self-Remixng's loss
        if self.selfremixing_loss_weight > 0:
            if self.normalize:
                y = y * pseudo_mix_std[..., None] + pseudo_ref_mean

            # get mixture estimation
            mix_est, y = self.remix.reconstruct_mix(y, perms)

            # adjust length just in case
            m = min(mix_est.shape[-1], org_mix.shape[-1])
            mix_est, org_mix = mix_est[..., :m], org_mix[..., :m]

            # compute Self-Remixing's loss
            selfremixing_loss = self.loss_func(
                mix_est[..., None, :], org_mix[..., None, :], return_mean=False
            )
            loss += self.selfremixing_loss_weight * selfremixing_loss

        return loss


class SupervisedSelfRemixingLoss(SelfRemixingLoss):
    r"""A class to compute supervised Self-Remixing's loss [1, 2].
    Supervised Self-Remixing's loss is the upper bound of
    Self-Remixing's performance.

    Args:
        separator: torch.nn.Module
            Separation model. Receiving time-domain mixtures as inputs,
            the separator have to return time-domain separated signals.
        loss_func: Callable
            A loss function to compute loss.
            Expected input shape is (n_batch, n_src, n_samples).
        remixit_loss_weight: Optional, float
            The weight of RemixIT's loss.
        selfremixing_loss_weight: Optional, float
            The weight of SelfRemixing's loss.
        constrained_batch_shuffle: Optional, bool
            Whether to appky constrained batch shuffling.
        channel_shuffle: Optional, bool
            Whether to apply channel shuffling.
        channel_shuffle_except_last: bool.
            Whether to apply channel shuffling except the last channel.
            Useful when we know the last channel is noise
            (e.g., semi-supervised speech separation).
        mixconsis: Optional, bool
            If True, ensure mixture consistency on outputs.
        normalize: Optional, bool
            If True, normalize the variance of mixtures.
            Note both initial mixtures and pseudo-mixtures are normalized.
        pit_from: Optional, str
            How to compute PIT loss (see PITLossWrapper for details).
            If specified, loss_func is wrapped by PITLossWrapper.
            If loss_func is already in form of PIT, please set pit_from to None.

    References
    [1] K. Saijo and T. Ogawa, "Self-Remixing: Unsupervised Speech Separation
        VIA Separation and Remixing," ICASSP 2023.

    [2] K. Saijo and T. Ogawa, "Remixing-based Unsupervised Source Separation
        from Scratch," Interspeech 2023.
    """

    def __init__(
        self,
        separator: torch.nn.Module,
        loss_func: Callable,
        remixit_loss_weight: Optional[float] = 0.0,
        selfremixing_loss_weight: Optional[float] = 1.0,
        constrained_batch_shuffle: Optional[bool] = True,
        channel_shuffle: Optional[bool] = True,
        channel_shuffle_except_last: Optional[bool] = False,
        mixconsis: Optional[bool] = False,
        normalize: Optional[bool] = False,
        pit_from: Optional[str] = None,
    ):
        assert remixit_loss_weight >= 0
        assert selfremixing_loss_weight >= 0

        super().__init__(
            solver_separator=separator,
            shuffler_separator=None,
            loss_func=loss_func,
            remixit_loss_weight=remixit_loss_weight,
            selfremixing_loss_weight=selfremixing_loss_weight,
            constrained_batch_shuffle=constrained_batch_shuffle,
            channel_shuffle=channel_shuffle,
            channel_shuffle_except_last=channel_shuffle_except_last,
            nsrc_to_remix=None,
            solver_mixconsis=mixconsis,
            shuffler_mixconsis=None,
            normalize=normalize,
            pit_from=pit_from,
        )

    def forward(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # concat noise with reference
        if noise is not None:
            ref = torch.cat((ref, noise), dim=-2)

        if self.normalize:
            mix_mean = mix.mean(dim=-1, keepdim=True)
            mix_std = mix.std(dim=-1, keepdim=True) + 1e-8
            mix = (mix - mix_mean) / mix_std
            ref = (ref - ref.mean(dim=-1, keepdim=True)) / mix_std[..., None, :]

        # remix reference signals
        pseudo_mix, pseudo_ref, perms = self.remix.remix(ref)

        # solver separation and loss computation
        loss = self.solver_process(mix, pseudo_mix, pseudo_ref, perms)

        return None, loss.mean()


class SupervisedSelfRemixingLossWithZeroReferences(torch.nn.Module):
    r"""
    A class to compute supervised Self-Remixing's loss [1, 2].
    Unlike SupervisedSelfRemixingLoss, this class can handle zero references.
    When pseudo-mixture becomes zero signal, that pseudo-mixture is discarded.

    Args:
        separator: torch.nn.Module
            Separation model. Receiving time-domain mixtures as inputs,
            the separator have to return time-domain separated signals.
        remixit_loss_func: Callable
            A loss function to compute RemixIT loss.
            Expected input shape is (n_batch, n_src, n_samples).
        selfremixing_loss_func: Callable
            A loss function to compute Self-Remixing loss.
            Expected input shape is (n_batch, 1, n_samples).
            If not given, remixit_loss_func is used.
        remixit_loss_weight: Optional, float
            The weight of RemixIT's loss.
        selfremixing_loss_weight: Optional, float
            The weight of SelfRemixing's loss.
        constrained_batch_shuffle: Optional, bool
            Whether to appky constrained batch shuffling.
        channel_shuffle: Optional, bool
            Whether to apply channel shuffling.
        channel_shuffle_except_last: bool.
            Whether to apply channel shuffling except the last channel.
            Useful when we know the last channel is noise
            (e.g., semi-supervised speech separation).
        mixconsis: Optional, bool
            If True, ensure mixture consistency on outputs.
        normalize: Optional, bool
            If True, normalize the variance of mixtures.
            Note both initial mixtures and pseudo-mixtures are normalized.
        pit_from: Optional, str
            How to compute PIT loss (see PITLossWrapper for details).
            If specified, loss_func is wrapped by PITLossWrapper.
            If loss_func is already in form of PIT, please set pit_from to None.

    References
    [1] K. Saijo and T. Ogawa, "Self-Remixing: Unsupervised Speech Separation
        VIA Separation and Remixing," ICASSP 2023.

    [2] K. Saijo and T. Ogawa, "Remixing-based Unsupervised Source Separation
        from Scratch," Interspeech 2023.
    """

    def __init__(
        self,
        separator: torch.nn.Module,
        remixit_loss_func: Callable,
        selfremixing_loss_func: Optional[Callable] = None,
        remixit_loss_weight: Optional[float] = 0.0,
        selfremixing_loss_weight: Optional[float] = 1.0,
        constrained_batch_shuffle: Optional[bool] = True,
        channel_shuffle: Optional[bool] = True,
        channel_shuffle_except_last: Optional[bool] = False,
        mixconsis: Optional[bool] = False,
        normalize: Optional[bool] = False,
        pit_from: Optional[str] = None,
    ):
        super().__init__()

        assert remixit_loss_weight >= 0
        assert selfremixing_loss_weight >= 0

        self.remix = utils.Remixing(
            constrained_batch_shuffle=constrained_batch_shuffle,
            channel_shuffle=channel_shuffle,
            channel_shuffle_except_last=channel_shuffle_except_last,
        )

        self.separator = separator
        self.constrained_batch_shuffle = constrained_batch_shuffle
        self.channel_shuffle = channel_shuffle
        self.channel_shuffle_except_last = channel_shuffle_except_last
        self.remixit_loss_weight = remixit_loss_weight
        self.selfremixing_loss_weight = selfremixing_loss_weight
        self.mixconsis = mixconsis
        self.normalize = normalize

        if pit_from is None:
            self.remixit_loss_func = remixit_loss_func
        else:
            self.remixit_loss_func = PITLossWrapper(
                remixit_loss_func, pit_from=pit_from
            )

        if selfremixing_loss_func is None:
            self.selfremixing_loss_func = self.remixit_loss_func
        else:
            self.selfremixing_loss_func = selfremixing_loss_func

    def forward(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pseudo_mix, pseudo_ref, perms, nonzero_indices = self.shuffler_process(
            mix, ref, noise
        )

        loss = self.solver_process(mix, pseudo_mix, pseudo_ref, perms, nonzero_indices)

        return None, loss.mean()

    @torch.no_grad()
    def shuffler_process(self, mix, ref, noise=None):
        if noise is not None:
            ref = torch.cat((ref, noise), dim=-2)

        if self.normalize:
            mix_mean = mix.mean(dim=-1, keepdim=True)
            mix_std = mix.std(dim=-1, keepdim=True) + 1e-8
            mix = (mix - mix_mean) / mix_std
            ref = (ref - ref.mean(dim=-1, keepdim=True)) / mix_std[..., None, :]

        # remixing
        pseudo_mix, pseudo_ref, perms = self.remix.remix(ref)

        # remove zero signals
        sum_amplitude = abs(pseudo_mix).sum(dim=-1)
        nonzero_indices = torch.nonzero(sum_amplitude)[..., 0]
        pseudo_mix, pseudo_ref = (
            pseudo_mix[nonzero_indices],
            pseudo_ref[nonzero_indices],
        )

        return pseudo_mix, pseudo_ref, perms, nonzero_indices

    def solver_process(self, org_mix, pseudo_mix, pseudo_ref, perms, nonzero_indices):
        if self.normalize:
            remix_std = pseudo_mix.std(dim=-1, keepdim=True) + 1e-8
            remix_mean = pseudo_mix.mean(dim=-1, keepdim=True)
            ref_mean = pseudo_ref.mean(dim=-1, keepdim=True)
            pseudo_mix = (pseudo_mix - remix_mean) / remix_std
            pseudo_ref = (pseudo_ref - ref_mean) / remix_std[..., None]

        # separation
        y_tmp = self.separator(pseudo_mix)

        m = min(y_tmp.shape[-1], pseudo_ref.shape[-1])
        y_tmp, pseudo_ref, org_mix, pseudo_mix = (
            y_tmp[..., :m],
            pseudo_ref[..., :m],
            org_mix[..., :m],
            pseudo_mix[..., :m],
        )

        if self.mixconsis:
            y_tmp = utils.mixture_consistency(y_tmp, pseudo_mix)

        remixit_loss, y_tmp = self.remixit_loss_func(
            y_tmp, pseudo_ref, pseudo_mix, return_est=True, return_mean=False
        )

        if self.selfremixing_loss_weight > 0:
            if self.normalize:
                y_tmp = y_tmp * remix_std[..., None] + ref_mean

            y = y_tmp.new_zeros(
                ((org_mix.shape[0],) + y_tmp.shape[-2:]),
            )
            y[nonzero_indices] = y_tmp

            mix_est, y = self.remix.reconstruct_mix(y, perms)
            selfremixing_loss = self.selfremixing_loss_func(
                mix_est[..., None, :], org_mix[..., None, :], return_mean=False
            )

            loss = (
                self.remixit_loss_weight * remixit_loss
                + self.selfremixing_loss_weight * selfremixing_loss
            )
        else:
            loss = remixit_loss

        return loss


class SemiSupervisedSelfRemixingLoss(SelfRemixingLoss):
    r"""
    A class for semi-supervised Self-Remixing training [1].
    Unlike SupervisedSelfRemixingLoss, this class uses both
    supervised and unsupervised data in each training step.
    When pseudo-mixture becomes zero signal, that pseudo-mixture is discarded.

    Args:
        separator: torch.nn.Module
            Separation model. Receiving time-domain mixtures as inputs,
            the separator have to return time-domain separated signals.
        remixit_loss_func: Callable
            A loss function to compute RemixIT loss.
            Expected input shape is (n_batch, n_src, n_samples).
        selfremixing_loss_func: Callable
            A loss function to compute Self-Remixing loss.
            Expected input shape is (n_batch, 1, n_samples).
            If not given, remixit_loss_func is used.
        remixit_loss_weight: Optional, float
            The weight of RemixIT's loss.
        selfremixing_loss_weight: Optional, float
            The weight of SelfRemixing's loss.
        constrained_batch_shuffle: Optional, bool
            Whether to appky constrained batch shuffling.
        channel_shuffle: Optional, bool
            Whether to apply channel shuffling.
        channel_shuffle_except_last: bool.
            Whether to apply channel shuffling except the last channel.
            Useful when we know the last channel is noise
            (e.g., semi-supervised speech separation).
        mixconsis: Optional, bool
            If True, ensure mixture consistency on outputs.
        normalize: Optional, bool
            If True, normalize the variance of mixtures.
            Note both initial mixtures and pseudo-mixtures are normalized.
        pit_from: Optional, str
            How to compute PIT loss (see PITLossWrapper for details).
            If specified, loss_func is wrapped by PITLossWrapper.
            If loss_func is already in form of PIT, please set pit_from to None.

    References
    [1] K. Saijo and T. Ogawa, "Self-Remixing: Unsupervised Speech Separation
        VIA Separation and Remixing," ICASSP 2023.

    [2] K. Saijo and T. Ogawa, "Remixing-based Unsupervised Source Separation
        from Scratch," Interspeech 2023.
    """

    def __init__(
        self,
        solver_separator: torch.nn.Module,
        shuffler_separator: torch.nn.Module,
        loss_func: Callable,
        remixit_loss_weight: Optional[float] = 0.0,
        selfremixing_loss_weight: Optional[float] = 1.0,
        remix_mode: Optional[str] = "random_all",
        nsrc_to_remix: Optional[int] = None,
        loss_thres_params: Optional[Dict] = None,
        solver_mixconsis: Optional[bool] = False,
        shuffler_mixconsis: Optional[bool] = True,
        normalize: Optional[bool] = False,
        pit_from: Optional[str] = None,
        remixing_conf: Optional[Dict] = None,
        supervised_loss_weight: Optional[float] = 0.5,
        noise_loss_weight: Optional[float] = 0.2,
    ):
        super().__init__(
            solver_separator=solver_separator,
            shuffler_separator=shuffler_separator,
            loss_func=loss_func,
            remixit_loss_weight=remixit_loss_weight,
            selfremixing_loss_weight=selfremixing_loss_weight,
            remix_mode=remix_mode,
            nsrc_to_remix=nsrc_to_remix,
            loss_thres_params=loss_thres_params,
            solver_mixconsis=solver_mixconsis,
            shuffler_mixconsis=shuffler_mixconsis,
            normalize=normalize,
            pit_from=pit_from,
            remixing_conf=remixing_conf,
        )

        self.supervised_loss_weight = supervised_loss_weight
        self.noise_loss_weight = noise_loss_weight

    def forward(
        self,
        mix: torch.Tensor,
        mix_synthetic: torch.Tensor,
        ref_synthetic: torch.Tensor,
        noise_synthetic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        mix: torch.Tensor
            The input mixture in time-domain,
            ``shape (..., n_samples)``
        Returns
        ----------
        y_return: torch.Tensor, ``shape (..., n_src, n_samples)``
            The separated signals from SHUFFLER
        loss: torch.Tensor,
            Computed loss value
        """

        n_batch = mix.shape[0]
        assert n_batch % 2 == 0

        # shuffler separation and remixing
        pseudo_mix, pseudo_ref, perms, y_return = self.shuffler_process(mix)

        # solver separation
        loss = self.solver_process(
            mix,
            pseudo_mix,
            pseudo_ref,
            perms,
            mix_synthetic,
            ref_synthetic,
            noise_synthetic,
        )
        return y_return, loss.mean()

    def solver_process(
        self,
        org_mix,
        pseudo_mix,
        pseudo_ref,
        perms,
        mix_synthetic,
        ref_synthetic,
        noise_synthetic,
    ):
        n_batch = pseudo_mix.shape[0]

        if self.normalize:
            pseudo_mix_std = pseudo_mix.std(dim=-1, keepdim=True) + 1e-8
            pseudo_mix_mean = pseudo_mix.mean(dim=-1, keepdim=True)
            pseudo_ref_mean = pseudo_ref.mean(dim=-1, keepdim=True)
            pseudo_mix = (pseudo_mix - pseudo_mix_mean) / pseudo_mix_std
            pseudo_ref = (pseudo_ref - pseudo_ref_mean) / pseudo_mix_std[..., None]

        if mix_synthetic is not None:
            if self.normalize:
                mix_synthetic_std = mix_synthetic.std(dim=-1, keepdim=True) + 1e-8
                mix_synthetic = (
                    mix_synthetic - mix_synthetic.mean(dim=-1, keepdim=True)
                ) / mix_synthetic_std
                ref_synthetic = (
                    ref_synthetic - ref_synthetic.mean(dim=-1, keepdim=True)
                ) / mix_synthetic_std[..., None, :]
                noise_synthetic = (
                    noise_synthetic - noise_synthetic.mean(dim=-1, keepdim=True)
                ) / mix_synthetic_std
            pseudo_mix = torch.cat((pseudo_mix, mix_synthetic), dim=0)

        # solver's separation
        y = self.solver_separator(pseudo_mix)

        if mix_synthetic is not None:
            y_synthetic = y[n_batch:]
            y = y[:n_batch]

        # select same number of most energetic sources as references
        if y.shape[-2] > pseudo_ref.shape[-2]:
            y = utils.most_energetic(y, n_src=pseudo_ref.shape[-2])

        assert y.shape == pseudo_ref.shape

        # ensure consistency b/w solver's outputs and pseudo mixture
        if self.solver_mixconsis:
            y = utils.mixture_consistency(y, pseudo_mix[:n_batch])
            y_synthetic = utils.mixture_consistency(y_synthetic, mix_synthetic)

        # compute RemixIT loss and permute sources back to the original order
        remixit_loss, y = self.loss_func(
            y, pseudo_ref, return_est=True, return_mean=False
        )
        loss = self.remixit_loss_weight * remixit_loss

        # compute Self-Remixng loss
        if self.selfremixing_loss_weight > 0:
            if self.normalize:
                y = y * pseudo_mix_std[..., None] + pseudo_ref_mean

            # get mixture estimation
            mix_est, y = self.remix.reconstruct_mix(y, perms)

            # compute Self-Remixing's loss
            selfremixing_loss = self.loss_func(
                mix_est[..., None, :], org_mix[..., None, :], return_mean=False
            )
            loss = loss + self.selfremixing_loss_weight * selfremixing_loss

        # loss thresholding
        if self.loss_thresholder is not None and self.solver_separator.training:
            loss = self.loss_thresholder(loss)

        # supervised loss
        if mix_synthetic is not None and self.supervised_loss_weight > 0:
            m = min(y_synthetic.shape[-1], ref_synthetic.shape[-1])
            ref_synthetic, y_synthetic = (
                ref_synthetic[..., :m],
                y_synthetic[..., :m],
            )
            supervised_loss = self.loss_func(
                y_synthetic[..., :-1, :], ref_synthetic
            ) + self.noise_loss_weight * self.loss_func(
                y_synthetic[..., [-1], :], noise_synthetic[..., None, :]
            )
            loss = (
                1 - self.supervised_loss_weight
            ) * loss + self.supervised_loss_weight * supervised_loss

        return loss
