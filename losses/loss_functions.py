import itertools

import my_torch_utils as utils
import torch
from torch import nn

from .pit_wrapper import PITLossWrapper


class MultiResolutionL1Loss(nn.Module):
    def __init__(
        self,
        stft_params,
        snr_max=30,
        solve_perm=True,
        only_denominator=False,
        threshold_with="reference",
        logarithm=True,
    ):
        super().__init__()

        assert threshold_with in ["mixture", "reference"]

        self.stft = []
        for params in stft_params:
            self.stft.append(utils.STFTEncoder(**params))
        self.temp = 0 if snr_max is None else 10 ** (-snr_max / 10)
        self.only_denominator = only_denominator
        self.threshold_with = threshold_with
        self.logarithm = logarithm

        self.solve_perm = solve_perm
        self.pit_loss = PITLossWrapper(
            self.pairwise_multi_resolution_l1, pit_from="pw_mtx"
        )

    def multi_resolution_l1(self, est, ref, eps=1e-8):
        assert ref.shape == est.shape
        n_batch, n_src, n_samples = ref.shape
        eps = 1e-4 if est.dtype == torch.float16 else 1e-8

        # time-domain SNR loss
        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)
        if self.threshold_with == "mixture":
            threshold = ((ref.sum(dim=-2)) ** 2).sum(dim=-1, keepdim=True)
        else:
            threshold = tgt
        snr_loss = 10 * torch.log10(noise + self.temp * threshold + eps)
        if not self.only_denominator:
            snr_loss -= 10 * torch.log10(tgt + eps)

        # time-domain L1 loss
        l1_loss = abs(ref - est).mean(dim=-1)

        # frequency domain multi-resolution L1 loss
        mr_loss = 0
        for stft in self.stft:
            ilens = torch.tile(torch.LongTensor([n_samples]), (n_batch * n_src,))
            ref_tmp, est_tmp = ref.reshape(-1, n_samples), est.reshape(-1, n_samples)
            REF = abs(stft(ref_tmp, ilens)[0])
            EST = abs(stft(est_tmp, ilens)[0])
            T, F = REF.shape[-2:]
            REF = REF.reshape(n_batch, n_src, T, F)
            EST = EST.reshape(n_batch, n_src, T, F)

            if self.logarithm:
                REF = torch.log(REF + eps)
                EST = torch.log(EST + eps)
            mr_loss = mr_loss + abs(REF - EST).mean(dim=(-1, -2))

        loss = snr_loss + l1_loss + (mr_loss / len(self.stft))

        if loss.ndim == 2:
            loss = loss.mean(dim=-1)
        return loss

    def pairwise_multi_resolution_l1(self, est, ref, eps=1e-8):
        assert ref.shape == est.shape
        n_batch, n_src, n_samples = ref.shape

        # ref: (..., 1, n_src, n_samples)
        ref = ref[..., None, :, :]
        # est: (..., n_src, 1, n_samples)
        est = est[..., None, :]

        # time-domain SNR loss
        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)
        if self.threshold_with == "mixture":
            threshold = ((ref.sum(dim=-2)) ** 2).sum(dim=-1, keepdim=True)
        else:
            threshold = tgt
        snr_loss = 10 * torch.log10(noise + self.temp * threshold + eps)
        if not self.only_denominator:
            snr_loss -= 10 * torch.log10(tgt + eps)

        # time-domain L1 loss
        l1_loss = abs(ref - est).mean(dim=-1)

        # frequency domain multi-resolution L1 loss
        mr_loss = torch.zeros_like(snr_loss)
        for stft in self.stft:
            # apply stft
            ilens = torch.tile(torch.LongTensor([n_samples]), (n_batch * n_src,))
            REF = abs(stft(ref.reshape(-1, n_samples), ilens)[0])
            EST = abs(stft(est.reshape(-1, n_samples), ilens)[0])
            # reshape for computing pair-wise loss
            dim1, dim2 = REF.shape[-2:]
            REF = REF.reshape(n_batch, 1, n_src, dim1, dim2)
            EST = EST.reshape(n_batch, n_src, 1, dim1, dim2)
            # use log magnitude spectrogram if specified
            if self.logarithm:
                REF = torch.log(REF + eps)
                EST = torch.log(EST + eps)
            # magnitude l1 loss
            mr_loss = mr_loss + abs(REF - EST).mean(dim=(-1, -2))

        assert (
            snr_loss.shape == l1_loss.shape == mr_loss.shape == (n_batch, n_src, n_src)
        )
        loss = snr_loss + l1_loss + (mr_loss / len(self.stft))

        return loss

    def forward(self, est, ref, return_est=False, return_mean=True):
        assert est.shape == ref.shape
        if self.solve_perm:
            return self.pit_loss(
                est, ref, return_est=return_est, return_mean=return_mean
            )
        else:
            return self.multi_resolution_l1(est, ref)


class ThresSNRLoss(nn.Module):
    def __init__(
        self,
        snr_max=30,
        solve_perm=True,
        only_denominator=False,
        threshold_with="reference",
    ):
        super().__init__()

        assert threshold_with in ["mixture", "reference"]

        self.snr_max = snr_max
        self.only_denominator = only_denominator
        self.temp = 0 if snr_max is None else 10 ** (-snr_max / 10)
        self.threshold_with = threshold_with
        self.solve_perm = solve_perm

        self.pit_loss = PITLossWrapper(self.pairwise_thresSNR, pit_from="pw_mtx")

    def singlesrc_thresSNR(self, est, ref, eps=1e-8):
        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)
        if self.threshold_with == "mixture":
            threshold = ((ref.sum(dim=-2)) ** 2).sum(dim=-1)
        else:
            threshold = tgt

        neg_snr = 10 * torch.log10(noise + self.temp * threshold + eps)
        if not self.only_denominator:
            neg_snr -= 10 * torch.log10(tgt + eps)

        # mean source dimension
        if neg_snr.ndim == 2:
            neg_snr = neg_snr.mean(dim=-1)
        return neg_snr

    def pairwise_thresSNR(self, est, ref, eps=1e-8):
        # ref: (..., 1, n_src, n_samples)
        ref = ref[..., None, :, :]
        # est: (..., n_src, 1, n_samples)
        est = est[..., None, :]

        noise = ((ref - est) ** 2).sum(dim=-1)
        tgt = (ref**2).sum(dim=-1)
        if self.threshold_with == "mixture":
            threshold = ((ref.sum(dim=-2)) ** 2).sum(dim=-1, keepdim=True)
        else:
            threshold = tgt

        neg_snr = 10 * torch.log10(noise + self.temp * threshold + eps)
        if not self.only_denominator:
            neg_snr -= 10 * torch.log10(tgt + eps)

        return neg_snr
        # return 10 * torch.log10(noise + self.temp * tgt + eps)

    def forward(self, est, ref, return_est=False, return_mean=True):
        # if input shape is (n_batch, n_time)
        if ref.ndim == 2 or not self.solve_perm:
            return self.singlesrc_thresSNR(est, ref)
        else:
            return self.pit_loss(
                est, ref, return_est=return_est, return_mean=return_mean
            )


class ThresSNRLossWithInactiveSource(nn.Module):
    def __init__(
        self,
        n_src,
        snr_max=30,
        inactive_thres=-60,
        only_denominator=True,
    ):
        super().__init__()
        self.n_src = n_src
        self.perms = list(itertools.permutations(range(n_src)))
        self.temp = 10 ** (-snr_max / 10) if snr_max is not None else 0
        self.inactive_thres = inactive_thres
        self.only_denominator = only_denominator

    def l2(self, est, ref):
        return ((est - ref) ** 2).sum(dim=-1)

    def forward(self, est, ref, mix, return_mean=True, return_est=False, eps=1e-8):
        assert self.n_src == est.shape[-2] == ref.shape[-2]

        ref_power = (ref**2).sum(dim=-1)
        mix_power = (mix**2).sum(dim=-1, keepdim=True)

        input_snr = 10 * torch.log10(ref_power / (mix_power + eps))
        activity = input_snr.ge(self.inactive_thres)
        denom_soft_thres = self.temp * (activity * ref_power + (~activity) * mix_power)

        snrs = []
        for p in self.perms:
            est_permed = est[..., p, :]
            snr = 10 * torch.log10(self.l2(est_permed, ref) + denom_soft_thres + eps)
            # snr = snr * activity  # ignores zero-reference
            if not self.only_denominator:
                snr = snr - 10 * torch.log10(activity * ref_power + (~activity) + eps)
            snr = snr.mean(dim=-1)
            snrs.append(snr)
        snrs = torch.stack(snrs, dim=-1)
        loss, idx = torch.min(snrs, dim=-1)

        if return_mean:
            loss = loss.mean()

        if return_est:
            for b in range(est.shape[0]):
                est[b] = est[b, self.perms[idx[b]]]
            return loss, est
        else:
            return loss


class ThresMultiL1LosswithInactiveSource(nn.Module):
    def __init__(
        self,
        n_src,
        stft_params,
        snr_max=30,
        inactive_thres=-60,
        only_denominator=True,
        logarithm=True,
    ):
        super().__init__()
        self.n_src = n_src
        self.perms = list(itertools.permutations(range(n_src)))
        self.temp = 10 ** (-snr_max / 10) if snr_max is not None else 0
        self.inactive_thres = inactive_thres
        self.only_denominator = only_denominator
        self.logarithm = logarithm
        self.stft = []

        for params in stft_params:
            self.stft.append(utils.STFTEncoder(**params))

    def l2(self, est, ref):
        return ((est - ref) ** 2).sum(dim=-1)

    def forward(self, est, ref, mix, return_mean=True, return_est=False):
        n_batch, n_src, n_samples = ref.shape
        assert est.dtype == torch.float32
        eps = 1e-8

        assert self.n_src == est.shape[-2] == ref.shape[-2]

        ref_power = (ref**2).sum(dim=-1)
        mix_power = (mix**2).sum(dim=-1, keepdim=True)

        input_snr = 10 * torch.log10(ref_power / (mix_power + eps))
        activity = input_snr.ge(self.inactive_thres)
        denom_soft_thres = self.temp * (activity * ref_power + (~activity) * mix_power)

        # apply stft before for loop
        # without clone, gradient cannot be computed
        ESTs, REFs = [], []
        for stft in self.stft:
            ilens = torch.tile(torch.LongTensor([n_samples]), (n_batch * n_src,))
            ref_tmp, est_tmp = ref.reshape(-1, n_samples), est.reshape(-1, n_samples)
            REF = abs(stft(ref_tmp, ilens)[0])
            EST = abs(stft(est_tmp.clone(), ilens)[0])
            T, F = REF.shape[-2:]
            REF = REF.reshape(n_batch, n_src, T, F)
            EST = EST.reshape(n_batch, n_src, T, F)

            if self.logarithm:
                REF = torch.log(REF + eps)
                EST = torch.log(EST + eps)
            REFs.append(REF)
            ESTs.append(EST)

        # start loss computation
        losses = []
        for p in self.perms:
            est_permed = est[..., p, :]
            snr = 10 * torch.log10(self.l2(est_permed, ref) + denom_soft_thres + eps)
            # snr = snr * ((~activity) * 0.1 + activity)
            if not self.only_denominator:
                snr = snr - 10 * torch.log10(activity * ref_power + (~activity) + eps)

            # time-domain L1 loss
            l1_loss = abs(ref - est_permed).mean(dim=-1)
            # frequency domain multi-resolution L1 loss
            mr_loss = 0
            for s in range(len(self.stft)):
                mr_loss += abs(REFs[s] - ESTs[s][..., p, :, :]).mean(dim=(-1, -2))
            loss = snr + l1_loss + mr_loss / len(self.stft)

            losses.append(loss.mean(dim=-1))

        losses = torch.stack(losses, dim=-1)
        min_loss, idx = torch.min(losses, dim=-1)

        for b in range(est.shape[0]):
            est[b] = est[b, self.perms[idx[b]]]
        if return_mean:
            min_loss = min_loss.mean()

        if return_est:
            return min_loss, est
        else:
            return min_loss


def sisdr_fuss(targets, est_targets, zero_mean=False, eps=1e-8):
    """
    More accurate sisdr computation proposed in [1].
    Note: This function does not solve permutation.

    Args:
        targets: torch.Tensor, shape (... n_src, n_time)
            The reference source signals.
        est_targets: torch.Tensor, shape (... n_src, n_time)
            The estimated source signals.

    References:
        [1] Wisdom, Scott, et al. "What’s all the fuss about free
        universal sound separation data?." ICASSP, 2021.
    """
    if targets.size() != est_targets.size():
        raise TypeError(
            f"Inputs must be of shape [n_src, time], got {targets.size()} and {est_targets.size()} instead"
        )
    if zero_mean:
        mean_source = torch.mean(targets, dim=-1, keepdim=True)
        mean_estimate = torch.mean(est_targets, dim=-1, keepdim=True)
        targets = targets - mean_source
        est_targets = est_targets - mean_estimate

    num = torch.sum(est_targets * targets, dim=-1)
    denom = (
        torch.sqrt((targets**2).sum(dim=-1))
        * torch.sqrt((est_targets**2).sum(dim=-1))
        + eps
    )
    rho = num / denom

    sisdr = 10 * torch.log10((rho**2 + eps) / (1 - rho**2 + eps))

    return sisdr


def sisdr_fuss_pit(ref, est, zero_mean=False, eps=1e-8, return_perm=False):
    nsrc_est = est.shape[-2]
    nsrc_ref = ref.shape[-2]
    permutations = torch.LongTensor(
        list(itertools.permutations(torch.arange(nsrc_est), nsrc_ref))
    )
    sisdrs = []

    for perm in permutations:
        est_permuted = est[..., perm, :]
        sisdr = sisdr_fuss(est_permuted, ref, zero_mean=zero_mean, eps=eps)
        sisdrs.append(sisdr)

    # num_perms x nsrc_ref
    sisdrs = torch.stack(sisdrs, 0)
    best_sisdr, idx = torch.max(sisdrs.sum(dim=-1, keepdim=True), dim=0)

    if return_perm:
        return sisdrs[idx[0]], permutations[idx[0]]
    else:
        return sisdrs[idx[0]]


def sparsity_loss(y, eps=1e-8):
    """Source sparsity loss for MixIT proposed in [1].

    Args:
        y: torch.Tensor, (..., n_src, n_time)
            Separated source signals.

    References:
        [1] Wisdom, Scott, et al. "Sparse, efficient, and semantic mixture invariant
        training: Taming in-the-wild unsupervised sound separation."WASPAA, 2021.
    """

    rm = torch.sqrt(torch.clamp(torch.mean(y**2, dim=-1), min=eps))
    r1 = torch.sum(rm, dim=-1)
    r2 = torch.sqrt(torch.sum(rm**2, dim=-1))

    loss = r1 / (r2 * y.shape[-2] + eps)

    return loss.mean()
