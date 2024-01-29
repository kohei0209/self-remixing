import torch
from torch import nn


class EfficentMixITWrapper(nn.Module):
    """Efficient MixIT loss wrapper.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        kwargs: dummy.

    References
        [1] Wisdom, Scott, et al. "Sparse, efficient, and semantic mixture invariant
        training: Taming in-the-wild unsupervised sound separation."WASPAA, 2021.
    """

    def __init__(self, loss_func, **kwargs):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, est_targets, targets, **kwargs):
        r"""Forward function to find the best permutation and return loss.

        Args:
            est_targets: torch.Tensor, shape (..., n_sources, n_samples).
                Estimated separation outputs in time-domain.
            targets: torch.Tensor, shape (..., n_mixtures, n_samples).
                Mixtures (used as ground truths) in time-domain.

        Returns:
            loss: torch.Tensor, shape (1, ).
                Mean loss value.
        """

        # obtain permutation matrix via least-square
        perm_mat = EfficentMixITWrapper.find_best_perm_with_least_square(
            est_targets, targets
        )
        # obtain estiamted mixtures
        est_mixtures = torch.einsum("...nm,...mt->...nt", perm_mat, est_targets)
        # compute loss
        loss = self.loss_func(est_mixtures, targets)

        return loss.mean()

    @staticmethod
    def find_best_perm_with_least_square(est_targets, targets):
        r"""Find the best permutation via the least square.

        Args:
            est_targets: torch.Tensor, shape (..., n_sources, n_samples).
                Estimated separation outputs in time-domain.
            targets: torch.Tensor, shape (..., n_mixtures, n_samples).
                Mixtures (used as ground truths) in time-domain.

        Returns:
            perm_mat: torch.Tensor, shape (..., n_sources, n_mixtures).
                Optimal permutation matrix.
        """
        # (..., n_samples, n_sources)
        est_targets = est_targets.transpose(-1, -2)
        # (..., n_samples, n_mixtures)
        targets = targets.transpose(-1, -2)

        # least-square
        A = torch.linalg.lstsq(est_targets, targets)[0]
        # get maximum number to obtain permutation matrix
        Amax, _ = torch.max(A, dim=-1, keepdim=True)
        # matrix whose component is 1 if the element is maximum number else 0
        perm_mat = (A == Amax).to(est_targets.dtype)

        return perm_mat.transpose(-1, -2)
