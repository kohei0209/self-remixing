from functools import partial
from typing import Optional, Tuple

import torch

from .utils import most_energetic


def source_selection(
    sources: torch.Tensor,
    nsrc_to_remix: int,
) -> torch.Tensor:
    n_src = sources.shape[-2]

    if nsrc_to_remix is None or nsrc_to_remix == n_src:
        return sources
    elif n_src > nsrc_to_remix:
        return most_energetic(sources, n_src=nsrc_to_remix)


def channel_shuffle(
    sources: torch.Tensor,
    except_last: Optional[bool] = False,
) -> torch.Tensor:
    if except_last:
        last_channel = sources[:, [-1]]
        sources = sources[:, :-1]

    for b in range(sources.shape[0]):
        p = torch.randperm(sources.shape[1])
        sources[b] = sources[b, p]

    if except_last:
        sources = torch.cat((sources, last_channel), dim=1)

    return sources


def prepare_permutations(n_batch, n_src, constrained=False):
    perms = [torch.randperm(n_batch)]
    while len(perms) < n_src:
        perm = torch.randperm(n_batch)
        # constrained batch shuffle should satisfy
        # n_batch >= n_src
        if constrained and n_batch >= n_src:
            # if all(not torch.equal(perm, p) for p in perms):
            while any(any(perm == p) for p in perms):
                perm = torch.randperm(n_batch)
            perms.append(perm)
        else:
            perms.append(perm)
    return perms


def prepare_permutations2(
    n_batch,
    n_src,
    constrained=False,
):
    # randomly set perm_mat
    p_vectors = [torch.randperm(n_batch)]
    for n in range(1, n_src, 1):
        # allow original combination, with random permutation
        if not constrained or n_src > n_batch:
            p_tmp = torch.randperm(n_batch)
        # not allow original combination strictly
        else:
            flag = True
            while flag:
                count = 0
                p_tmp = torch.randperm(n_batch)
                for i in range(n):
                    if torch.any(p_vectors[i] == p_tmp):
                        break
                    count += 1
                if count == n:
                    flag = False
            p_vectors.append(p_tmp)
    return p_vectors


def random_batch_remix(
    sources: torch.Tensor,
    constrained_batch_shuffle: Optional[bool] = False,
    apply_channel_shuffle: Optional[bool] = False,
    channel_shuffle_except_last: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert sources.ndim == 3, sources.shape
    n_batch, n_src, n_samples = sources.shape

    # channel shuffle
    if apply_channel_shuffle:
        sources = channel_shuffle(sources, except_last=channel_shuffle_except_last)

    # batch shuffle
    perms = prepare_permutations(n_batch, n_src, constrained=constrained_batch_shuffle)
    shuffled_srcs = torch.zeros_like(sources)
    for i in range(n_src):
        perms[i] = perms[i].to(sources.device)
        shuffled_srcs[:, i] = sources[perms[i], i]

    # pseudo-mixture
    pseudo_mixtures = shuffled_srcs.sum(dim=-2)

    return pseudo_mixtures, shuffled_srcs, perms


def random_batch_remix_back(
    sources: torch.Tensor,
    perms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sources.ndim == 3, sources.shape
    n_batch, n_src, n_samples = sources.shape

    # restore the order
    sources_restored = torch.zeros_like(sources)
    for i in range(n_src):
        sources_restored[perms[i], i] = sources[:, i]

    est_mixtures = sources_restored.sum(dim=-2)

    return est_mixtures, sources_restored


class Remixing:
    def __init__(
        self,
        constrained_batch_shuffle: bool = False,
        channel_shuffle: bool = False,
        channel_shuffle_except_last: bool = False,
    ):
        self.remix_func = partial(
            random_batch_remix,
            constrained_batch_shuffle=constrained_batch_shuffle,
            apply_channel_shuffle=channel_shuffle,
            channel_shuffle_except_last=channel_shuffle_except_last,
        )
        self.reconstruct_func = random_batch_remix_back

    def remix(self, sources):
        return self.remix_func(sources)

    def reconstruct_mix(self, sources, perms):
        return self.reconstruct_func(sources, perms)


if __name__ == "__main__":
    src = torch.randn(4, 3, 8000)
    mix = src.sum(dim=1)
    remix = Remixing(
        constrained_batch_shuffle=False,
        channel_shuffle=False,
        channel_shuffle_except_last=False,
    )
    pseudo_mix, shuffled_src, perms = remix.remix(src)

    est_mix, est_src = remix.reconstruct_mix(shuffled_src, perms)

    # shuffled_src is different from org_src
    print(abs(shuffled_src - src).sum(dim=-1))
    # est_src has to equal to sources
    print(abs(est_src - src).sum(dim=-1))
    # est_src has to equal to sources
    print(abs(est_mix - mix).sum(dim=-1))
