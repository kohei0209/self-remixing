import torch
from my_torch_utils import Remixing


def test_remixing():
    src = torch.randn(4, 3, 8000)
    mix = src.sum(dim=1)
    remix = Remixing(
        constrained_batch_shuffle=False,
        channel_shuffle=False,
        channel_shuffle_except_last=False,
    )
    pseudo_mix, shuffled_src, perms = remix.remix(src)

    est_mix, est_src = remix.reconstruct_mix(shuffled_src, perms)

    # src and est_src should be the same
    assert torch.allclose(src, est_src)
    # est_mix and mix should be the same
    assert torch.allclose(mix, est_mix)


def test_remixing_cbs():
    src = torch.randn(4, 3, 8000)
    mix = src.sum(dim=1)
    remix = Remixing(
        constrained_batch_shuffle=True,
        channel_shuffle=False,
        channel_shuffle_except_last=False,
    )
    pseudo_mix, shuffled_src, perms = remix.remix(src)

    est_mix, est_src = remix.reconstruct_mix(shuffled_src, perms)

    # src and est_src should be the same
    assert torch.allclose(src, est_src)
    # est_mix and mix should be the same
    assert torch.allclose(mix, est_mix)
    # src and shuffled_src should be different
    assert not torch.allclose(src, shuffled_src)
    # pseudo_mix and mix should be different
    assert not torch.allclose(mix, pseudo_mix)
