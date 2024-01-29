from typing import Callable, Dict, Optional

import torch

from .selfremixing_loss import SelfRemixingLoss, SupervisedSelfRemixingLoss


class RemixITLoss(SelfRemixingLoss):
    def __init__(
        self,
        student_separator: torch.nn.Module,
        teacher_separator: torch.nn.Module,
        loss_func: Callable,
        constrained_batch_shuffle: Optional[bool] = True,
        channel_shuffle: Optional[bool] = True,
        channel_shuffle_except_last: Optional[bool] = False,
        nsrc_to_remix: Optional[int] = None,
        student_mixconsis: Optional[bool] = False,
        teacher_mixconsis: Optional[bool] = True,
        normalize: Optional[bool] = False,
        pit_from: Optional[str] = None,
    ):
        super().__init__(
            student_separator,
            teacher_separator,
            loss_func,
            remixit_loss_weight=1.0,
            selfremixing_loss_weight=0.0,
            constrained_batch_shuffle=constrained_batch_shuffle,
            channel_shuffle=channel_shuffle,
            channel_shuffle_except_last=channel_shuffle_except_last,
            nsrc_to_remix=nsrc_to_remix,
            solver_mixconsis=student_mixconsis,
            shuffler_mixconsis=teacher_mixconsis,
            pit_from=pit_from,
        )
