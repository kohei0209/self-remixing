import torch

from .efficient_mixit_wrapper import EfficentMixITWrapper
from .loss_functions import (
    MultiResolutionL1Loss,
    ThresMultiL1LosswithInactiveSource,
    ThresSNRLoss,
    ThresSNRLossWithInactiveSource,
    sisdr_fuss,
    sisdr_fuss_pit,
    sparsity_loss,
)
from .mixit_loss import MixITLoss
from .mixit_wrapper import MixITLossWrapper
from .pit_loss import PITLoss, PITLossWithZeroReference
from .pit_wrapper import PITLossWrapper
from .remixit_loss import RemixITLoss
from .selfremixing_loss import (
    SelfRemixingLoss,
    SemiSupervisedSelfRemixingLoss,
    SupervisedSelfRemixingLoss,
    SupervisedSelfRemixingLossWithZeroReferences,
)

loss_funcs = {
    "snr": ThresSNRLoss,
    "mrl1": MultiResolutionL1Loss,
    "snr_inactive": ThresSNRLossWithInactiveSource,
    "mrl1_inactive": ThresMultiL1LosswithInactiveSource,
}


class LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_func,
        **conf,
    ):
        super().__init__()
        self.loss_func = loss_funcs[loss_func](**conf)

    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            return self.loss_func(*args, **kwargs)


def get_loss_wrapper(
    algo,
    separator,
    loss_func,
    teacher_separator=None,
    **kwargs,
):
    assert algo in [
        "pit",
        "sup_remixit",
        "sup_selfremixing",
        "mixit",
        "remixit",
        "selfremixing",
        "semisup_selfremixing",
    ]

    if algo == "pit":
        assert teacher_separator is None
        return PITLoss(separator, loss_func, **kwargs)
    elif algo == "sup_selfremixing" or algo == "sup_remixit":
        assert teacher_separator is None
        return SupervisedSelfRemixingLoss(separator, loss_func, **kwargs)
    elif algo == "mixit":
        assert teacher_separator is None
        return MixITLoss(separator, loss_func, **kwargs)
    elif algo == "remixit":
        assert teacher_separator is not None
        return RemixITLoss(separator, teacher_separator, loss_func, **kwargs)
    elif algo == "selfremixing":
        assert teacher_separator is not None
        return SelfRemixingLoss(separator, teacher_separator, loss_func, **kwargs)
    elif algo == "semisup_selfremixing":
        assert teacher_separator is not None
        return SemiSupervisedSelfRemixingLoss(
            separator, teacher_separator, loss_func, **kwargs
        )
