import math
from typing import Optional, Tuple

import my_torch_utils as utils
import torch
from losses import LossWrapper, get_loss_wrapper


class AbsTrainer:
    def __init__(
        self,
        config,
        separator,
        optimizer,
        train_loader,
        valid_loader,
        teacher_separator=None,
    ):
        # Initialize the scaler for AMP
        self.use_amp = config["amp_params"]["enabled"]
        self.scaler = torch.cuda.amp.GradScaler(**config["amp_params"])

        # dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # separators and optimizer
        self.separator = separator
        self.teacher_separator = teacher_separator
        self.optimizer = optimizer

        # other parameters
        self.config = config
        self.device = config["device"]
        self.algo = config["algo"]
        self.use_wandb = config["use_wandb"]
        self.max_grad_norm = config["max_grad_norm"]

        try:
            self.log_interval = config["log_interval"]
        except KeyError:
            self.log_interval = 100

        self.loss_func = LossWrapper(
            config["loss"],
            **config["loss_conf"],
        )

        # later mixit's loss is wrapped by MixIT's loss wrapper
        if self.algo == "mixit":
            self.loss_func.loss_func.solve_perm = False
        else:
            config["algo_conf"]["pit_from"] = None

        self.supervised = self.algo in [
            "pit",
            "sup_remixit",
            "sup_selfremixing",
        ]

        # prepare loss calculator
        self.compute_loss = get_loss_wrapper(
            self.algo,
            self.separator,
            self.loss_func,
            teacher_separator=self.teacher_separator,
            **config["algo_conf"],
        )

        # teacher update setups for remixit and selfremixing
        if "teacher_update" in config:
            self.teacher_update_timing = config["teacher_update"]["update_timing"]

            # epoch-wise teacher update like original RemixIT
            if self.teacher_update_timing != "step":
                self.alpha = config["teacher_update"]["weight"]

            # step-wise teacher update as in momentum pseudo-labeling
            elif self.teacher_update_timing == "step":
                weight = config["teacher_update"]["weight"]
                # special case, the same condition as MixCycle paper
                if weight == -1:
                    self.alpha = 0
                # expected configuration
                # step-wise update as in MPL paper
                else:
                    self.alpha = math.exp((1 / len(train_loader)) * math.log(weight))
                    print(self.alpha)
            else:
                raise NotImplementedError()
        else:
            if self.teacher_separator is not None:
                # sometimes we want to see the behavior of the teacher separator
                # when it's not updated
                print("!!!!! Teacher separator is NOT UPDATED !!!!!")
            self.teacher_update_timing = None

    def separator_update(
        self,
        mix: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update the solver/student separator.
        Automatic Mixed Precision (AMP) is applied if specified in config.

        Parameters
        ----------
        mix: torch.Tensor, (..., n_chan, n_samples)
            Time-domain mixture.
        ref: torch.Tensor, optional,  (..., n_src, n_samples)
            Time-domain reference singals.
            Note that references are necessary only when supervised learning.
        noise: torch.Tensor, optional,  (..., 1, n_samples)
            Time-domain noise singals.
            If noise is not None, used as ground truth in supervised learning.
            Noise is necessary only when supervised learning.

        Returns
        -------
        y: torch.Tensor, (..., n_src, n_samples)
            Time-domain separated signals.
        loss: torch.Tensor, (1)
            Loss value.
        """
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if self.supervised:
                y, loss = self.compute_loss(mix, ref, noise)
            else:
                assert ref is None and noise is None
                y, loss = self.compute_loss(mix)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        grad_norm = utils.grad_norm(self.separator)
        torch.nn.utils.clip_grad_norm_(self.separator.parameters(), self.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.teacher_update_timing == "step":
            self.teacher_separator = utils.update_teacher_model(
                self.teacher_separator,
                self.separator,
                alpha=self.alpha,
            )
            self.teacher_separator.to(self.device).eval()

        return y, loss, grad_norm

    def train(self, epoch):
        raise NotImplementedError()

    def valid(self):
        raise NotImplementedError()
