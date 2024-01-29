import fast_bss_eval
import my_torch_utils as utils
import torch
import wandb
from losses import (
    SupervisedSelfRemixingLossWithZeroReferences,
    ThresMultiL1LosswithInactiveSource,
    ThresSNRLossWithInactiveSource,
)
from tqdm import tqdm

from .abs_trainer import AbsTrainer


class WSJTrainer(AbsTrainer):
    def __init__(
        self,
        config,
        separator,
        optimizer,
        train_loader,
        valid_loader,
        teacher_separator=None,
    ):
        super().__init__(
            config,
            separator,
            optimizer,
            train_loader,
            valid_loader,
            teacher_separator=teacher_separator,
        )

        if (
            self.algo in ["sup_remixit", "sup_selfremixing"]
            and "allow_non_overlap" in config["dataset_conf"]["params"]
            and config["dataset_conf"]["params"]["allow_non_overlap"]
        ):
            if config["loss"] == "snr":
                remixit_loss_func = ThresSNRLossWithInactiveSource(
                    n_src=3,
                    **config["loss_conf"],
                )
            elif config["loss"] == "mrl1":
                remixit_loss_func = ThresMultiL1LosswithInactiveSource(
                    n_src=3,
                    **config["loss_conf"],
                )
            self.compute_loss = SupervisedSelfRemixingLossWithZeroReferences(
                self.separator,
                remixit_loss_func,
                selfremixing_loss_func=self.loss_func,
                **config["algo_conf"],
            )

    def train(self, epoch):
        self.separator.train()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

        n_data = 0
        loss_total = 0.0

        for step, data in tqdm(
            enumerate(self.train_loader),
            desc="train",
            leave=False,
            total=len(self.train_loader),
        ):
            if len(data) == 3:
                mix, ref, noise = data
                noise = noise[..., None, :].to(self.device)
            else:
                mix, ref = data
                noise = None
            mix, ref = mix.to(self.device), ref.to(self.device)

            # make batchsize even number
            mix = mix[: (mix.shape[0]) // 2 * 2]
            ref = ref[: mix.shape[0]]
            if noise is not None:
                noise = noise[: mix.shape[0]]

            n_batch = mix.shape[0]
            n_data += n_batch

            if n_batch == 0:
                continue

            if self.supervised:
                _, loss, grad_norm = self.separator_update(mix, ref, noise)
            else:
                _, loss, grad_norm = self.separator_update(mix)

            loss = loss.to("cpu").detach().numpy().copy()
            loss_total += loss

            results = {
                "epoch": epoch,
                "loss": loss,
                "grad_norm": grad_norm,
            }
            if self.use_wandb and step % self.log_interval == 0:
                wandb.log(results)

        results_summary = {"loss": round(loss_total / len(self.train_loader), 4)}

        if self.teacher_update_timing == "epoch":
            assert self.alpha <= 1
            self.teacher_separator = utils.update_teacher_model(
                self.teacher_separator,
                self.separator,
                alpha=self.alpha,
            )
            self.teacher_separator.eval()

        return results_summary

    @torch.no_grad()
    def valid(self):
        self.separator.eval()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

        loss_total = 0.0
        sisdr_total = 0.0
        teacher_sisdr_total = 0.0
        n_data = 0

        for data in tqdm(self.valid_loader, desc="valid", leave=False):
            if len(data) == 3:
                mix, ref, noise = data
                noise = noise[..., None, :].to(self.device)
            else:
                mix, ref = data
                noise = None

            mix, ref = mix.to(self.device), ref.to(self.device)

            # make batchsize even number to compute loss
            mix = mix[: (mix.shape[0]) // 2 * 2]
            ref = ref[: mix.shape[0]]
            if noise is not None:
                noise = noise[: mix.shape[0]]

            n_batch = mix.shape[0]
            n_data += n_batch

            if n_batch == 0:
                continue

            with torch.cuda.amp.autocast(enabled=False):
                if self.supervised:
                    y_student, loss = self.compute_loss(mix, ref, noise)
                    if self.algo in ["sup_remixit", "sup_selfremixing"]:
                        y_student = self.separator(mix)
                elif self.algo in ["remixit", "selfremixing"]:
                    y_teacher, loss = self.compute_loss(mix)
                    y_student = self.separator(mix)
                else:
                    y_student, loss = self.compute_loss(mix)
                loss = loss.cpu().detach().numpy()
                loss_total += loss

                if self.algo == "mixit":
                    std = torch.std(mix, dim=-1, keepdim=True)
                    mix /= std
                    ref /= std[..., None, :]
                    y_student = self.separator(mix)

                m = min(y_student.shape[-1], mix.shape[-1])
                y_student, ref, mix = (
                    y_student[..., :m],
                    ref[..., :m],
                    mix[..., :m],
                )

                # ensure mixture consistency
                if self.algo == "mixit" and self.compute_loss.ensure_mixconsis:
                    y_student = utils.mixture_consistency(y_student, mix)
                elif (
                    self.algo in ["selfremixing", "remixit"]
                    and self.compute_loss.solver_mixconsis
                ):
                    y_student = utils.mixture_consistency(y_student, mix)
                    y_teacher = utils.mixture_consistency(y_teacher, mix)

                # compute sisdr
                sisdr = fast_bss_eval.si_sdr(
                    ref.to(torch.float64), y_student.to(torch.float64)
                )
                sisdr_total = (
                    sisdr_total + sisdr.mean(dim=-1).sum().cpu().detach().numpy()
                )

                if (
                    self.algo in ["remixit", "selfremixing"]
                    and self.teacher_update_timing is not None
                ):
                    sisdr = fast_bss_eval.si_sdr(
                        ref.to(torch.float64), y_teacher.to(torch.float64)
                    )
                    teacher_sisdr_total = (
                        teacher_sisdr_total
                        + sisdr.mean(dim=-1).sum().cpu().detach().numpy()
                    )

        loss_mean = loss_total / len(self.valid_loader)
        results_summary = {
            "loss": round(loss_mean, 4),
            "sisdr": round(sisdr_total / n_data, 4),
        }

        if self.teacher_update_timing is not None:
            results_summary["sisdr_teacher"] = round(teacher_sisdr_total / n_data, 4)

        # learning rate update
        current_lr = self.optimizer.epoch_end(loss_mean)

        if self.use_wandb:
            results_to_log = {"lr": current_lr}
            for key, value in results_summary.items():
                new_key = "valid_" + key
                results_to_log[new_key] = value

            wandb.log(results_to_log)

        return results_summary
