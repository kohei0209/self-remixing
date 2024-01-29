from typing import Optional, Tuple

import losses
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


class FUSSTrainer(AbsTrainer):
    def __init__(
        self,
        config,
        separator,
        optimizer,
        train_loader,
        valid_loader,
        teacher_separator=None,
        valid_loader2=None,
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
            and config["dataset_conf"]["params"]["allow_non_overlap"]
        ):
            if config["loss"] == "snr":
                remixit_loss_func = ThresSNRLossWithInactiveSource(
                    n_src=4,
                    **config["loss_conf"],
                )
            elif config["loss"] == "mrl1":
                remixit_loss_func = ThresMultiL1LosswithInactiveSource(
                    n_src=4,
                    **config["loss_conf"],
                )
            self.compute_loss = SupervisedSelfRemixingLossWithZeroReferences(
                self.separator,
                remixit_loss_func,
                selfremixing_loss_func=self.loss_func,
                **config["algo_conf"],
            )

    def separator_update(
        self,
        mix: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        n_refs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update separators.

        Parameters
        ----------
        mix: torch.Tensor, (..., n_chan, n_samples)
            Time-domain mixture.
        ref: torch.Tensor, optional,  (..., n_src, n_samples)
            Time-domain reference singals.
            References are necessary only when supervised learning.

        Returns
        -------
        y: torch.Tensor, (..., n_src, n_samples)
            Time-domain separated signals.
        loss: torch.Tensor, (1)
            Total loss value.
        """

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if self.algo == "pit":
                y, loss = self.compute_loss(mix, ref, n_refs)
            elif self.algo in ["sup_remixit", "sup_selfremixing"]:
                y, loss = self.compute_loss(mix, ref, noise=None)
            else:
                assert ref is None and n_refs is None
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

        return y.to(torch.float32), loss, grad_norm

    def train(self, epoch):
        self.separator.train()
        self.epoch = epoch

        n_data = 0
        loss_total = 0.0

        for step, data in tqdm(
            enumerate(self.train_loader),
            desc="train",
            leave=False,
            total=len(self.train_loader),
        ):
            mix, ref, n_refs = data
            mix, ref = mix.to(self.device), ref.to(self.device)

            # make batchsize even number
            mix = mix[: (mix.shape[0]) // 2 * 2]
            ref = ref[: mix.shape[0]]

            n_batch = mix.shape[0]
            n_data += n_batch

            if n_batch == 0:
                continue

            if self.algo == "pit":
                ref_clone = ref.clone()
                y, loss, grad_norm = self.separator_update(mix, ref_clone, n_refs)
            else:
                y, loss, grad_norm = self.separator_update(mix)

            loss = loss.to("cpu").detach().numpy().copy()
            loss_total += loss

            results = {
                "epoch": epoch,
                "loss": loss,
                "grad_norm": grad_norm,
            }
            if self.use_wandb and step % 10 == 0:
                wandb.log(results)

        results_summary = {"loss": round(loss_total / len(self.train_loader), 3)}

        if self.teacher_update_timing == "epoch":
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
        loss_total = 0.0
        n_data = 0

        if self.teacher_update_timing is not None:
            keys = [
                "student",
                "teacher",
            ]
        else:
            keys = ["student"]

        results = {}
        results_summary = {}
        n_src = ["_1src", "_2src", "_3src", "_4src"]
        for key in keys:
            for n in n_src:
                results[key + n] = 0

        sources_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for data in tqdm(self.valid_loader, desc="valid", leave=False):
            mix, ref, n_refs = data
            mix, ref = mix.to(self.device), ref.to(self.device)

            # make batchsize even number
            mix = mix[: (mix.shape[0]) // 2 * 2]
            ref = ref[: mix.shape[0]]

            n_batch = mix.shape[0]
            n_data += n_batch

            if n_batch == 0:
                continue

            separation_results = {}

            if self.algo == "pit":
                y, loss = self.compute_loss(mix, ref, n_refs)
                separation_results["student"] = y
            elif self.algo in ["sup_remixit", "sup_selfremixing"]:
                ref_clone = ref.clone()
                _, loss = self.compute_loss(mix, ref_clone, noise=None)
                y = self.separator(mix)
                separation_results["student"] = y
            else:
                if self.algo == "mixit" and self.compute_loss.standardization:
                    std = torch.std(mix, dim=-1, keepdim=True)
                    mix /= std
                    ref /= std[..., None, :]
                y, loss = self.compute_loss(mix)

            if self.algo not in ["pit", "remixit", "selfremixing"]:
                separation_results["student"] = self.separator(mix)

            elif self.algo in ["remixit", "selfremixing"]:
                separation_results["teacher"] = y
                separation_results["student"] = self.separator(mix)

            loss = loss.cpu().detach().numpy()

            mix, ref = mix.to(torch.float64), ref.to(torch.float64)
            mix_sisdr = []
            # compute sisdr of mixtures first
            for b in range(n_batch):
                if n_refs[b] == 1:
                    mix_sisdr.append(0)
                else:
                    mix_b = torch.tile(mix[[b], None], (1, n_refs[b], 1))
                    ref_b = ref[[b], : n_refs[b]]
                    mix_sisdr.append(
                        losses.sisdr_fuss(ref_b, mix_b, eps=1e-8)
                        .mean()
                        .to("cpu")
                        .detach()
                        .numpy()
                        .copy()
                    )
                sources_counts[int(n_refs[b])] += 1

            # compute sisdr of separated sources
            for key in keys:
                for b in range(n_batch):
                    ref_b = ref[[b], : n_refs[b]]

                    sisdr = (
                        losses.sisdr_fuss_pit(
                            ref_b,
                            separation_results[key][[b]].to(torch.float64),
                            eps=1e-8,
                        )
                        .mean()
                        .to("cpu")
                        .detach()
                        .numpy()
                        .copy()
                    )

                    results[f"{key}_{int(n_refs[b])}src"] += sisdr - mix_sisdr[b]

            loss_total += loss

        loss_mean = loss_total / len(self.valid_loader)

        # assert n_data == 1000
        for key in keys:
            trf = 0
            msi = 0
            for n in range(1, 5, 1):
                trf += results[f"{key}_{n}src"] / n_data
                if n > 1:
                    msi += results[f"{key}_{n}src"]
                results[f"{key}_{n}src"] = round(
                    results[f"{key}_{n}src"] / sources_counts[n], 3
                )
            msi /= sources_counts[2] + sources_counts[3] + sources_counts[4]

            results[f"{key}_trf"] = round(trf, 3)
            results_summary[f"{key}_trf"] = round(trf, 3)
            results_summary[f"{key}_msi"] = round(msi, 3)
            results_summary[f"{key}_1s"] = results[f"{key}_1src"]

        results["val_loss"] = round(loss_mean, 3)

        if self.use_wandb:
            if (
                self.algo == "selfremixing"
                and self.compute_loss.loss_thresholder is not None
            ):
                loss_thres = (
                    self.compute_loss.loss_thresholder.get_current_threshold()
                    .numpy()
                    .copy()
                )
                results["loss_threshold"] = loss_thres
            wandb.log(results)

        self.optimizer.epoch_end(loss_mean)

        return results_summary
