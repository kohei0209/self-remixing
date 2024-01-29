from typing import Optional, Tuple

import fast_bss_eval
import my_torch_utils as utils
import torch
import wandb
from asr.utilities.wer import get_wer
from tqdm import tqdm as tqdm

from .abs_trainer import AbsTrainer


class LibriCSSTrainer(AbsTrainer):
    def __init__(
        self,
        config,
        separator,
        optimizer,
        train_loader,
        valid_loader,
        teacher_separator=None,
        asr_model=None,
        sup_train_loader=None,
        sup_valid_loader=None,
    ):
        super().__init__(
            config,
            separator,
            optimizer,
            train_loader,
            valid_loader,
            teacher_separator=teacher_separator,
        )
        self.sup_train_loader = sup_train_loader
        self.sup_valid_loader = sup_valid_loader
        self.asr_model = asr_model
        # assert asr_model is not None, "we need asr model for getting wer"

        assert self.teacher_update_timing in [
            "step",
            "epoch",
            "libricss_epoch",
        ]

    def separator_update(
        self,
        mix_libricss: torch.Tensor,
        mix_synthetic: Optional[torch.Tensor] = None,
        ref: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if mix_synthetic is not None:
                _, loss = self.compute_loss(mix_libricss, mix_synthetic, ref, noise)
            else:
                _, loss = self.compute_loss(mix_libricss)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        grad_norm = utils.grad_norm(self.separator)
        torch.nn.utils.clip_grad_norm_(self.separator.parameters(), self.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # teacher (shuffler) model EMA update
        if self.teacher_update_timing == "step":
            self.teacher_separator = utils.update_teacher_model(
                self.teacher_separator,
                self.separator,
                alpha=self.alpha,
            )
            self.teacher_separator.eval()

        return loss, grad_norm

    def train(self, epoch):
        self.separator.train()
        self.epoch = epoch

        loss_total = 0.0
        step_size_libricss = len(self.train_loader)
        libri_iterator = self.train_loader.__iter__()

        if self.sup_train_loader is not None:
            step_size_synthetic = len(self.sup_train_loader)
            step_size = max([step_size_libricss, step_size_synthetic])
            synthetic_iterator = self.sup_train_loader.__iter__()
        else:
            step_size = step_size_libricss

        for step in tqdm(range(step_size), desc="train", leave=False):
            try:
                mix_libricss, _ = libri_iterator.next()
            except StopIteration:
                # All libricss samples are used once
                libri_iterator = self.train_loader.__iter__()
                mix_libricss, _ = libri_iterator.next()

                # teacher (shuffler) model EMA update
                if self.teacher_update_timing == "libricss_epoch":
                    self.teacher_separator = utils.update_teacher_model(
                        self.teacher_separator,
                        self.separator,
                        alpha=self.alpha,
                    )
                    self.teacher_separator.eval()

            if mix_libricss.shape[0] % 2 != 0:
                continue
            mix_libricss = mix_libricss.to(self.device)

            # load supervised data
            if self.sup_train_loader is not None:
                try:
                    data = synthetic_iterator.next()
                except StopIteration:
                    synthetic_iterator = self.sup_train_loader.__iter__()
                    data = synthetic_iterator.next()

                # if self.return_noise:
                #     mix_synthetic, ref, noise = data
                #     noise = noise.to(self.device)
                # else:
                #     mix_synthetic, ref = data
                if len(data) == 3:
                    mix_synthetic, ref, noise = data
                    noise = noise.to(self.device)
                else:
                    mix_synthetic, ref = data
                    noise = None

                # we assume batch sizes are same
                if mix_synthetic.shape[0] != mix_libricss.shape[0]:
                    continue
                # to device
                mix_synthetic = mix_synthetic.to(self.device)
                ref = ref.to(self.device)
            else:
                mix_synthetic = ref = noise = None

            # train one step
            loss, grad_norm = self.separator_update(
                mix_libricss, mix_synthetic, ref, noise
            )
            loss = loss.to("cpu").detach().numpy().copy()

            # write wandb log
            results = {"epoch": epoch, "loss": loss, "grad_norm": grad_norm}
            if self.use_wandb and step % self.log_interval == 0:
                wandb.log(results)

            loss_total += loss

        results_summary = {
            "loss": round(loss_total / step_size, 4),
        }

        return results_summary

    @torch.no_grad()
    def valid(self):
        # teacher (shuffle) model EMA update
        if self.teacher_update_timing == "epoch":
            assert self.alpha < 1
            self.teacher_separator = utils.update_teacher_model(
                self.teacher_separator,
                self.separator,
                alpha=self.alpha,
            )
            self.teacher_separator.eval()

        self.separator.eval()

        errors_total = 0.0
        words_total = 0.0
        sisdr_total = 0.0

        step_size_libricss = len(self.valid_loader)
        libri_iterator = self.valid_loader.__iter__()

        if self.sup_valid_loader is not None:
            step_size_synthetic = len(self.sup_valid_loader)
            step_size = step_size_libricss + step_size_synthetic
            synthetic_iterator = self.sup_valid_loader.__iter__()
        else:
            step_size = step_size_libricss

        for step in tqdm(range(step_size), desc="valid", ncols=100, leave=False):
            data_type = "libricss" if step < step_size_libricss else "synthetic"

            if data_type == "synthetic":
                data = synthetic_iterator.next()
                if len(data) == 3:
                    mix, ref, noise = data
                    noise = noise.to(self.device)
                else:
                    mix, ref = data
                    noise = None
                mix, ref = mix.to(self.device), ref.to(self.device)

            elif data_type == "libricss":
                mix, trans = libri_iterator.next()
                mix = mix.to(self.device)
                trans = trans[0]

            # we assume batchsize is 1
            assert mix.shape[0] == 1

            # standardization
            mix_std = mix.std(dim=-1, keepdim=True) + 1e-8
            mix_mean = mix.mean(dim=-1, keepdim=True)
            mix = (mix - mix_mean) / mix_std

            # separation
            try:
                y = self.separator(mix)
            except RuntimeError:
                print(f"CUDA out of memory: {mix.shape}")
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                try:
                    y = self.separator(mix)
                except RuntimeError:
                    print("Skip this sample")
                    continue

            m = min(y.shape[-1], mix.shape[-1])
            y, mix = y[..., :m], mix[..., :m]

            # we evaluate wer in libricss
            if data_type == "libricss":
                if y.shape[-2] > 2:
                    # semi-supervised learning
                    # if y.shape[-2] == 3:
                    if self.sup_train_loader is not None:
                        y = y[..., :2, :]
                    else:
                        y = utils.most_energetic(y, n_src=2)
                errors = []
                # normalization, neccesary for ASR!
                mx, _ = torch.max(abs(y), dim=-1)
                y = y / mx[..., None]
                # wer evaluation
                for n in range(y.shape[-2]):
                    text = self.asr_model(y[0, n, :])
                    errors.append(
                        get_wer(
                            trans.split(" "),
                            text.split(" "),
                            return_n_errors=True,
                        )
                    )
                    # print("\nGround-truth ",trans)
                    # print("   Estimated ", text)
                # libricss has only one transcription of dominant speaker
                error = min(errors)
                errors_total += error
                words_total += len(trans.split(" "))
            # we evaluate sisdr for supervised dataset
            else:
                ref = ref[..., :m]
                # if pretrained with PIT
                if y.shape[-2] == 3:
                    y = y[..., :-1, :]
                # if trained with MixIT
                elif y.shape[-2] > 3:
                    y = utils.most_energetic(y, n_src=3)

                sisdr = fast_bss_eval.si_sdr(ref, y).mean(dim=-1).sum()
                sisdr_total += sisdr.cpu().detach().numpy()

        wer = (errors_total / words_total) * 100
        # wer = 0.00000

        # learning rate update
        # NOTE: we assume we use steplr
        current_lr = self.optimizer.epoch_end(None)

        results_summary = {"wer": round(wer, 3)}

        if self.sup_valid_loader is not None:
            sisdr_total /= step_size_synthetic
            results_summary["sisdr"] = round(sisdr_total, 2)

        if self.use_wandb:
            results_to_log = {
                "val_wer": round(wer, 3),
                "lr": current_lr,
            }
            if self.sup_valid_loader is not None:
                results_to_log["sisdr"] = round(sisdr_total, 2)
            wandb.log(results_to_log)

        return results_summary
