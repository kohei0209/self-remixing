import torch


class Warmup_Wapper(object):
    """
    A wrapper for warming up of the learning rate.
    """

    def __init__(
        self,
        optimizer,
        lr=1e-3,
        min_lr=None,
        patience=2,
        factor=0.98,
        warmup_steps=4000,
        mode="steplr",
        decay_start_epoch=0,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.epoch_num = 0
        self.epoch_counter = 0
        self.decay_start_epoch = decay_start_epoch
        self.mode = mode
        self.min_loss = float(10**10)
        self.param_groups = self.optimizer.param_groups

        assert self.mode in ["steplr", "reducelr"]
        if mode == "steplr":
            print("StepLR, patience is", patience)
        elif mode == "reducelr":
            print("ReduceLR, patience is", patience)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.lr * (self.step_num / self.warmup_steps)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def epoch_end(self, val_loss):
        if (
            self.step_num > self.warmup_steps
            and self.epoch_num >= self.decay_start_epoch
        ):
            if self.mode == "steplr":
                self.epoch_counter += 1
                if self.epoch_counter % self.patience == 0:
                    for param_group in self.optimizer.param_groups:
                        if (
                            self.min_lr is None
                            or param_group["lr"] > self.min_lr
                        ):
                            param_group["lr"] *= self.factor
                        self.epoch_counter = 0

            elif self.mode == "reducelr":
                if val_loss < self.min_loss:
                    self.epoch_counter = 0
                    self.min_loss = val_loss
                else:
                    self.epoch_counter += 1

                if self.epoch_counter == self.patience:
                    for param_group in self.optimizer.param_groups:
                        if (
                            self.min_lr is None
                            or param_group["lr"] > self.min_lr
                        ):
                            param_group["lr"] *= self.factor
                    print("lr is reduced to", param_group["lr"])
                    self.epoch_counter = 0

            else:
                raise NotImplementedError()

        self.epoch_num += 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]
        return current_lr

    def load_state_dict(self, state_dict, device):
        self.__dict__.update(state_dict["scheduler"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def state_dict(self):
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer" and key != "param_groups"
        }
        return optimizer_state_dict, scheduler_state_dict
