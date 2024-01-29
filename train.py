import argparse
import json
import random
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Dict

import models
import numpy as np
import torch
import wandb
import yaml
from asr import call_asr_model
from datasets import call_collate_fn, call_dataset
from my_torch_utils import Warmup_Wapper
from torch.utils.data import DataLoader
from trainer import trainers


def save_random_state(file_path, device):
    random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state(device)

    torch.save(
        {
            "random_state": random_state,
            "numpy_random_state": numpy_random_state,
            "torch_random_state": torch_random_state,
            "cuda_random_state": cuda_random_state,
        },
        file_path,
    )


def load_random_state(file_path):
    state = torch.load(file_path)
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.random.set_rng_state(state["torch_random_state"])
    torch.cuda.set_rng_state(state["cuda_random_state"])


def keep_nbest_models(
    nbests: Dict,
    epoch_num: int,
    score: float,
    metric: str,
    min_or_max: str,
    nbest: int,
):
    # nbest: dict, sorted with values
    # keys are epoch numbers
    # values are metric score

    assert min_or_max in ["min", "max"], min_or_max
    if len(nbests) < nbest:
        nbests[epoch_num] = score
        return nbests, epoch_num, None

    scores = nbests.values()
    if min_or_max == "min":
        max_value = max(scores)
        if score < max_value:
            keys_to_remove = [
                key for key, value in nbests.items() if value == max_value
            ]
        else:
            return nbests, None, None
    else:
        min_value = min(scores)
        if score > min_value:
            keys_to_remove = [
                key for key, value in nbests.items() if value == min_value
            ]
        else:
            return nbests, None, None

    # remove other keys if multiple scores have the same values
    key_to_remove = min(keys_to_remove)
    # remove the key
    del nbests[key_to_remove]
    # add new one
    nbests[epoch_num] = score
    return nbests, epoch_num, key_to_remove


def train(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    print(config)

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    model_dir = (
        args.model_output_dir
        / "journal_rerun"
        / config["dataset"]
        / config["algo"]
        / config["name"]
    )
    model_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutputs are in: {str(model_dir)}")

    # set device
    num_gpus = torch.cuda.device_count()
    device = "cuda" if num_gpus > 0 else "cpu"
    assert device == "cuda", f"GPU is not available: {device}"
    config["device"] = device
    print("Using", device)

    try:
        n_batch_valid = config["batch_size_valid"]
    except KeyError:
        n_batch_valid = config["batch_size"]

    # prepare dataset
    stages = ["train", "valid"]
    datasets = {}
    for stage in stages:
        datasets[stage] = call_dataset(
            config["dataset"],
            args.data_dir,
            stage=stage,
            num_data=config["dataset_conf"][f"num_{stage}_data"],
            kwargs=config["dataset_conf"]["params"],
        )
    collate_fn = call_collate_fn(config["dataset"])

    # prepare dataloader
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=n_batch_valid,
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # load supervised dataset for semi-supervised learning
    if config["dataset"] == "libricss":
        if "sup_dataset" in config:
            supervised_datasets = {}
            for stage in stages:
                supervised_datasets[stage] = call_dataset(
                    config["sup_dataset"],
                    args.sup_data_dir,
                    stage=stage,
                    num_data=config["sup_dataset_conf"][f"num_{stage}_data"],
                    kwargs=config["sup_dataset_conf"]["params"],
                )
            sup_collate_fn = call_collate_fn(config["sup_dataset"])
            sup_train_loader = DataLoader(
                supervised_datasets["train"],
                batch_size=config["batch_size"],
                num_workers=config["workers"],
                shuffle=True,
                collate_fn=sup_collate_fn,
            )
            sup_valid_loader = DataLoader(
                supervised_datasets["valid"],
                batch_size=n_batch_valid,
                num_workers=config["workers"],
                shuffle=False,
                collate_fn=sup_collate_fn,
            )
        else:
            sup_train_loader = sup_valid_loader = None

    # prepare separation model
    separator = models.Separator(config)
    # remixit or self-remixing needs teacher model
    load_teacher_model = config["algo"] in [
        "remixit",
        "selfremixing",
        "semisup_selfremixing",
    ]
    if load_teacher_model:
        teacher_separator = models.Separator(config)
    else:
        teacher_separator = None

    # log the number of parameters
    total_params = sum(p.numel() for p in separator.parameters())
    print("\nParams: ", round(total_params / 10**6, 3), "M\n")

    # separators on specified device
    # TODO: support DDP
    # separator.to(device)
    # if len(args.gpu) > 1:
    #     separator = torch.nn.DataParallel(separator, device_ids=args.gpu)
    if teacher_separator is not None:
        teacher_separator.to(device)
        device_ids = list(range(num_gpus))
        if num_gpus > 1:
            teacher_separator = torch.nn.DataParallel(
                teacher_separator,
                device_ids=device_ids,
            )

    # set optimizer
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    optimizer = optimizers[config["optimizer"]](
        separator.parameters(), **config["optimizer_conf"]
    )

    # apply warmup wrapper to optimizer
    assert config["scheduler"] in ["warmup", None]
    if config["scheduler"] == "warmup":
        optimizer = Warmup_Wapper(
            optimizer,
            **config["scheduler_conf"],
        )

    # specify checkpoint directory for resuming training
    if args.resume_from is not None:
        # when specified, resume training from specified checkpoint
        checkpoint_dir = args.resume_from
        print(f"Resume training from specified checkpoint: {str(checkpoint_dir)}")
    elif (model_dir / "checkpoint").exists():
        # automatically resume training from latest checkpoint
        checkpoint_dir = model_dir / "checkpoint"
        print(f"Resume training from latest checkpoint: {str(checkpoint_dir)}")
    else:
        # start training from scratch
        checkpoint_dir = None
        # load pre-trained parameters if specified (e.g., MixIT -> Self-Remixing)
        if "student_init_param" in config:
            print("Load student model from: ", config["student_init_param"])
            separator.load_state_dict(torch.load(config["student_init_param"]))
        if "teacher_init_param" in config:
            print("Load teacher model from: ", config["teacher_init_param"])
            teacher_separator.load_state_dict(torch.load(config["teacher_init_param"]))

    # resume training if necessary
    if checkpoint_dir is not None:
        # re-start epoch number
        with open(model_dir / "train_result.json", "r") as f:
            valid_results = json.load(f)
        if args.resume_from is not None:
            start_epoch = int(re.sub(f"\D", "", checkpoint_dir.name)) + 1
        else:
            start_epoch = valid_results[-1]["epoch"] + 1

        # load random state
        load_random_state(checkpoint_dir / "random_state.pt")

        # load optimizer statedict
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pth",
                map_location="cpu",
            ),
            device,
        )

        # load student model statedict
        separator.load_state_dict(torch.load(checkpoint_dir / "separator.pth"))

        # load teacher model statedict
        assert (teacher_separator is None) == (
            not (checkpoint_dir / "teacher_separator.pth").exists()
        )
        if teacher_separator is not None:
            teacher_separator.load_state_dict(
                torch.load(checkpoint_dir / "teacher_separator.pth")
            )
    else:
        # start training from scratch
        start_epoch = 1
        valid_results = []

    # separators on specified device
    # TODO: support DDP
    separator.to(device)
    if num_gpus > 1:
        separator = torch.nn.DataParallel(separator, device_ids=device_ids)
    # if teacher_separator is not None:
    #    teacher_separator.to(device)
    #    if len(args.gpu) > 1:
    #        teacher_separator = torch.nn.DataParallel(
    #            teacher_separator, device_ids=args.gpu
    #        )

    if config["dataset"] == "libricss":
        config["asr_conf"]["device"] = device
        asr_model = call_asr_model(
            config["asr_model"],
            **config["asr_conf"],
        )
        trainer = trainers[config["dataset"]](
            config,
            separator,
            optimizer,
            train_loader,
            valid_loader,
            teacher_separator=teacher_separator,
            asr_model=asr_model,
            sup_train_loader=sup_train_loader,
            sup_valid_loader=sup_valid_loader,
        )

    else:
        trainer = trainers[config["dataset"]](
            config,
            separator,
            optimizer,
            train_loader,
            valid_loader,
            teacher_separator=teacher_separator,
        )

    # save the training configuration
    with open(model_dir / "train_setting.yaml", "w") as f:
        yaml.dump(config, f)

    # function to keep nbest models
    assert len(config["best_model_criterion"]) == 2
    best_model_criterion = config["best_model_criterion"][0]
    nbest_keeper = partial(
        keep_nbest_models,
        metric=config["best_model_criterion"][0],
        min_or_max=config["best_model_criterion"][1],
        nbest=config["keep_nbest_models"],
    )
    nbest_epochs = {}

    # if specified, run multiple training epochs per validation
    # else, run one training epoch per validation in default
    # but for MixIT, run two training epochs per validation in default
    # since MixIT draws twice as many samples in each training step
    # and the number of training steps per epoch is half as other algorithms
    try:
        train_epochs_per_valid = config["train_epochs_per_valid"]
    except KeyError:
        if config["algo"] == "mixit":
            train_epochs_per_valid = 2
        else:
            train_epochs_per_valid = 1

    # how often to save checkpoint
    # NOTE: the latest checkpoint is always saved
    try:
        save_checkpoint_interval = config["save_checkpoint_interval"]
    except KeyError:
        save_checkpoint_interval = 50

    # initialize wandb logger
    if config["use_wandb"]:
        # config is also saved in Wandb
        # TODO: change entity name!
        wandb.init(config=config, entity="user name", **config["wandb"])

    # Finally, training loop
    for epoch in range(start_epoch, config["max_epoch"] + 1):
        # train one epoch
        for _ in range(train_epochs_per_valid):
            train_results = trainer.train(epoch)

        # validation
        valid_result = trainer.valid()

        # keep nbest scores
        nbest_epochs, added, removed = nbest_keeper(
            nbest_epochs, epoch, valid_result[best_model_criterion]
        )

        # save model parameters
        if added is not None:
            print(f"Parameters at epoch {added} is saved")
            model_dir_epoch = model_dir / ("epoch" + str(added))
            model_dir_epoch.mkdir(exist_ok=True)

            # save student model
            if num_gpus == 1:
                state_dict = separator.to("cpu").state_dict()
            else:
                # DP model somehow has "module"
                state_dict = separator.to("cpu").module.state_dict()
            torch.save(state_dict, model_dir_epoch / "separator.pth")
            separator.to(device)

            # save teahcher model when exists
            if load_teacher_model:
                teacher_state_dict = teacher_separator.to("cpu").state_dict()
                torch.save(
                    teacher_state_dict,
                    model_dir_epoch / "teacher_separator.pth",
                )
                teacher_separator.to(device)

        # delete model parameters that is not nbest
        if removed is not None:
            print(f"Parameters at epoch {removed} is removed")
            remove_dir = model_dir / ("epoch" + str(removed))
            shutil.rmtree(remove_dir)

        # save new checkpoint and delete the previous one
        # save student model
        # if epoch % save_checkpoint_interval == 0:
        #     checkpoint_dir = model_dir / f"checkpoint_epoch{epoch}"
        #     print(f"Saved checkpoint at {epoch} epoch")
        # else:
        #     checkpoint_dir = model_dir / "checkpoint"
        #     if checkpoint_dir.exists():
        #         shutil.rmtree(checkpoint_dir)
        checkpoint_dir = model_dir / "checkpoint"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

        if num_gpus == 1:
            state_dict = separator.to("cpu").state_dict()
        else:
            state_dict = separator.to("cpu").module.state_dict()
        separator.to(device)
        torch.save(state_dict, checkpoint_dir / "separator.pth")

        # save teahcher model when self-training
        if teacher_separator is not None:
            teacher_state_dict = teacher_separator.to("cpu").state_dict()
            torch.save(
                teacher_state_dict,
                checkpoint_dir / "teacher_separator.pth",
            )
            teacher_separator.to(device)

        # optimizer statedict
        save_random_state(checkpoint_dir / "random_state.pt", "cpu")

        # optimizer and scheduler state dict
        optimizer_state_dict, scheduler_state_dict = optimizer.state_dict()
        opt_sd = {
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }
        torch.save(opt_sd, checkpoint_dir / "optimizer.pth")

        # mixed precision, scaler statedict
        if config["amp_params"]["enabled"]:
            torch.save(
                trainer.scaler.state_dict(),
                checkpoint_dir / "amp_grad_scler.pth",
            )

        # save checkpoint every "save_checkpoint_interval" epochs
        # this is not overwritten and used for resuming training
        if epoch % save_checkpoint_interval == 0:
            _ = shutil.copytree(
                checkpoint_dir,
                model_dir / f"checkpoint_epoch{epoch}",
                dirs_exist_ok=True,
            )
            print(f"Saved checkpoint at {epoch} epoch")

        # print and save results summary
        epoch_result = f"EPOCH{epoch}: {str(train_results)} | {str(valid_result)}"
        print(epoch_result)
        mode = "w" if epoch == 0 else "a"
        with open(model_dir / "train_result.txt", mode, encoding="UTF-8") as f:
            f.write(epoch_result + "\n")

        valid_result["epoch"] = epoch
        valid_results.append(valid_result)

        # looks wiered but write results every epoch
        # since sometimes training is interrupted during an epoch
        # and lose some information
        with open(model_dir / "train_result.json", "w") as f:
            json.dump(valid_results, f, indent=0)
        with open(model_dir / "nbest_epochs.json", "w") as f:
            json.dump(nbest_epochs, f, indent=0)

    print("training ends.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--sup_data_dir",
        type=Path,
        default="wsjmix",
        help="supervised dataset used in semi-supervised learning",
    )
    parser.add_argument(
        "--model_output_dir",
        type=Path,
        default="../model",
    )

    parser.add_argument(
        "-r",
        "--resume_from",
        type=Path,
        default=None,
        help="checkpoint to resume training",
    )

    args = parser.parse_args()

    train(args)


main()
