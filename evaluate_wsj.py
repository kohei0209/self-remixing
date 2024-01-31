import argparse
import json
from pathlib import Path

import fast_bss_eval
import my_torch_utils as utils
import numpy as np
import soundfile as sf
import torch
import yaml
from datasets import call_dataset
from datasets.wsjmix_dataset import stages as wsjmix_stages
from models import Separator
from pystoi.stoi import stoi as pystoi
from torch.utils.data import DataLoader

sample_rate = 8000


def save_audios(output_path, data, fs):
    data = (data / abs(data).max()) * 0.95
    sf.write(str(output_path), data.numpy().T, fs)


def test(args):
    if args.model_dir is not None:
        with open(args.model_dir / "train_setting.yaml") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    # prepare dataloader
    test_dataset = call_dataset(
        config["dataset"],
        args.data_dir,
        stage=args.stage,
        num_data=None,
        kwargs=config["dataset_conf"]["params"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    if args.teacher_or_student == "teacher":
        args.criteria += "_teacher"

    if args.epochs is None:
        with open(args.model_dir / "train_result.json") as f:
            train_result = json.load(f)
        args.epochs = utils.search_epochs_with_best_criteria2(
            args.model_dir,
            train_result,
            args.num_epochs,
            args.criteria,
        )
        print(f"Use best {args.num_epochs} epochs: {args.epochs}")

    else:
        print(f"Use specified epochs: {args.epochs}")

    output_dir = args.model_dir / "enhanced"
    # output_dir = output_dir / 'mixture'

    if config["dataset"] in ["wsjmix", "smswsj"]:
        stage_name = wsjmix_stages[args.stage]
        test_dataset.return_paths = True
        test_dataset.return_noise = False
    else:
        raise NotImplementedError()

    # need to modify dataset config in MixIT
    if config["algo"] == "mixit" and config["algo_conf"]["normalize"]:
        test_dataset.normalization = True

    eval_output_dir = output_dir / stage_name
    wav8k_output_dir = output_dir / "wav8k"

    eval_output_dir.mkdir(exist_ok=True, parents=True)

    if not args.skip_saving_wavs:
        wav8k_output_dir.mkdir(exist_ok=True, parents=True)
        print("Output dir:", output_dir)
    else:
        print("Wavs are not generated !!")

    model_filename = "separator"
    if args.teacher_or_student == "teacher":
        assert (
            args.criteria != "sisdr"
        ), "--criteria sisdr is for student, need to choose teacher_sisdr"
        model_filename = "teacher_" + model_filename
        print("TEACHER model is evaluated rather than STUDENT")

    # load pre-trained separation model
    separator = Separator(config)
    state_dict = utils.average_model_params(
        args.model_dir, args.epochs, filename=model_filename
    )
    separator.load_state_dict(state_dict)
    separator.to(device).eval()

    assert sample_rate in [8000, 16000]

    try:
        nsrc_to_remix = config["algo_conf"]["nsrc_to_remix"]
    except KeyError:
        nsrc_to_remix = None

    metrics = ["sisdr", "sisir", "sisar", "stoi"]
    total_results = {}
    for m in metrics:
        total_results[m] = 0

    results = []
    f = open(eval_output_dir / "results.txt", "w")

    for i, data in enumerate(test_loader):
        mix, ref, ids, kaldi_trans = data
        mix, ref = mix.to(device), ref.to(device)

        with torch.no_grad():
            y = separator(mix)

        m = min(y.shape[-1], ref.shape[-1])
        y, ref, mix = y[..., :m], ref[..., :m], mix[..., :m]

        # ensure mixture consistency if specified
        if args.mixture_consistency:
            # if the model has more outputs but uses only fewer sources for remixing
            if nsrc_to_remix is not None and y.shape[-2] > nsrc_to_remix:
                y = utils.most_energetic(y, n_src=nsrc_to_remix)
            y = utils.mixture_consistency(y, mix)

        # select sources that have the highest powers
        # if not evaluating with oracle assignment
        if args.power_based_selection:
            # pit ensures that speeches are output from the first two channels
            if config["algo"] == "pit":
                y = y[..., : ref.shape[-2], :]
            # if not pit, select the same number of sources as reference
            else:
                y = utils.most_energetic(y, n_src=ref.shape[-2])
            assert y.shape == ref.shape

        ref = ref.to(torch.float64)
        y = y.to(torch.float64)

        # solve permutation in terms of SI-SDR
        ref, y = ref[0], y[0]  # remove batch dimension
        sisdr, perm = fast_bss_eval.si_sdr(ref, y, return_perm=True)
        y = y[perm]
        _, sisir, sisar = fast_bss_eval.si_bss_eval_sources(
            ref, y, compute_permutation=False
        )

        ref, y = ref.to("cpu").detach(), y.to("cpu").detach()

        assert y.shape == ref.shape

        # save wavs
        if config["dataset"] in ["wsjmix", "smswsj"]:
            mix_path, ref1_path, ref2_path = ids
            json_id = Path(mix_path[0]).name
            json_id1, json_id2 = (
                Path(ref1_path[0]).name,
                Path(ref2_path[0]).name,
            )
            y1_output_path_8k = wav8k_output_dir / json_id1
            y2_output_path_8k = wav8k_output_dir / json_id2

        if not args.skip_saving_wavs:
            y1_tosave = y[[0]]
            y2_tosave = y[[1]]
            save_audios(y1_output_path_8k, y1_tosave.to(torch.float32), sample_rate)
            save_audios(y2_output_path_8k, y2_tosave.to(torch.float32), sample_rate)

        # eval stoi
        y = y.numpy().copy()
        ref = ref.numpy().copy()

        sisdr = sisdr.cpu().detach().numpy().copy()
        sisir = sisir.cpu().detach().numpy().copy()
        sisar = sisar.cpu().detach().numpy().copy()

        stoi = []
        for s in range(y.shape[0]):
            stoi.append(pystoi(ref[s], y[s], sample_rate))

        result = {
            json_id: {
                json_id1: {
                    "sisdr": float(sisdr[0]),
                    "sisir": float(sisir[0]),
                    "sisar": float(sisar[0]),
                    "stoi": float(stoi[0]),
                    "wav_path_8k": str(y1_output_path_8k),
                },
                json_id2: {
                    "sisdr": float(sisdr[1]),
                    "sisir": float(sisir[1]),
                    "sisar": float(sisar[1]),
                    "stoi": float(stoi[1]),
                    "wav_path_8k": str(y2_output_path_8k),
                },
            }
        }

        for m in metrics:
            metric = 0
            for n, r in enumerate(result[json_id].values()):
                metric += r[m]
            total_results[m] += metric / (n + 1)

        results.append(result)

        ith_result = "{:.0f}-th sample | SISDR: {:.3f}  SISIR: {:.3f}  SISAR: {:.3f}  STOI: {:.4f}".format(
            i,
            sisdr.mean(),
            sisir.mean(),
            sisar.mean(),
            np.mean(stoi),
        )

        f.write(ith_result + "\n")

        if args.verbose:
            print(ith_result)

        if args.limit is not None and i == args.limit - 1:
            break

    for m in metrics:
        total_results[m] = round(total_results[m] / (i + 1), 4)

    total_result = "\nTotal Result | SISDR: {:.2f} SISIR: {:.2f} SISAR: {:.2f} STOI: {:.4f}".format(
        total_results["sisdr"],
        total_results["sisir"],
        total_results["sisar"],
        total_results["stoi"],
    )
    print(total_result)
    f.write(total_result)
    f.close()

    with open(eval_output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=3)

    with open(eval_output_dir / "results_summary.json", "w") as f:
        json.dump(total_results, f, indent=2)

    if args.verbose:
        print("EPOCHs", args.epochs)
        print("Output dir is: ", output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--stage", type=str, default="test", choices=["valid", "test"])

    parser.add_argument(
        "--teacher_or_student",
        type=str,
        choices=["student", "teacher"],
        default="student",
        help="Choose the one you want to evaluate from [teacher, student]",
    )

    parser.add_argument(
        "-n",
        "--num_epochs",
        type=int,
        default=5,
        help="Specify number of epochs to be chosen with best criterion."
        "This parser must be specified with --criteria.",
    )
    parser.add_argument(
        "-c",
        "--criteria",
        type=str,
        default="sisdr",
        choices=[
            "loss",
            "sisdr",
            "teacher_sisdr",
        ],
        help="Criteria to select checkpoint."
        "This parser must be specified with --num_epochs.",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        nargs="+",
        help="Specify list of epoch numbers instead of num_epochs and criteria"
        "if we want to evaluate with specific epochs.",
    )

    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("-m", "--mixture_consistency", action="store_true")
    parser.add_argument("-p", "--power_based_selection", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=None)

    parser.add_argument("--skip_saving_wavs", action="store_true")

    args = parser.parse_args()

    test(args)


main()
