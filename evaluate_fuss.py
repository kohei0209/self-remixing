import argparse
import json
import statistics
from pathlib import Path

import losses
import my_torch_utils as utils
import soundfile as sf
import torch
import yaml
from datasets import FUSSDataset
from models import Separator
from torch.utils.data import DataLoader

eps = 1e-8
zero_mean = True


def save_audios(output_path, data, fs):
    data = (data / abs(data).max()) * 0.95
    sf.write(str(output_path), data.numpy().T, fs)


def test(args):
    if args.model_dir is not None:
        with open(args.model_dir / "train_setting.yaml") as f:
            config = yaml.safe_load(f)
        torch.manual_seed(config["seed"])
    else:
        config = {}
        torch.manual_seed(10)

    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    # setup dataloader
    dataset = FUSSDataset(
        args.data_dir,
        args.stage,
        return_audio_id=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    cri = args.teacher_or_student
    if args.mixture_consistency:
        cri += "_mixconsis"
    args.criteria = cri + "_" + args.criteria

    if args.epochs is None:
        with open(args.model_dir / "train_result.json") as f:
            train_result = json.load(f)
        args.epochs = utils.search_epochs_with_best_criteria2(
            args.model_dir,
            train_result,
            args.num_epochs,
            args.criteria,
        )
        # args.epochs = utils.search_epochs_with_best_criteria(
        #     train_result, args.num_epochs, args.criteria
        # )
        print(f"Use best {args.num_epochs} epochs: {args.epochs}")

    else:
        print(f"Use specified epochs: {args.epochs}")

    # define output directory
    output_folder_name = "epoch"
    for i, epoch in enumerate(args.epochs):
        output_folder_name += "-" + str(epoch)
    output_dir = (
        args.model_dir / f"best_{args.criteria}_{output_folder_name}" / args.stage
    )

    eval_output_dir = output_dir / "eval_results"
    wav16k_output_dir = output_dir / "wavs"
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    if args.save_wavs:
        wav16k_output_dir.mkdir(exist_ok=True, parents=True)
        print("Output dir:", output_dir)
    else:
        print("Wavs are not generated !!")

    model_filename = "separator"
    if args.teacher_or_student == "teacher":
        model_filename = "teacher_" + model_filename
        print("TEACHER model is evaluated rather than STUDENT")

    # load pre-trained separation model
    separator = Separator(config)
    state_dict = utils.average_model_params(
        args.model_dir, args.epochs, filename=model_filename
    )
    separator.load_state_dict(state_dict)
    separator.to(device).eval()

    results = []
    results_total = {
        "1src": {"num_data": 0, "sisdr": 0},
        "2src": {"num_data": 0, "sisdr": 0},
        "3src": {"num_data": 0, "sisdr": 0},
        "4src": {"num_data": 0, "sisdr": 0},
    }

    results_total_mixture = {
        "2src": {"num_data": 0, "sisdr": 0},
        "3src": {"num_data": 0, "sisdr": 0},
        "4src": {"num_data": 0, "sisdr": 0},
    }

    f = open(eval_output_dir / "results.txt", "w")
    mix_sisdrs = {2: [], 3: [], 4: []}

    # for i, (audio_id, data) in enumerate(metadata.items()):
    for i, data in enumerate(dataloader):
        mix, ref, n_refs, audio_id = data
        n_refs = int(n_refs)
        mix, ref = mix.to(device), ref[..., :n_refs, :].to(device)
        audio_id = audio_id[0]

        if abs(ref[..., 0, :]).sum() == 0:
            if n_refs == 1:
                print(f"remove the {i}-th sample with only-zero component")
                continue
            else:
                print(f"remove background component from the {i}-th sample")
                ref = ref[..., 1:, :]
                n_refs -= 1

        assert torch.all(
            abs(ref).sum(dim=-1) > 0
        ), f"{i}-th sample is zero {abs(ref.sum(dim=-1))}"
        mix = ref.sum(dim=-2)
        std = torch.std(mix, dim=-1, keepdim=True)
        mix = mix / std

        with torch.no_grad():
            y = separator(mix)

        m = min(y.shape[-1], ref.shape[-1])
        y, ref, mix = y[..., :m], ref[..., :m], mix[..., :m]

        if args.mixture_consistency:
            y = utils.mixture_consistency(y, mix)

        # if y.shape[-2] > ref.shape[-2]:
        #    y = utils.most_energetic(y, n_src=ref.shape[-2])
        #    assert y.shape[-2] == ref.shape[-2]
        y, ref, mix = y[0], ref[0], mix[0]

        sisdr, perm = losses.sisdr_fuss_pit(
            ref.to(torch.float64),
            y.to(torch.float64),
            eps=eps,
            zero_mean=zero_mean,
            return_perm=True,
        )
        assert sisdr.shape[0] == ref.shape[-2]

        # discard the quiet samples
        y = y[..., perm, :]

        if n_refs > 1:
            mix_sisdr = losses.sisdr_fuss(
                ref.to(torch.float64),
                mix.tile(n_refs, 1).to(torch.float64),
                eps=eps,
                zero_mean=zero_mean,
            )

            assert sisdr.shape == mix_sisdr.shape
            sisdr -= mix_sisdr

        sisdr = sisdr.to("cpu").numpy()
        y = y.to("cpu")  # y = y[..., perm, :].to("cpu")
        # assert sisdr.shape[-1] == n_refs

        results_total[f"{n_refs}src"]["num_data"] += 1
        results_total[f"{n_refs}src"]["sisdr"] += sisdr.mean()

        if n_refs > 1:
            for z in range(n_refs):
                mix_sisdrs[n_refs].append(mix_sisdr.to("cpu").numpy()[z])
            results_total_mixture[f"{n_refs}src"]["num_data"] += 1
            results_total_mixture[f"{n_refs}src"]["sisdr"] += (
                mix_sisdr.to("cpu").numpy().mean()
            )

        result = {
            audio_id: {
                "background": {
                    "audio_path": str(wav16k_output_dir / audio_id / "background.wav"),
                    "sisdr": sisdr[0],
                }
            }
        }
        if args.save_wavs:
            (wav16k_output_dir / audio_id).mkdir(parents=True)
            save_audios(
                str(wav16k_output_dir / audio_id / "background.wav"),
                y[..., [0], :],
                16000,
            )
        for n in range(n_refs - 1):
            audio_path = str(wav16k_output_dir / audio_id / (f"foreground{n}.wav"))
            if args.save_wavs:
                save_audios(audio_path, y[..., [n + 1], :], 16000)

            result[audio_id][f"foreground{n}"] = {}
            result[audio_id][f"foreground{n}"]["audio_path"] = (audio_path,)
            result[audio_id][f"foreground{n}"]["sisdr"] = sisdr[n + 1]
        results.append(result)

        ith_result = "{:.0f}-th sample | {:.0f} sources | SISDR: {:.3f}".format(
            i, n_refs, sisdr.mean()
        )
        f.write(ith_result + "\n")
        if args.verbose:
            print(ith_result)
        if args.limit is not None and i == args.limit - 1:
            break

    f.close()

    for z in range(2, 5, 1):
        print(
            f"Min:{min(mix_sisdrs[z])} Max:{max(mix_sisdrs[z])} Median:{statistics.median(mix_sisdrs[z])}"
        )

    trf, msi, total_data = 0, 0, 0
    # print(results_total, "\n\n")
    for n in range(4):
        total_data += results_total[f"{n+1}src"]["num_data"]
        trf += results_total[str(n + 1) + "src"]["sisdr"]
        if n > 0:
            msi += results_total[str(n + 1) + "src"]["sisdr"]
        results_total[str(n + 1) + "src"]["sisdr"] /= results_total[f"{n+1}src"][
            "num_data"
        ]
        results_total[str(n + 1) + "src"]["sisdr"] = round(
            results_total[f"{n+1}src"]["sisdr"], 4
        )
        if n > 0:
            results_total_mixture[f"{n+1}src"]["sisdr"] /= results_total_mixture[
                str(n + 1) + "src"
            ]["num_data"]
            results_total_mixture[f"{n+1}src"]["sisdr"] = round(
                results_total_mixture[str(n + 1) + "src"]["sisdr"], 4
            )
    msi = round(msi / (total_data - results_total["1src"]["num_data"]), 3)
    trf = round(trf / total_data, 3)
    results_total["msi"] = msi
    results_total["trf"] = trf
    print("\n", results_total)
    print("mixture", results_total_mixture)
    print(f"MSi: {msi} | TRF: {trf}")

    with open(eval_output_dir / "results_per_data.json", "w") as f:
        json.dump(results, f, indent=3)
    with open(eval_output_dir / "results_summary.json", "w") as f:
        json.dump(results_total, f, indent=3)

    # print("EPOCHs", args.epochs)
    # print("Output dir is: ", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--stage", type=str, default="test", choices=["train", "valid", "test"]
    )

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
        help="Specify number of epochs to be chosen with best criterion. This parser must be specified with --criteria.",
    )
    parser.add_argument(
        "-c",
        "--criteria",
        type=str,
        default="trf",
        choices=["trf", "msi"],
        help="Criteria to select checkpoint. This parser must be specified with --num_epochs.",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        nargs="+",
        help="List of epoch numbers. Basically we should specify -n and -c instead of -e",
    )

    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("-m", "--mixture_consistency", action="store_true")
    parser.add_argument("--save_wavs", action="store_true")

    args = parser.parse_args()

    test(args)
