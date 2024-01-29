import argparse
import copy
import json
from pathlib import Path

import my_torch_utils as utils
import torch
import torchaudio
import yaml
from datasets import LibriCSSDataset
from models import Separator
from tqdm import tqdm


def save_audios(output_path, data, fs, fs_to_save_audio=None):
    if fs_to_save_audio is not None and fs != fs_to_save_audio:
        data = torchaudio.transforms.Resample(fs, fs_to_save_audio)(data)
    else:
        fs_to_save_audio = fs

    data = data / abs(data).max()
    torchaudio.save(output_path, data, fs_to_save_audio)


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

    # load dataset and dataloader
    config["dataset_conf"]["params"]["return_trans"] = False
    config["dataset_conf"]["params"]["return_data"] = True
    config["dataset_conf"]["params"]["max_audio_len"] = None
    dataset = LibriCSSDataset(
        args.data_dir,
        stage="test",
        **config["dataset_conf"]["params"],
    )

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    with open(args.data_dir / "all_info_monoaural.json") as f:
        all_info_dict = json.load(f)
    output_json = copy.deepcopy(all_info_dict)

    if args.epochs is None:
        with open(args.model_dir / "train_result.json") as f:
            train_result = json.load(f)
        args.epochs = utils.search_epochs_with_best_criteria2(
            args.model_dir,
            train_result,
            args.num_epochs,
            args.criteria,
        )
        print(
            "Search best {:.0f} epochs automatically, epochs:".format(args.num_epochs),
            args.epochs,
        )
    else:
        print("Use specified epochs:", args.epochs)

    # define output directory
    if args.eval_mixture:
        output_dir = args.model_dir.parent.parent / "mixture"
    else:
        output_dir = args.model_dir / "enhanced"
    print(f"Output dir: {str(output_dir)}")

    wav_output_dir = output_dir / "utterance_separation"
    wav_output_dir.mkdir(exist_ok=False, parents=True)

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
    print("Model loaded...")

    for i, data in enumerate(tqdm(test_loader)):
        # load data
        mix, info = data
        mix = mix.to(device)

        # separation
        if args.eval_mixture:
            y = torch.stack((mix, mix), dim=-2)
        else:
            with torch.no_grad():
                y, *_ = separator(mix)

        # post processing
        m = min(y.shape[-1], mix.shape[-1])
        y, mix = y[..., :m], mix[..., :m]

        if y.shape[-2] > 2:
            if y.shape[-2] == 3:
                y = y[:, :2, :]
            else:
                y = utils.most_energetic(y, n_src=2)

        # save audios
        org_path = Path(info["path"][0])
        wav_output_dir_temp = wav_output_dir / org_path.parent.name
        wav_output_dir_temp.mkdir(exist_ok=True, parents=True)

        y1_output_path = wav_output_dir_temp / (org_path.stem + "_0.wav")
        y2_output_path = wav_output_dir_temp / (org_path.stem + "_1.wav")

        save_audios(y1_output_path, y[0, 0, None].cpu(), info["sample_rate"][0])
        save_audios(y2_output_path, y[0, 1, None].cpu(), info["sample_rate"][0])

        output_json[org_path.parent.name][org_path.stem]["separation_results"] = [
            str(y1_output_path),
            str(y2_output_path),
        ]

        if i == args.limit:
            break

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_json, f, indent=4)

    print("EPOCHs", args.epochs)
    print("Output dir is: ", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--data_dir", type=Path, default="/mnt/aoni04/saijo/libri_css")

    parser.add_argument(
        "--teacher_or_student",
        type=str,
        choices=["student", "teacher"],
        default="student",
        help="Choose the one you want to evaluate from [teacher, student]",
    )

    parser.add_argument("-n", "--num_epochs", type=int, default=5)
    parser.add_argument("-e", "--epochs", nargs="+", help="list of epoch numbers")
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument(
        "-c",
        "--criteria",
        type=str,
        default="loss",
        help="criteria to select checkpoint",
        choices=["loss", "sisdr", "wer"],
    )
    parser.add_argument(
        "--eval_mixture",
        action="store_true",
        help="whether to evaluate mixture or not",
    )

    args = parser.parse_args()

    test(args)
