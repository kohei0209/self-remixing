import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm
from utilities.wer import get_word_error_rate
from whisper_asr import WhisperASR

warnings.simplefilter("ignore")


def normalize_audio(audio):
    return audio / np.max(abs(audio)) * 0.95


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--eval_mixture", action="store_true")
    parser.add_argument("--eval_clean", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if args.gpu is not None else "cpu"

    with open(args.data_dir / "sms_wsj.json") as f:
        database = json.load(f)
    mix_info = []
    mix_info_tmp = database["datasets"]["test_eval92"]
    for info in mix_info_tmp.values():
        mix_info.append(info)
    asr_results = copy.deepcopy(mix_info)

    # define asr model and output directory
    config["device"] = device
    speech2text = WhisperASR(**config)

    if args.eval_mixture:
        wav_dir = args.results_dir / "mixture"
    elif args.eval_clean:
        wav_dir = args.results_dir / "reference"
    else:
        wav_dir = args.results_dir / "enhanced"

    output_dir = wav_dir / "asr_result_whisper" / config["model"].split("/")[-1]
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Output_dir: ", output_dir)

    # separated results
    wav_dir = wav_dir / "wav8k"

    # preparation of dict for saving results per overlap ratio
    result_base = {
        "total_reference_words": 0,
        "total_errors": 0,
        "total_correct_words": 0,
        "wer": 0,
        "error": {
            "insertion": 0,
            "substitution": 0,
            "deletion": 0,
        },
    }

    # evaluation
    for i, info in enumerate(tqdm(mix_info)):
        transcripts = info["kaldi_transcription"]
        for n in range(2):
            # path to separated signals
            if args.eval_mixture:
                # mixture signal
                wav_path = info["audio_path"]["observation"]
            elif args.eval_clean:
                # ground-truth reverberant clean signal
                wav_path = info["audio_path"]["speech_reverberant_clean"][n]
            else:
                # separated signal
                wav_path = wav_dir / (info["example_id"] + f"_{n}.wav")

            # load audio and normalize it to [-1, 1]
            speech, fs = sf.read(str(wav_path))
            if args.eval_mixture or args.eval_clean:
                speech = normalize_audio(speech)

            # speech recognition
            text = speech2text(speech, sr=fs)
            # text normalization if necessary (when using Whisper)
            if getattr(speech2text, "normalize", True):
                trans = speech2text.processor.tokenizer._normalize(transcripts[n])
            if args.verbose:
                print(trans)
                print(text)
            # compute wer
            score = get_word_error_rate(trans.split(" "), text.split(" "))

            result_base["total_reference_words"] += len(trans.split(" "))
            result_base["total_errors"] += score["ins"] + score["del"] + score["sub"]
            result_base["total_correct_words"] += score["cor"]
            result_base["error"]["insertion"] += score["ins"]
            result_base["error"]["deletion"] += score["del"]
            result_base["error"]["substitution"] += score["sub"]

    result_base["wer"] = round(
        100 * result_base["total_errors"] / result_base["total_reference_words"],
        3,
    )
    # save results
    with open(output_dir / "result.json", "w") as f:
        json.dump(result_base, f, indent=3)

    with open(output_dir / "test_setting.yaml", "w") as f:
        yaml.dump(config, f)

    print("WER: ", result_base["wer"])
    err, total, ins, sub, dele = (
        result_base["total_errors"],
        result_base["total_reference_words"],
        result_base["error"]["insertion"],
        result_base["error"]["substitution"],
        result_base["error"]["deletion"],
    )
    print(f"{err}/{total}, {ins} ins, {dele} del, {sub} sub")
