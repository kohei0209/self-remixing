import argparse
import copy
import json
import warnings
from pathlib import Path

import soundfile as sf
import torch
import yaml
from espnet_asr import ESPNetPretrainedASR
from tqdm import tqdm
from utilities.wer import get_word_error_rate
from whisper_asr import WhisperASR

warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--eval_one_session", type=int, default=None)
    parser.add_argument("--eval_mixture", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if args.gpu is not None else "cpu"

    if args.eval_mixture:
        with open(args.results_dir / "all_info_monoaural.json") as f:
            separated_results = json.load(f)
    else:
        with open(args.results_dir / "results.json") as f:
            separated_results = json.load(f)
    asr_results = copy.deepcopy(separated_results)

    output_dir = args.results_dir / "asr_result" / args.config.stem
    if args.eval_one_session is not None:
        output_dir = output_dir / ("session" + str(args.eval_one_session))
    else:
        output_dir = output_dir / "all_sessions"

    # define asr model and output directory
    if "whisper" in config["model"]:
        config["device"] = device
        speech2text = WhisperASR(**config)
    else:
        config["params"]["device"] = device
        speech2text = ESPNetPretrainedASR(config["model"], **config["params"])

    output_dir.mkdir(exist_ok=True, parents=True)
    print("Output_dir: ", output_dir)

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

    overlap_ratios = ["0.0S", "0.0L", "10.0", "20.0", "30.0", "40.0"]
    result_per_session = {}
    result_per_overlap = {}

    for i in range(10):
        result_per_session[("session" + str(i))] = {}
        for o in overlap_ratios:
            result_per_session[("session" + str(i))][o] = copy.deepcopy(result_base)
        result_per_session[("session" + str(i))]["Total"] = copy.deepcopy(result_base)

    for o in overlap_ratios:
        result_per_overlap[o] = copy.deepcopy(result_base)
    result_per_overlap["Total"] = copy.deepcopy(result_base)

    # evaluation
    for i, (session_id, info_per_overlap) in enumerate(
        tqdm(list(separated_results.items()))
    ):
        for j, (wav_id, info_per_wav) in enumerate(info_per_overlap.items()):
            session_id_splited = session_id.split("_")
            if session_id_splited[2] == "0.0":
                if session_id_splited[3] == "sil0.1":
                    overlap_ratio = "0.0S"
                elif session_id_splited[3] == "sil2.9":
                    overlap_ratio = "0.0L"
            else:
                overlap_ratio = session_id_splited[2]

            # if not session0, skip
            if args.eval_one_session is not None:
                if not (
                    session_id_splited[5] == "session" + str(args.eval_one_session)
                ):
                    continue

            if args.eval_mixture:
                wav_paths = [info_per_wav["path"]]
            else:
                wav_paths = info_per_wav["separation_results"]
            trans = info_per_wav["transcription"]

            if not isinstance(wav_paths, list):
                wav_paths = [wav_paths]

            scores = []
            estimated_texts = []
            for wav_path in wav_paths:
                # load audio
                speech, fs = sf.read(wav_path)
                # assr
                text = speech2text(speech)
                # text normalization in whisper
                if getattr(speech2text, "normalize", True):
                    trans = speech2text.processor.tokenizer._normalize(trans)
                estimated_texts.append(text)
                # compute wer
                scores.append(get_word_error_rate(trans.split(" "), text.split(" ")))

            # choose score with lower WER
            if len(scores) == 2:
                if scores[0]["wer"] > scores[1]["wer"]:
                    score = scores[1]
                    est = estimated_texts[1]
                else:
                    score = scores[0]
                    est = estimated_texts[0]
            else:
                score = scores[0]
                est = estimated_texts[0]

            asr_results[session_id][wav_id]["asr_results"] = {
                "estimated": text,
                "score": {
                    "wer": score["wer"],
                    "reference_length": len(trans.split(" ")),
                    "correct": score["cor"],
                    "error": {
                        "insertion": score["ins"],
                        "substitution": score["sub"],
                        "deletion": score["del"],
                    },
                },
            }

            # save results per overlap ratio
            result_per_overlap[overlap_ratio]["total_reference_words"] += len(
                trans.split(" ")
            )
            result_per_overlap[overlap_ratio]["error"]["insertion"] += score["ins"]
            result_per_overlap[overlap_ratio]["error"]["deletion"] += score["del"]
            result_per_overlap[overlap_ratio]["error"]["substitution"] += score["sub"]
            result_per_overlap[overlap_ratio]["total_errors"] += (
                score["ins"] + score["del"] + score["sub"]
            )
            result_per_overlap[overlap_ratio]["total_correct_words"] += score["cor"]

            result_per_overlap["Total"]["total_reference_words"] += len(
                trans.split(" ")
            )
            result_per_overlap["Total"]["error"]["insertion"] += score["ins"]
            result_per_overlap["Total"]["error"]["deletion"] += score["del"]
            result_per_overlap["Total"]["error"]["substitution"] += score["sub"]
            result_per_overlap["Total"]["total_errors"] += (
                score["ins"] + score["del"] + score["sub"]
            )
            result_per_overlap["Total"]["total_correct_words"] += score["cor"]

            # save results per session
            session = session_id_splited[5]
            result_per_session[session][overlap_ratio][
                "total_reference_words"
            ] += len(trans.split(" "))
            result_per_session[session][overlap_ratio]["error"][
                "insertion"
            ] += score["ins"]
            result_per_session[session][overlap_ratio]["error"][
                "deletion"
            ] += score["del"]
            result_per_session[session][overlap_ratio]["error"][
                "substitution"
            ] += score["sub"]
            result_per_session[session][overlap_ratio][
                "total_errors"
            ] += (score["ins"] + score["del"] + score["sub"])
            result_per_session[session][overlap_ratio][
                "total_correct_words"
            ] += score["cor"]

            result_per_session[session]["Total"][
                "total_reference_words"
            ] += len(trans.split(" "))
            result_per_session[session]["Total"]["error"][
                "insertion"
            ] += score["ins"]
            result_per_session[session]["Total"]["error"][
                "deletion"
            ] += score["del"]
            result_per_session[session]["Total"]["error"][
                "substitution"
            ] += score["sub"]
            result_per_session[session]["Total"]["total_errors"] += (
                score["ins"] + score["del"] + score["sub"]
            )
            result_per_session[session]["Total"][
                "total_correct_words"
            ] += score["cor"]

            if args.verbose:
                print(session_id)
                print("Ref: ", trans)
                print("Est: ", est)
                print(score, "\n")

    total_words = 0
    total_errors = 0

    print("--------- Total results ---------")
    for i, r in enumerate(overlap_ratios):
        result_per_overlap[r]["wer"] = round(
            100
            * result_per_overlap[r]["total_errors"]
            / result_per_overlap[r]["total_reference_words"],
            3,
        )
        print(r, ": ", result_per_overlap[r]["wer"])
    result_per_overlap["Total"]["wer"] = round(
        100
        * result_per_overlap["Total"]["total_errors"]
        / result_per_overlap["Total"]["total_reference_words"],
        3,
    )
    print("Total: ", result_per_overlap["Total"]["wer"])

    if args.eval_one_session is None:
        for i in range(10):
            for j, r in enumerate(overlap_ratios):
                result_per_session["session" + str(i)][r]["wer"] = round(
                    100
                    * result_per_session["session" + str(i)][r]["total_errors"]
                    / result_per_session["session" + str(i)][r][
                        "total_reference_words"
                    ],
                    3,
                )
            result_per_session["session" + str(i)]["Total"]["wer"] = round(
                100
                * result_per_session["session" + str(i)]["Total"]["total_errors"]
                / result_per_session["session" + str(i)]["Total"][
                    "total_reference_words"
                ],
                3,
            )

        # save results of each sessions
        for i in range(10):
            with open(output_dir / ("results_session" + str(i) + ".json"), "w") as f:
                json.dump(result_per_session["session" + str(i)], f, indent=5)

        with open(output_dir / "results_per_wav.json", "w") as f:
            json.dump(asr_results, f, indent=5)

        with open(output_dir / "results_per_overlap.json", "w") as f:
            json.dump(result_per_overlap, f, indent=5)

    else:
        i = args.eval_one_session
        for j, r in enumerate(overlap_ratios):
            result_per_session["session" + str(i)][r]["wer"] = round(
                100
                * result_per_session["session" + str(i)][r]["total_errors"]
                / result_per_session["session" + str(i)][r]["total_reference_words"],
                3,
            )
            result_per_session["session" + str(i)]["Total"]["wer"] = round(
                100
                * result_per_session["session" + str(i)]["Total"]["total_errors"]
                / result_per_session["session" + str(i)]["Total"][
                    "total_reference_words"
                ],
                3,
            )
        with open(output_dir / ("results_session" + str(i) + ".json"), "w") as f:
            json.dump(result_per_session["session" + str(i)], f, indent=5)

    with open(output_dir / "test_setting.yaml", "w") as f:
        yaml.dump(config, f)

    # session 0 and 1 are dev and test
    if args.eval_one_session is None:
        for i in range(2):
            print("--------- Session " + str(i) + " results ---------")
            for j, r in enumerate(overlap_ratios):
                print(r, ": ", result_per_session["session" + str(i)][r])
            print("Total: ", result_per_session["session" + str(i)]["Total"])
