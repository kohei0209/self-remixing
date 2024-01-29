from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE


class WhisperASR:
    def __init__(
        self,
        model="openai/whisper-large-v2",
        language="english",
        normalize=True,
        device="cpu",
        ref_channel=0,
    ):
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)
        self.model.eval()
        if device != "cpu":
            self.model.to(device=device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        self.normalize = normalize
        self.ref_channel = ref_channel
        self.device = device

    def __call__(self, audio, sr=16000):
        if isinstance(audio, torch.Tensor) and audio.device != "cpu":
            audio = audio.to("cpu")
        # if multi-channel, input must be (n_chan, n_samples)
        if audio.ndim > 1:
            assert audio.shape[0] > self.ref_channel + 1, audio.shape
            audio = audio[self.ref_channel]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        input_features = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features
        if self.device != "cpu":
            input_features = input_features.to(device=self.device)
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=self.forced_decoder_ids
        )[0]
        transcription = self.processor.decode(
            predicted_ids, skip_special_tokens=True, beam_size=None
        )
        if self.normalize:
            transcription = self.processor.tokenizer._normalize(transcription)
        return transcription


@torch.no_grad()
def transcribe(
    model,
    language,
    audios,
    output,
    normalize=True,
    device="cpu",
    ref_channel=0,
):
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(model)
    model.eval()
    if device != "cpu":
        model.to(device=device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    if output:
        output = open(output, "w")
    iterable = tqdm(audios) if output else audios
    for uid, audio_path in iterable:
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        if audio.ndim > 1:
            assert audio.shape[1] > ref_channel, (audio_path, audio.shape)
            audio = audio[:, ref_channel]
        input_features = processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features
        if device != "cpu":
            input_features = input_features.to(device=device)
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )[0]
        transcription = processor.decode(
            predicted_ids, skip_special_tokens=True, beam_size=None
        )
        if normalize:
            transcription = processor.tokenizer._normalize(transcription)
        if output is None:
            print(f"({uid}) {transcription}")
        else:
            output.write(f"{uid} {transcription}\n")
    if output:
        output.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the directory containing audios or path to the wav.scp file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="-",
        help="Path to the output file for writing transcripts. "
        "If is '-', then write to stdout.",
    )
    parser.add_argument("--model", type=str, default="openai/whisper-large-v2")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ref_channel", type=int, default=0)
    args = parser.parse_args()

    if args.language not in TO_LANGUAGE_CODE:
        raise ValueError(
            f"Language {args.language} is not supported. "
            f"Supported languages are {list(TO_LANGUAGE_CODE.keys())}"
        )

    audio_path = Path(args.audio_path)
    if audio_path.is_dir():
        audio_paths = [(p.stem, str(p)) for p in audio_path.rglob("*.wav")]
    elif audio_path.suffix in (".scp", ".txt", ".lst", ""):
        audio_paths = [line.strip().split(maxsplit=1) for line in audio_path.open()]
    else:
        audio_paths = [(audio_path.stem, str(audio_path))]

    output_file = Path(args.output_file)
    if output_file == Path("-"):
        output_file = None
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    transcribe(
        args.model,
        args.language,
        audio_paths,
        output_file,
        device=args.device,
        ref_channel=args.ref_channel,
    )
