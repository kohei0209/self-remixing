import json
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .utils import get_start_and_end, zeropad_sources

stages = {
    "train": "train_si284",
    "valid": "cv_dev93",
    "test": "test_eval92",
}


class SMSWSJDataset(Dataset):
    def __init__(
        self,
        mixinfo_path,
        stage,
        num_data=None,
        audio_len=6,
        return_noise=False,
        allow_non_overlap=False,
        ref_is_reverb=True,
        normalization=False,
        sample_rate=8000,
        return_paths=False,
        force_single_channel=True,
    ):
        assert stage in stages.keys()

        # prepare mixinfo
        with open(mixinfo_path / "sms_wsj.json") as f:
            database = json.load(f)
        self.mix_info = []
        mix_info_tmp = database["datasets"][stages[stage]]
        for info in mix_info_tmp.values():
            self.mix_info.append(info)

        if num_data is not None:
            self.mix_info = self.mix_info[:num_data]
        if stage == "valid":
            self.mix_info = sorted(
                self.mix_info, key=lambda x: x["num_samples"]["observation"]
            )

        self.stage = stage
        self.ref_is_reverb = ref_is_reverb
        self.return_noise = return_noise
        self.fs = sample_rate
        self.audio_length = self.fs * audio_len
        self.allow_non_overlap = allow_non_overlap
        self.return_paths = return_paths
        self.normalization = normalization
        self.force_single_channel = force_single_channel

        assert force_single_channel, "Training recipe only supports monaural audio."

    def __getitem__(self, idx):
        data = self.mix_info[idx]["audio_path"]

        # get start and end of the audio to be loaded
        if self.stage == "train":
            if self.allow_non_overlap:
                if (
                    self.mix_info[idx]["num_samples"]["observation"] - self.audio_length
                    <= 0
                ):
                    start = 0
                    stop = self.mix_info[idx]["num_samples"]["observation"]
                else:
                    start = random.randint(
                        0,
                        self.mix_info[idx]["num_samples"]["observation"]
                        - self.audio_length,
                    )
                    stop = start + self.audio_length
            else:
                start, stop = get_start_and_end(
                    self.mix_info[idx]["num_samples"]["observation"],
                    self.audio_length,
                    self.mix_info[idx]["offset"],
                    self.mix_info[idx]["num_samples"]["original_source"],
                    margen=self.fs * 0.2,
                )
        else:
            start, stop = 0, -1

        mix_path = data["observation"]
        ref1_path = data["speech_reverberation_early"][0]
        ref2_path = data["speech_reverberation_early"][1]

        loading_params = {"start": start, "stop": stop, "dtype": "float32"}

        mix, fs = sf.read(str(mix_path), **loading_params)
        ref1, fs1 = sf.read(str(ref1_path), **loading_params)
        ref2, fs2 = sf.read(str(ref2_path), **loading_params)

        # smswsj has early and tail reverberation separately
        if self.ref_is_reverb:
            tail1 = sf.read(data["speech_reverberation_tail"][0], **loading_params)[0]
            tail2 = sf.read(data["speech_reverberation_tail"][1], **loading_params)[0]
            ref1, ref2 = ref1 + tail1, ref2 + tail2

        mix = torch.from_numpy(mix)
        ref1 = torch.from_numpy(ref1)
        ref2 = torch.from_numpy(ref2)

        assert self.fs == fs == fs1 == fs2

        if self.return_noise:
            noise_path = data["noise_image"]
            noise, fn = sf.read(str(noise_path), **loading_params)
            assert fn == fs

            noise = torch.from_numpy(noise)

        # if force_single_channel, take only first channel
        if self.force_single_channel and mix.ndim > 1:
            mix = mix[:, 0]
            ref1 = ref1[:, 0]
            ref2 = ref2[:, 0]
            if self.return_noise:
                noise = noise[:, 0]

        # preprocessing
        if self.normalization:
            std = torch.std(mix, dim=0, keepdim=True) + 1e-8
            mix /= std
            ref1 /= std
            ref2 /= std
            if self.return_noise:
                noise /= std

        # zero padding if audio length is shorter than desired
        if self.stage == "train" and mix.shape[-1] < self.audio_length:
            span = random.randint(0, self.audio_length - mix.shape[-1] - 1)
            mix = zeropad_sources(mix, self.audio_length, span)
            ref1 = zeropad_sources(ref1, self.audio_length, span)
            ref2 = zeropad_sources(ref2, self.audio_length, span)
            if self.return_noise:
                noise = zeropad_sources(noise, self.audio_length, span)

        ref = torch.stack((ref1, ref2), dim=0)

        if self.return_noise:
            if self.return_paths:
                return (
                    mix,
                    ref,
                    noise,
                    [mix_path, ref1_path, ref2_path, noise_path],
                    self.mix_info[idx]["kaldi_transcription"],
                )
            else:
                return mix, ref, noise
        else:
            if self.return_paths:
                return (
                    mix,
                    ref,
                    [mix_path, ref1_path, ref2_path],
                    self.mix_info[idx]["kaldi_transcription"],
                )
            else:
                return mix, ref

    def __len__(self):
        return len(self.mix_info)
