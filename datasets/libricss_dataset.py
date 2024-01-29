import json
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .utils import zeropad_sources

session_numbers = {
    "train": [2, 3, 4, 5, 6, 7, 8, 9],
    "valid": [1],
    "test": list(range(0, 10)),
}


class LibriCSSDataset(Dataset):
    def __init__(
        self,
        json_path,
        stage,
        num_data=None,
        audio_len=6,
        return_trans=True,
        return_data=False,
        sample_rate=16000,
        max_audio_len=None,
        normalization=False,
    ):
        # load mixinfo
        with open(json_path / "all_info_monoaural.json") as f:
            all_info_dict = json.load(f)
        # prepare mixinfo for training and validation
        self.mix_info = []
        sessions = [("session" + str(s)) for s in session_numbers[stage]]
        for session_id, info in all_info_dict.items():
            for utter, utter_info in info.items():
                if session_id.split("_")[5] in sessions:
                    self.mix_info.append(utter_info)
        if num_data is not None:
            print(f"{stage.upper()}: Use only {num_data} data !!!")
            self.mix_info = self.mix_info[:num_data]

        self.fs = sample_rate
        self.audio_length = audio_len * self.fs
        self.training = stage == "train"
        self.return_trans = return_trans
        self.return_data = return_data
        self.normalization = normalization

        # return_trans is for validation, return_data is for test
        assert return_trans != return_data

        if not self.training and max_audio_len is not None:
            # remove long audio to avoid out of memory
            print(f"\nRemove data longer than {max_audio_len}[s]")
            del_list = []
            for i in range(len(self.mix_info)):
                if self.mix_info[i]["num_samples"] // sample_rate > max_audio_len:
                    del_list.append(i)
                    print(i, self.mix_info[i]["num_samples"])
            print(f"Before filtering: {len(self.mix_info)}")
            for i, d in enumerate(del_list):
                del self.mix_info[d - i]
            print(f"After filtering: {len(self.mix_info)}")
            # sort in ascending order
            self.mix_info = sorted(
                self.mix_info, key=lambda x: x["num_samples"], reverse=True
            )
            # for debugging
            for i in range(len(self.mix_info)):
                if self.mix_info[i]["num_samples"] // sample_rate > max_audio_len:
                    print(i, self.mix_info[i]["num_samples"])

    def __getitem__(self, idx):
        data = self.mix_info[idx]
        mix_path = data["path"]

        num_samples = data["num_samples"]
        diff = num_samples - self.audio_length
        if self.training and diff > 0:
            start = random.randint(0, diff - 1)
            stop = start + self.audio_length
        else:
            start = 0
            stop = num_samples

        mix, fs = sf.read(
            str(mix_path),
            start=start,
            stop=stop,
            dtype="float32",
        )

        assert self.fs == fs

        # numpy to torch
        mix = torch.from_numpy(mix)

        # preprocessing
        if self.normalization:
            std = torch.std(mix, dim=-1, keepdim=True) + 1e-8
            mix /= std

        # zero-padding if the mixture is shorter than specified audio_length
        if self.training and diff < 0:
            span = random.randint(0, self.audio_length - mix.shape[-1] - 1)
            mix = zeropad_sources(mix, self.audio_length, span)
        if self.return_trans:
            return mix, data["transcription"]
        elif self.return_data:
            return mix, data
        else:
            return mix

    def __len__(self):
        return len(self.mix_info)


class Collator_LibriCSS:
    def __init__(self):
        pass

    def __call__(self, batch_list):
        max_len = max([s[0].shape[-1] for s in batch_list])
        n_batch = len(batch_list)

        mix = batch_list[0][0].new_zeros((n_batch, max_len))
        trans = []
        offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

        for b, ((m, t), o) in enumerate(zip(batch_list, offsets)):
            mix[b, o : o + m.shape[-1]] = m
            trans.append(t)

        return mix, trans
