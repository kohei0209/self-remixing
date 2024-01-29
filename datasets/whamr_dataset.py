import json
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils import data

DATASET = "WHAMR"

# WHAMR tasks
# Many tasks can be considered with this dataset, we only consider the 4 core
# separation tasks presented in the paper for now.
sep_clean = {
    "mixture": "mix_clean_anechoic",
    "sources": ["s1_anechoic", "s2_anechoic"],
    "infos": [],
    "default_nsrc": 2,
}
sep_noisy = {
    "mixture": "mix_both_anechoic",
    "sources": ["s1_anechoic", "s2_anechoic"],
    "infos": ["noise"],
    "default_nsrc": 2,
}
sep_reverb = {
    "mixture": "mix_clean_reverb",
    # "sources": ["s1_anechoic", "s2_anechoic"],
    "sources": ["s1_reverb", "s2_reverb"],
    "infos": [],
    "default_nsrc": 2,
}
sep_reverb_noisy = {
    "mixture": "mix_both_reverb",
    # "sources": ["s1_anechoic", "s2_anechoic"],
    "sources": ["s1_reverb", "s2_reverb"],
    "infos": ["noise"],
    "default_nsrc": 2,
}
enh_reverb = {
    "mixture": "mix_single_reverb",
    # "sources": ["s1_anechoic", "s2_anechoic"],
    "sources": ["s1_reverb"],
    "infos": ["noise"],
    "default_nsrc": 1,
}
enh_anechoic = {
    "mixture": "mix_single_anechoic",
    # "sources": ["s1_anechoic", "s2_anechoic"],
    "sources": ["s1_anechoic"],
    "infos": ["noise"],
    "default_nsrc": 1,
}

WHAMR_TASKS = {
    "sep_clean": sep_clean,
    "sep_noisy": sep_noisy,
    "sep_reverb": sep_reverb,
    "sep_reverb_noisy": sep_reverb_noisy,
    "enh_anechoic": enh_anechoic,
    "enh_reverb": enh_reverb,
}
# Support both order, confusion is easy
WHAMR_TASKS["sep_noisy_reverb"] = WHAMR_TASKS["sep_reverb_noisy"]

stages = {
    "train-100": "tr",
    "train-360": "tr",
    "train": "tr",
    "valid": "cv",
    "test": "tt",
}


class WhamRDataset(data.Dataset):
    """Dataset class for WHAMR source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'sep_clean'``, ``'sep_noisy'``, ``'sep_reverb'``
            or ``'sep_reverb_noisy'``.

            * ``'sep_clean'`` for two-speaker clean (anechoic) source
              separation.
            * ``'sep_noisy'`` for two-speaker noisy (anechoic) source
              separation.
            * ``'sep_reverb'`` for two-speaker clean reverberant
              source separation.
            * ``'sep_reverb_noisy'`` for two-speaker noisy reverberant source
              separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.

    References
        "WHAMR!: Noisy and Reverberant Single-Channel Speech Separation", Maciejewski et al. 2020
    """

    dataset_name = "WHAMR"

    def __init__(
        self,
        json_dir,
        stage,
        task="sep_reverb_noisy",
        min_or_max="min",
        sample_rate=8000,
        segment=4.0,
        nondefault_nsrc=None,
        normalization=False,
        return_noise=False,
        num_data=None,
    ):
        super(WhamRDataset, self).__init__()
        if task not in WHAMR_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of "
                "{}".format(task, WHAMR_TASKS.keys())
            )
        # Task setting
        assert min_or_max in ["min", "max"]
        json_dir = json_dir / f"wav{sample_rate//1000}k" / min_or_max / stages[stage]
        self.task = task
        self.task_dict = WHAMR_TASKS[task]
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.normalization = normalization

        if task in ["sep_clean", "sep_reverb"]:
            assert return_noise is False
        self.return_noise = return_noise
        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
        sources_json = [
            os.path.join(json_dir, source + ".json")
            for source in self.task_dict["sources"]
        ]
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

        if num_data is not None:
            print(f"{stage}: use only {num_data} data")
            self.mix = self.mix[:num_data]
            for s in range(len(self.sources)):
                self.sources[s] = self.sources[s][:num_data]

    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError(
                "Only datasets having the same number of sources"
                "can be added together. Received "
                "{} and {}".format(self.n_src, wham.n_src)
            )
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print(
                "Segment length mismatched between the two Dataset"
                "passed one the smallest to the sum."
            )
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        if (
            self.seg_len is not None and self.mix[idx][1] <= self.seg_len
        ) or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test or (
            self.seg_len is not None and self.mix[idx][1] < self.seg_len
        ):
            stop = None
        else:
            stop = rand_start + self.seg_len

        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len,))
            else:
                s, _ = sf.read(
                    src[idx][0], start=rand_start, stop=stop, dtype="float32"
                )
            source_arrays.append(s)

        if not self.like_test and self.mix[idx][1] < self.seg_len:
            if x.ndim == 2:
                n_chan = x.shape[-1]
                s1 = np.random.randint(0, self.seg_len - self.mix[idx][1])
                s2 = int(self.seg_len - self.mix[idx][1] - s1)
                x = np.concatenate(
                    (
                        np.zeros((s1, n_chan), dtype=np.float32),
                        x,
                        np.zeros((s2, n_chan), dtype=np.float32),
                    ),
                    axis=0,
                )
                for i in range(len(source_arrays)):
                    source_arrays[i] = np.concatenate(
                        (
                            np.zeros((s1, n_chan), dtype=np.float32),
                            source_arrays[i],
                            np.zeros((s2, n_chan), dtype=np.float32),
                        ),
                        axis=0,
                    )
            else:
                s1 = np.random.randint(0, self.seg_len - self.mix[idx][1])
                s2 = int(self.seg_len - self.mix[idx][1] - s1)
                x = np.concatenate(
                    (
                        np.zeros((s1), dtype=np.float32),
                        x,
                        np.zeros((s2), dtype=np.float32),
                    ),
                    axis=0,
                )
                for i in range(len(source_arrays)):
                    source_arrays[i] = np.concatenate(
                        (
                            np.zeros((s1), dtype=np.float32),
                            source_arrays[i],
                            np.zeros((s2), dtype=np.float32),
                        ),
                        axis=0,
                    )

        sources = np.stack(source_arrays, axis=0)

        x = torch.from_numpy(x)
        sources = torch.from_numpy(sources)

        # standardization
        if self.normalization:
            # devide by standard deviation
            std = torch.std(x, dim=-1, keepdim=True)
            x /= std
            sources /= std[..., None, :]

        if self.return_noise:
            noise = x - sources.sum(dim=-2)
            return x, sources, noise
        else:
            return x, sources

        # x: (n_chan, n_samples),  sources: (n_src, n_chan, n_samples)
        # return torch.from_numpy(x).transpose(-1,-2), torch.from_numpy(sources).transpose(-1,-2)

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [wsj0_license]
        else:
            data_license = [wsj0_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos


wham_noise_license = dict(
    title="The WSJ0 Hipster Ambient Mixtures dataset",
    title_link="http://wham.whisper.ai/",
    author="Whisper.ai",
    author_link="https://whisper.ai/",
    license="CC BY-NC 4.0",
    license_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=True,
)

wsj0_license = dict(
    title="CSR-I (WSJ0) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC93S6A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)
