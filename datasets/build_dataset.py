from pathlib import Path
from typing import Dict

from .fuss_dataset import FUSSDataset
from .libricss_dataset import LibriCSSDataset
from .librimix_dataset import LibriMixDataset
from .smswsj_dataset import SMSWSJDataset
from .wsjmix_dataset import Collator_WSJMix, WSJMixDataset

datasets = {
    "wsjmix": WSJMixDataset,
    "smswsj": SMSWSJDataset,
    "librimix": LibriMixDataset,
    "libricss": LibriCSSDataset,
    "fuss": FUSSDataset,
}

collators = {
    "wsjmix": Collator_WSJMix,
    "smswsj": Collator_WSJMix,
    "librimix": Collator_WSJMix,
    "libricss": None,
    "fuss": None,
}


def call_dataset(
    dataset_name: str,
    data_path: Path,
    stage: str,
    num_data: int,
    kwargs: Dict,
):
    assert dataset_name in datasets, f"dataset name must be in {datasets.keys()}"

    return datasets[dataset_name](
        data_path,
        stage,
        num_data=num_data,
        **kwargs,
    )


def call_collate_fn(
    dataset_name: str,
    kwargs: Dict = {},
):
    assert dataset_name in collators, f"dataset name must be in {datasets.keys()}"

    if collators[dataset_name] is None:
        return None
    else:
        return collators[dataset_name](**kwargs)
