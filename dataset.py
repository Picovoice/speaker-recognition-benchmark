import glob
import os.path
from enum import Enum
from typing import *

from pyannote.core import Annotation

from util import load_rttm, rttm_to_annotation, get_audio_length

Sample = Tuple[str, float, str]


class Datasets(Enum):
    VOX_CONVERSE = "VoxConverse"


class Dataset:
    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def samples(self) -> Sequence[Sample]:
        raise NotImplementedError()

    def get(self, index: int) -> Tuple[str, float, Annotation]:
        raise NotImplementedError()

    @staticmethod
    def sample_info(sample: Sample) -> Tuple[str, float, Annotation]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Datasets, data_folder: str, label_folder: str, **kwargs: Any) -> "Dataset":
        try:
            subclass = {
                Datasets.VOX_CONVERSE: VoxConverse,
            }[x]
        except KeyError:
            raise ValueError(f"cannot create `{cls.__name__}` of type `{x.value}`")
        return subclass(data_folder, label_folder, **kwargs)


class VoxConverse(Dataset):
    def __init__(self, data_folder: str, label_folder: str) -> None:
        self._samples = list()

        files = glob.iglob(os.path.join(data_folder, "*.wav"))
        for file in files:
            name = os.path.basename(file)
            label_path = os.path.join(label_folder, name.replace(".wav", ".rttm"))
            if not os.path.exists(label_path):
                raise ValueError(f"cannot find label file `{label_path}`")
            audio_length = get_audio_length(file)
            self._samples.append((file, audio_length, label_path))

    @property
    def size(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> Sequence[Sample]:
        return self._samples

    def get(self, index: int) -> Tuple[str, float, Annotation]:
        sample = self._samples[index]
        return self.sample_info(sample)

    @staticmethod
    def sample_info(sample: Sample) -> Tuple[str, float, Annotation]:
        audio_path, audio_length, label_path = sample
        rttm = load_rttm(label_path)
        label = rttm_to_annotation(rttm)
        label.uri = os.path.basename(audio_path)
        return audio_path, audio_length, label

    def __str__(self) -> str:
        return "VoxConverse"


__all__ = [
    "Datasets",
    "Dataset",
    "Sample"
]
