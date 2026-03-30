import os
from enum import Enum
from random import Random
from typing import (
    Optional,
    Sequence
)

import soundfile


class Keywords(Enum):
    ALEXA = 'alexa'
    COMPUTER = 'computer'
    JARVIS = 'jarvis'
    SMART_MIRROR = 'smart mirror'
    SNOWBOY = "snowboy"
    VIEW_GLASS = "view glass"


class Dataset(object):
    def __init__(
            self,
            keyword: Keywords,
            num_enrollments: int,
            seed: Optional[int] = 18670701, # 🇨🇦
    ) -> None:
        self._keyword = keyword
        self._enrollments = dict()
        self._inferences = dict()

        r = Random(seed)

        folder = os.path.join(os.path.dirname(__file__), "audio", keyword.value)

        for speaker in sorted(os.listdir(folder), key=lambda x: int(x.split('_')[1])):
            speaker_folder = os.path.join(folder, speaker)
            utterances = list(sorted(os.listdir(speaker_folder)))
            if len(utterances) > num_enrollments:
                r.shuffle(utterances)
                audios = [soundfile.read(os.path.join(speaker_folder, x), dtype='int16')[0] for x in utterances]
                self._enrollments[len(self._enrollments)] = audios[:num_enrollments]
                self._inferences[len(self._inferences)] = audios[num_enrollments:]

    def enrollments(self, speaker: int) -> Sequence[Sequence[int]]:
        return self._enrollments[speaker]

    def inferences(self, speaker: int) -> Sequence[Sequence[int]]:
        return self._inferences[speaker]

    @property
    def num_speakers(self) -> int:
        return len(self._enrollments)

    @property
    def sample_rate(self) -> int:
        return 16000

    def __str__(self) -> str:
        return f"""💬 {{
  keyword: {self._keyword.value},
  num-speakers: {self.num_speakers},
  num-enrollments: {sum(len(x) for x in self._enrollments.values())}
  num-inferences: {sum(len(x) for x in self._inferences.values())}
}}"""


__all__ = [
    "Dataset",
    "Keywords",
]
