import time
from enum import Enum
from typing import (
    Any,
    Sequence,
    Tuple
)

import numpy as np
import torch
from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding
)
from speechbrain.inference.classifiers import EncoderClassifier


class Engines(Enum):
    PICOVOICE_EAGLE = "eagle"
    PYANNOTE = "pyannote"
    SPEECHBRAIN = "speechbrain"


class Engine(object):
    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Sequence[float]:
        raise NotImplementedError

    def infer(self, pcm: Sequence[int], profiles: Sequence[Sequence[float]]) -> Sequence[float]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def create(cls, engine: Engines, **kwargs: Any) -> "Engine":
        children = {
            Engines.PICOVOICE_EAGLE: EagleEngine,
            Engines.PYANNOTE: PyannoteEngine,
            Engines.SPEECHBRAIN: SpeechBrainEngine,
        }

        if engine not in children:
            raise ValueError(f"Cannot create `{cls.__name__}` of type `{engine.value}`")

        return children[engine](**kwargs)


class EagleEngine(Engine):
    def __init__(self, access_key: str, device: str, voice_threshold: float = .15) -> None:
        import pveagle

        self._profiler = pveagle.create_profiler(
            access_key=access_key,
            device=device,
            min_enrollment_chunks=3,
            voice_threshold=voice_threshold)
        self._eagle = pveagle.create_recognizer(
            access_key=access_key,
            device=device,
            voice_threshold=voice_threshold)

    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Any:
        start_time = time.perf_counter()

        frame_length = self._profiler.frame_length

        progress = 0.
        for enrollment in enrollments:
            for i in range(len(enrollment) // self._profiler.frame_length):
                self._profiler.enroll(enrollment[i * frame_length:(i + 1) * frame_length])
            progress = self._profiler.flush()
        if progress < 100.:
            raise RuntimeError()

        profile = self._profiler.export()
        self._profiler.reset()

        end_time = time.perf_counter()
        return profile, end_time - start_time

    def infer(self, pcm: Sequence[int], profiles: Sequence[Any]) -> Tuple[Sequence[float], float]:
        start_time = time.perf_counter()
        res = self._eagle.process(pcm, speaker_profiles=profiles)

        if res is None:
            raise RuntimeError()

        end_time = time.perf_counter()
        return res, end_time - start_time

    def __str__(self) -> str:
        return f"🤖[{Engines.PICOVOICE_EAGLE.value}]"


class PyannoteEngine(Engine):
    def __init__(self, auth_token: str) -> None:
        self._model = PretrainedSpeakerEmbedding(
            embedding="pyannote/embedding",
            token=auth_token)

    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Sequence[float]:
        start_time = time.perf_counter()

        waveform = \
            np.concatenate([np.asarray(x, dtype=np.int16) for x in enrollments], axis=0).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model(waveform).squeeze(0)

        end_time = time.perf_counter()
        return embedding.tolist(), end_time - start_time

    def infer(self, pcm: Sequence[int], profiles: Sequence[Sequence[float]]) -> Sequence[float]:
        start_time = time.perf_counter()

        waveform = np.asarray(pcm, dtype=np.int16).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model(waveform).squeeze(0)

            embedding = embedding / np.linalg.norm(embedding)

            profile_tensor = np.asarray(profiles, dtype=np.float32)
            profile_tensor = profile_tensor / np.linalg.norm(profile_tensor, axis=1, keepdims=True)

            scores = profile_tensor @ embedding

        end_time = time.perf_counter()
        return scores.tolist(), end_time - start_time

    def __str__(self) -> str:
        return f"🤖[{Engines.PYANNOTE.value}]"


class SpeechBrainEngine(Engine):
    def __init__(self) -> None:
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"})

    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Sequence[float]:
        start_time = time.perf_counter()

        waveform = \
            np.concatenate([np.asarray(x, dtype=np.int16) for x in enrollments], axis=0).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.encode_batch(waveform, normalize=False)

        embedding = embedding.squeeze().cpu().numpy().astype(np.float32)

        end_time = time.perf_counter()
        return embedding.tolist(), end_time - start_time

    def infer(self, pcm: Sequence[int], profiles: Sequence[Sequence[float]]) -> Sequence[float]:
        start_time = time.perf_counter()

        waveform = np.asarray(pcm, dtype=np.int16).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.encode_batch(waveform, normalize=False)

        embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
        embedding = embedding / np.clip(np.linalg.norm(embedding), 1e-12, None)

        profile_tensor = np.asarray(profiles, dtype=np.float32)
        profile_tensor = profile_tensor / np.clip(
            np.linalg.norm(profile_tensor, axis=1, keepdims=True),
            1e-12,
            None,
        )

        scores = profile_tensor @ embedding

        end_time = time.perf_counter()
        return scores.tolist(), end_time - start_time

    def __str__(self) -> str:
        return f"🤖[{Engines.SPEECHBRAIN.value}]"


__all__ = [
    "Engine",
    "Engines"
]
