import time
from enum import Enum
from typing import (
    Any,
    Sequence,
    Tuple
)

import numpy as np
import torch


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

        return profile

    def infer(self, pcm: Sequence[int], profiles: Sequence[Any]) -> Tuple[Sequence[float], float, float]:
        start_time = time.perf_counter()
        res = self._eagle.process(pcm, speaker_profiles=profiles)
        end_time = time.perf_counter()
        if res is None:
            raise RuntimeError()
        
        process_time = end_time - start_time
        audio_time = len(pcm) / self._eagle.sample_rate

        return res, process_time, audio_time

    def __str__(self) -> str:
        return f"🤖[{Engines.PICOVOICE_EAGLE.value}]"


class PyannoteEngine(Engine):
    def __init__(self, auth_token: str) -> None:
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        self._model = PretrainedSpeakerEmbedding(
            embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
            token=auth_token)

    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Sequence[float]:
        waveform = \
            np.concatenate([np.asarray(x, dtype=np.int16) for x in enrollments], axis=0).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).view(1, 1, -1)

        with torch.no_grad():
            embedding = self._model(waveform)

        return embedding.flatten().tolist()

    def infer(self, pcm: Sequence[int], profiles: Sequence[Sequence[float]]) -> Sequence[float]:
        waveform = np.asarray(pcm, dtype=np.int16).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).view(1, 1, -1)

        with torch.no_grad():
            start_time = time.perf_counter()
            embedding = self._model(waveform).flatten()
            embedding = embedding / np.linalg.norm(embedding)

            profile_tensor = np.asarray(profiles, dtype=np.float32)
            profile_tensor = profile_tensor / np.linalg.norm(profile_tensor, axis=1, keepdims=True)

            scores = profile_tensor @ embedding
            end_time = time.perf_counter()

        return scores.tolist(), end_time - start_time, len(pcm) / 16000

    def __str__(self) -> str:
        return f"🤖[{Engines.PYANNOTE.value}]"


class SpeechBrainEngine(Engine):
    def __init__(self) -> None:
        from speechbrain.inference.classifiers import EncoderClassifier
        self._model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def enroll(self, enrollments: Sequence[Sequence[int]]) -> Sequence[float]:
        waveform = \
            np.concatenate([np.asarray(x, dtype=np.int16) for x in enrollments], axis=0).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.encode_batch(waveform, normalize=False)

        embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
        return embedding.tolist()

    def infer(self, pcm: Sequence[int], profiles: Sequence[Sequence[float]]) -> Sequence[float]:
        waveform = np.asarray(pcm, dtype=np.int16).astype(np.single) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        with torch.no_grad():
            start_time = time.perf_counter()
            embedding = self._model.encode_batch(waveform, normalize=False)
            end_time = time.perf_counter()

        embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
        embedding = embedding / np.clip(np.linalg.norm(embedding), 1e-12, None)

        profile_tensor = np.asarray(profiles, dtype=np.float32)
        profile_tensor = profile_tensor / np.clip(
            np.linalg.norm(profile_tensor, axis=1, keepdims=True),
            1e-12,
            None,
        )

        scores = profile_tensor @ embedding
        return scores.tolist(), end_time - start_time, len(pcm) / 16000

    def __str__(self) -> str:
        return f"🤖[{Engines.SPEECHBRAIN.value}]"


__all__ = [
    "Engine",
    "Engines"
]
