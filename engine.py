import os
import time
from enum import Enum
from typing import *

import pveagle
import soundfile as sf
import torch
import torchaudio
from pyannote.audio import Audio
from pyannote.audio import Inference
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Annotation
from pyannote.core import Segment
from scipy.spatial.distance import cdist
from speechbrain.pretrained import EncoderClassifier

Profile = Any

NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

AUDIO_FRAME_LENGTH_SEC = 0.096


class Engines(Enum):
    PICOVOICE_EAGLE = "PICOVOICE_EAGLE"
    PYANNOTE = "PYANNOTE"
    SPEECHBRAIN = "SPEECHBRAIN"
    WESPEAKER = "WESPEAKER"


class Engine:
    def enrollment(self, path: str) -> Profile:
        raise NotImplementedError()

    def recognition(self, path: str, profile: Profile) -> Tuple[Annotation, float]:
        raise NotImplementedError()

    def cleanup(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Engines, **kwargs: Any) -> "Engine":
        try:
            subclass = {
                Engines.PICOVOICE_EAGLE: PicovoiceEagleEngine,
                Engines.PYANNOTE: PyAnnoteEngine,
                Engines.SPEECHBRAIN: SpeechBrainEngine,
                Engines.WESPEAKER: WeSpeakerEngine,
            }[x]
        except KeyError:
            raise ValueError(f"cannot create `{cls.__name__}` of type `{x.value}`")
        return subclass(**kwargs)


class PicovoiceEagleEngine(Engine):
    def __init__(
            self,
            access_key: str,
            detection_threshold: float = 0.5) -> None:

        self._access_key = access_key
        self._detection_threshold = detection_threshold
        self._eagle = None
        super().__init__()

    def enrollment(self, path: str) -> Profile:
        eagle_profiler = pveagle.create_profiler(
            access_key=self._access_key)

        data, sample_rate = sf.read(path, dtype="int16")
        assert sample_rate == eagle_profiler.sample_rate
        enroll_percentage, feedback = eagle_profiler.enroll(data)

        if enroll_percentage < 100.0:
            raise ValueError(f"failed to create speaker profile for `{path}` with {enroll_percentage}% enrollment")

        profile = eagle_profiler.export()
        eagle_profiler.delete()

        return profile

    def recognition(self, path: str, profile: Profile) -> Tuple[Annotation, float]:
        eagle = pveagle.create_recognizer(
            access_key=self._access_key,
            speaker_profiles=[profile])

        total_time = 0
        data, sample_rate = sf.read(path, dtype="int16")
        num_frames = len(data) // eagle.frame_length
        frame_to_second = eagle.frame_length / eagle.sample_rate
        step_frames = int(AUDIO_FRAME_LENGTH_SEC / frame_to_second)
        assert step_frames > 0
        scores_list = list()
        start_time = 0
        score_max = 0
        tic = time.perf_counter()
        for i in range(num_frames):
            frame = data[i * eagle.frame_length:(i + 1) * eagle.frame_length]
            scores = eagle.process(frame)
            score_max = max(score_max, scores[0])
            if (i + 1) % step_frames == 0:
                end_time = (i + 1) * frame_to_second
                scores_list.append((start_time, end_time, score_max))
                start_time = end_time
                score_max = 0
        total_time += time.perf_counter() - tic

        annotation = self._scores_to_annotation(scores_list, self._detection_threshold)

        eagle.delete()

        return annotation, total_time

    @staticmethod
    def _scores_to_annotation(scores_list: List[Tuple[float, float, float]], threshold: float) -> Annotation:
        annotation = Annotation()
        for start, end, scores in scores_list:
            if scores > threshold:
                annotation[Segment(start, end)] = 0

        return annotation

    def cleanup(self) -> None:
        pass

    def __str__(self):
        return Engines.PICOVOICE_EAGLE.value


class PyAnnoteBaseEngine(Engine):
    def __init__(
            self,
            auth_token: str,
            use_gpu: bool,
            model: str,
            detection_threshold: float = 0.5) -> None:
        if use_gpu and torch.cuda.is_available():
            torch_device = torch.device("cuda:1")
        else:
            torch_device = torch.device("cpu")
        self._model = PretrainedSpeakerEmbedding(
            embedding=model,
            device=torch_device,
            use_auth_token=auth_token)
        self._audio = Audio(sample_rate=16000)
        self._inference = Inference(model, window="sliding", step=AUDIO_FRAME_LENGTH_SEC).to(torch_device)
        self._detection_threshold = detection_threshold
        super().__init__()

    def enrollment(self, path: str) -> Profile:
        waveform1, sample_rate = self._audio(path)
        profile = self._model(waveform1[None])
        return profile

    def recognition(self, path: str, profile: Profile) -> Tuple[Annotation, float]:
        tic = time.perf_counter()
        embeddings = self._inference(path)
        total_time = time.perf_counter() - tic
        distance_list = list()
        for embedding in embeddings:
            segment, emb = embedding
            distance = cdist(profile, emb.reshape(1, -1), metric="cosine")
            distance_list.append((segment.start, segment.end, distance[0, 0]))

        annotation = self._distance_to_annotation(distance_list, threshold=self._detection_threshold)

        return annotation, total_time

    @staticmethod
    def _distance_to_annotation(distance_list: List[Tuple[Any, Any, Any]], threshold: float) -> Annotation:
        annotation = Annotation()
        for start, end, distance in distance_list:
            if distance < threshold:
                annotation[Segment(start, end)] = 0

        return annotation.support(0.1)

    def cleanup(self) -> None:
        self._model = None


class PyAnnoteEngine(PyAnnoteBaseEngine):
    def __init__(self, auth_token: str, use_gpu: bool = False) -> None:
        super().__init__(
            auth_token=auth_token,
            use_gpu=use_gpu,
            model="pyannote/embedding")

    def __str__(self) -> str:
        return Engines.PYANNOTE.value


class WeSpeakerEngine(PyAnnoteBaseEngine):
    def __init__(self, auth_token: str = '', use_gpu: bool = False) -> None:
        super().__init__(
            auth_token=auth_token,
            use_gpu=use_gpu,
            model="pyannote/wespeaker-voxceleb-resnet34-LM")

    def __str__(self) -> str:
        return Engines.WESPEAKER.value


class SpeechBrainEngine(Engine):
    def __init__(self, use_gpu: bool = False) -> None:
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self._torch_device = torch.device(device)
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device})
        self._step = AUDIO_FRAME_LENGTH_SEC
        self._audio = Audio(sample_rate=16000, mono="downmix")
        self._detection_threshold = 0.5
        self._duration = 3
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        super().__init__()

    def enrollment(self, path: str) -> Profile:
        waveform, _ = torchaudio.load(path)
        waveform = waveform.to(self._torch_device)
        profile = self._model.encode_batch(waveform)
        return profile

    def recognition(self, path: str, profile: Profile) -> Tuple[Annotation, float]:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(self._torch_device)
        audio_length = waveform.shape[1] / sample_rate
        start_time = 0
        total_time = 0
        score_list = list()
        while start_time + self._duration < audio_length:
            end_time = start_time + self._duration
            waveform_chunk = waveform[0, int(start_time * sample_rate):int(end_time * sample_rate)]
            tic = time.perf_counter()
            embedding = self._model.encode_batch(waveform_chunk)
            total_time += time.perf_counter() - tic
            score = self.similarity(profile, embedding)
            if start_time == 0:
                score_list.append((start_time, end_time, score[0][0].tolist()))
            else:
                score_list.append((end_time - self._step, end_time, score[0][0].tolist()))
            start_time = start_time + self._step

        annotation = self._score_to_annotation(score_list, threshold=self._detection_threshold)

        return annotation, total_time

    @staticmethod
    def _score_to_annotation(score_list: List[Tuple[float, float, float]], threshold: float) -> Annotation:
        annotation = Annotation()
        for start, end, score in score_list:
            if score > threshold:
                annotation[Segment(start, end)] = 0

        return annotation

    def cleanup(self) -> None:
        pass

    def __str__(self) -> str:
        return Engines.SPEECHBRAIN.value
