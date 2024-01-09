import os
from collections import defaultdict
from typing import Sequence, Dict, Tuple, Hashable

import numpy as np
import soundfile as sf
from pyannote.core import Segment, Annotation

RTTM = Dict[str, Sequence[Tuple[str, float, float]]]


def load_rttm(file: str) -> RTTM:
    rttm = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            spk = parts[7]
            start = float(parts[3])
            end = start + float(parts[4])
            rttm[file_id].append((spk, start, end))
    return rttm


def rttm_to_annotation(rttm: RTTM) -> "Annotation":
    reference = Annotation()
    segments = list(rttm.values())[0]
    for segment in segments:
        label, start, end = segment
        reference[Segment(start, end)] = label
    return reference


def get_audio_length(file: str) -> float:
    data, samplerate = sf.read(file)
    return len(data) / samplerate


def split_audio(
        path: str,
        label: Annotation,
        speaker: Hashable,
        folder: str,
        enrolled_length_seconds: int = 15,
        min_total_length_seconds: int = 30) -> Tuple[str, str, Annotation]:

    detection_label = label.subset([speaker])
    detection_label.uri = f"{os.path.basename(path)}-{speaker}"
    if detection_label.get_timeline().duration() < min_total_length_seconds:
        raise ValueError(f"not enough audio for testing `{path}`")

    data, sample_rate = sf.read(path, dtype="int16")
    assert sample_rate == 16000

    segments = label.label_timeline(speaker)
    overlap = segments.get_overlap()
    segments.extrude(overlap)

    enroll_data = list()
    test_data = data

    enrolled_sec = 0
    for segment in segments:
        start = int(segment.start * 16000)
        end = int(segment.end * 16000)
        enroll_data.append(data[start:end])
        enrolled_sec += segment.duration
        if enrolled_sec >= enrolled_length_seconds:
            break

    enroll_file = os.path.join(folder, "enroll.wav")
    test_file = os.path.join(folder, "test.wav")

    sf.write(enroll_file, np.concatenate(enroll_data), 16000)
    sf.write(test_file, test_data, 16000)

    return enroll_file, test_file, detection_label


__all__ = [
    "RTTM",
    "load_rttm",
    "rttm_to_annotation",
    "get_audio_length",
    "split_audio",
]
