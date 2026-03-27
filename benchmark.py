import argparse
import json
import os
from typing import (
    Any,
    Dict
)

import torch

from dataset import *
from engine import *
from metric import *

RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


def _engine_params_parser(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs_engine = dict()
    engine = Engines(args.engine)
    if engine is Engines.PICOVOICE_EAGLE:
        if args.picovoice_access_key is None:
            raise ValueError(f"Engine {args.engine} requires --picovoice-access-key")
        kwargs_engine.update(access_key=args.picovoice_access_key)
        kwargs_engine.update(device=args.picovoice_device)
    elif engine in {Engines.PYANNOTE}:
        if args.auth_token is None:
            raise ValueError(f"Engine {args.engine} requires --auth-token")
        kwargs_engine.update(auth_token=args.auth_token)
    return kwargs_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", choices=[x.value for x in Keywords], required=True)
    parser.add_argument("--engine", choices=[x.value for x in Engines], required=True)
    parser.add_argument("--metric", choices=[x.value for x in Metrics], default=Metrics.EER.value)
    parser.add_argument("--num-enrollments", type=int, default=3)
    parser.add_argument("--picovoice-access-key")
    parser.add_argument("--picovoice-device", default="cpu:1")
    parser.add_argument("--auth-token")
    args = parser.parse_args()

    keyword = Keywords(args.keyword)
    engine = Engines(args.engine)
    metric = Metrics(args.metric)
    num_enrollments = args.num_enrollments

    torch.set_num_threads(1)

    dataset = Dataset(keyword=keyword, num_enrollments=num_enrollments)
    print(dataset)

    engine_kwargs = _engine_params_parser(args)
    engine = Engine.create(engine=engine, **engine_kwargs)
    print(engine)

    metric = Metric.create(metric)

    profiles = list()
    total_enroll_process_time = 0.0
    total_enroll_audio_time = 0.0
    for i in range(dataset.num_speakers):
        profile, process_time = engine.enroll(dataset.enrollments(i))
        profiles.append(profile)
        total_enroll_process_time += process_time
        for enroll in dataset.enrollments(i):
            total_enroll_audio_time += len(enroll) / dataset.sample_rate

    positives = list()
    negatives = list()
    total_process_time = 0.0
    total_audio_time = 0.0
    for i in range(dataset.num_speakers):
        for inference in dataset.inferences(i):
            probs, process_time = engine.infer(pcm=inference, profiles=profiles)
            audio_time = len(inference) / dataset.sample_rate
            positives.append(probs[i])
            negatives.extend(probs[:i])
            negatives.extend(probs[i + 1:])
            total_process_time += process_time
            total_audio_time += audio_time

    eer = metric.compute(positives, negatives)
    rtf = total_process_time/total_audio_time

    print(f"{metric} {eer * 100.:.2f}%")
    print(f"🚀 RTF {rtf:.02f}")

    results_path = os.path.join(RESULTS_FOLDER, "data", args.keyword, f"{args.engine}.json")
    results = {
        args.metric: eer,
        "process_time": total_process_time,
        "audio_time": total_audio_time,
        "enroll_process_time": total_enroll_process_time,
        "enroll_audio_time": total_enroll_audio_time
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
