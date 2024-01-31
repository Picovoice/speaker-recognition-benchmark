import argparse
import json
import math
import tempfile
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
from pyannote.core import notebook
from pyannote.metrics.detection import (
    DetectionAccuracy,
    DetectionErrorRate,
)
from tqdm import tqdm

from dataset import *
from engine import *
from util import split_audio

RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


class BenchmarkTypes(Enum):
    ACCURACY = "ACCURACY"
    CPU = "CPU"


def _engine_params_parser(in_args: argparse.Namespace) -> Dict[str, Any]:
    kwargs_engine = dict()
    engine = Engines(in_args.engine)
    if engine is Engines.PICOVOICE_EAGLE:
        if in_args.picovoice_access_key is None:
            raise ValueError(f"Engine {in_args.engine} requires --picovoice-access-key")
        kwargs_engine.update(access_key=in_args.picovoice_access_key)
    elif engine in {Engines.PYANNOTE, Engines.WESPEAKER}:
        if in_args.auth_token is None:
            raise ValueError(f"Engine {in_args.engine} requires --auth-token")
        kwargs_engine.update(auth_token=in_args.auth_token)
    return kwargs_engine


def _process_accuracy(engine: Engine, dataset: Dataset, verbose: bool, plot: bool, name: str) -> None:
    metric_da = DetectionAccuracy(skip_overlap=True)
    metric_der = DetectionErrorRate(skip_overlap=True)
    metrics = [metric_da, metric_der]

    os.makedirs(os.path.join(RESULTS_FOLDER, str(dataset)), exist_ok=True)
    try:
        for index in tqdm(range(dataset.size)):
            audio_path, audio_length, ground_truth = dataset.get(index)
            if verbose:
                print(f"Processing {audio_path}...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                speakers = ground_truth.labels()
                for speaker in speakers:
                    print("Speaker:", speaker)
                    try:
                        enroll_audio, test_audio, test_ground_truth = split_audio(
                            path=audio_path,
                            label=ground_truth,
                            speaker=speaker,
                            folder=tmp_dir)
                    except ValueError as e:
                        print(f"Error: {e}")
                        continue
                    try:
                        profile = engine.enrollment(enroll_audio)
                        hypothesis, _ = engine.recognition(test_audio, profile)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                    if plot:
                        notebook.reset()
                        notebook.width = 10
                        notebook.crop = Segment(0, audio_length)
                        plt.rcParams["figure.figsize"] = (notebook.width, 3)

                        plt.subplot(211)
                        plt.title(os.path.basename(audio_path), loc="right")
                        notebook.plot_annotation(test_ground_truth, legend=True, time=True)
                        plt.gca().text(0.6, 0.15, "reference", fontsize=16)

                        plt.subplot(212)
                        notebook.plot_annotation(hypothesis, legend=True, time=True)
                        plt.gca().text(0.6, 0.15, "hypothesis", fontsize=16)

                        plt.show()

                    for metric in metrics:
                        res = metric(test_ground_truth, hypothesis, detailed=True)
                        if verbose:
                            print(f"{metric.name}: {res}")
    except KeyboardInterrupt:
        print("Stopping benchmark...")

    results = dict()
    for metric in metrics:
        results[metric.name] = abs(metric)
    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}{name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    results_details_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}{name}.log")
    with open(results_details_path, "w") as f:
        for metric in metrics:
            f.write(f"{metric.name}:\n{str(metric)}")
            f.write("\n")


WorkerResult = namedtuple(
    'WorkerResult',
    [
        'total_audio_sec',
        'process_time_sec',
    ])


def _process_worker(
        engine_type: str,
        engine_params: Dict[str, Any],
        enroll_audio_path: str,
        samples: Sequence[Tuple[str, float]]) -> WorkerResult:
    engine = Engine.create(Engines(engine_type), **engine_params)
    total_audio_sec = 0
    process_time = 0
    profile = engine.enrollment(enroll_audio_path)
    for sample in samples:
        print(sample)
        audio_path, audio_length = sample
        try:
            _, elapsed_time = engine.recognition(audio_path, profile)
            process_time += elapsed_time
            total_audio_sec += audio_length
        except Exception as e:
            print(f"Error: {e}")
            continue

    engine.cleanup()
    return WorkerResult(total_audio_sec, process_time)


def _process_cpu(
        engine: str,
        engine_params: Dict[str, Any],
        dataset: Dataset,
        num_samples: Optional[int] = None) -> None:
    num_workers = min(os.cpu_count(), 4)

    samples = list(dataset.samples[:])
    if num_samples is not None:
        samples = samples[:num_samples]

    enroll_audio_path, _, _ = dataset.sample_info(samples[0])

    chunk_size = math.floor(len(samples) / num_workers)
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            chunk = samples[i * chunk_size: (i + 1) * chunk_size]
            chunk_samples = list()
            for s in chunk:
                audio_path, audio_length, _ = dataset.sample_info(s)
                chunk_samples.append((audio_path, audio_length))

            future = executor.submit(
                _process_worker,
                engine_type=engine,
                engine_params=engine_params,
                enroll_audio_path=enroll_audio_path,
                samples=chunk_samples)
            futures.append(future)

    res = [f.result() for f in futures]
    total_audio_time_sec = sum([r.total_audio_sec for r in res])
    total_process_time_sec = sum([r.process_time_sec for r in res])

    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}_cpu.json")
    results = {
        "total_audio_time_sec": total_audio_time_sec,
        "total_process_time_sec": total_process_time_sec,
        "num_workers": num_workers,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--label-folder", required=True)
    parser.add_argument("--engine", choices=[en.value for en in Engines], required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--name", default='')
    parser.add_argument("--type", choices=[bt.value for bt in BenchmarkTypes], required=True)
    parser.add_argument("--picovoice-access-key")
    parser.add_argument("--auth-token")
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    engine_args = _engine_params_parser(args)

    dataset = Dataset.create(Datasets(args.dataset), data_folder=args.data_folder, label_folder=args.label_folder)
    print(f"Dataset: {dataset}")

    engine = Engine.create(Engines(args.engine), **engine_args)
    print(f"Engine: {engine}")

    if args.type == BenchmarkTypes.ACCURACY.value:
        _process_accuracy(engine, dataset, verbose=args.verbose, plot=args.plot, name=args.name)
    elif args.type == BenchmarkTypes.CPU.value:
        _process_cpu(
            engine=args.engine,
            engine_params=engine_args,
            dataset=dataset,
            num_samples=args.num_samples)


if __name__ == "__main__":
    main()

__all__ = [
    "RESULTS_FOLDER",
]
