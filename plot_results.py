import argparse
import json
import os
from typing import *

import matplotlib.pyplot as plt

from benchmark import RESULTS_FOLDER
from dataset import Datasets
from engine import Engines

Color = Tuple[float, float, float]


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex("#000000")
GREY1 = rgb_from_hex("#3F3F3F")
GREY2 = rgb_from_hex("#5F5F5F")
GREY3 = rgb_from_hex("#7F7F7F")
GREY4 = rgb_from_hex("#9F9F9F")
GREY5 = rgb_from_hex("#BFBFBF")
WHITE = rgb_from_hex("#FFFFFF")
BLUE = rgb_from_hex("#377DFF")

ENGINES = [
    Engines.PICOVOICE_EAGLE,
    Engines.PYANNOTE,
    Engines.SPEECHBRAIN,
    Engines.WESPEAKER,
]

ENGINE_ORDER_KEYS = {
    Engines.PICOVOICE_EAGLE: 1,
    Engines.PYANNOTE: 3,
    Engines.SPEECHBRAIN: 2,
    Engines.WESPEAKER: 4,
}

ENGINE_COLORS = {
    Engines.PICOVOICE_EAGLE: BLUE,
    Engines.PYANNOTE: GREY2,
    Engines.SPEECHBRAIN: GREY1,
    Engines.WESPEAKER: GREY3,
}

ENGINE_PRINT_NAMES = {
    Engines.PICOVOICE_EAGLE: "Picovoice\nEagle",
    Engines.PYANNOTE: "pyannote",
    Engines.SPEECHBRAIN: "SpeechBrain",
    Engines.WESPEAKER: "WeSpeaker",
}

METRIC_NAME = [
    "detection error rate",
    "detection accuracy",
]


def _plot_accuracy(
        engine_list: List[Engines],
        result_path: str,
        save_path: str,
        show: bool) -> None:
    for metric in METRIC_NAME:
        max_value = 0
        min_value = 100
        fig, ax = plt.subplots(figsize=(6, 4))
        for engine_type in engine_list:
            engine_result_path = os.path.join(result_path, f"{engine_type.value}.json")
            if not os.path.exists(engine_result_path):
                continue

            with open(engine_result_path, "r") as f:
                results_json = json.load(f)

            engine_value = results_json[metric] * 100
            engine_value = round(engine_value, 1)
            max_value = max(max_value, engine_value)
            min_value = min(min_value, engine_value)
            ax.bar(
                ENGINE_PRINT_NAMES[engine_type],
                engine_value,
                width=0.5,
                color=ENGINE_COLORS[engine_type],
                edgecolor="none",
                label=ENGINE_PRINT_NAMES[engine_type]
            )
            ax.text(
                ENGINE_PRINT_NAMES[engine_type],
                engine_value + 1,
                f"{engine_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                color=ENGINE_COLORS[engine_type],
            )

        more_info = ""
        if metric in ["detection error rate"]:
            more_info = " (lower is better)"
        elif metric in ["detection accuracy"]:
            more_info = " (higher is better)"
        plt.ylim([min_value - 5, max_value + 5])
        ax.set_ylabel(f"{metric.title()} {more_info}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        plot_path = os.path.join(save_path, metric.replace(" ", "_") + ".png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")

        if show:
            plt.show()

        plt.close()


def _plot_cpu(
        engine_list: List[Engines],
        result_path: str,
        save_path: str,
        show: bool) -> None:
    engines_results_cpu = dict()
    for engine_type in engine_list:
        engine_result_path = os.path.join(result_path, engine_type.value + "_cpu.json")
        if not os.path.exists(engine_result_path):
            continue

        with open(engine_result_path, "r") as f:
            results_json = json.load(f)

        engines_results_cpu[engine_type] = results_json

    fig, ax = plt.subplots(figsize=(6, 4))
    x_limit = 0
    for engine_type, engine_value in engines_results_cpu.items():
        core_hour = (engine_value["total_process_time_sec"] / engine_value["total_audio_time_sec"] * 100)
        x_limit = max(x_limit, core_hour)
        ax.barh(
            ENGINE_PRINT_NAMES[engine_type],
            core_hour,
            height=0.5,
            color=ENGINE_COLORS[engine_type],
            edgecolor="none",
            label=ENGINE_PRINT_NAMES[engine_type],
        )
        ax.text(
            core_hour + 50,
            ENGINE_PRINT_NAMES[engine_type],
            f"{core_hour:.0f}\nCore-hour",
            ha="center",
            va="center",
            fontsize=12,
            color=ENGINE_COLORS[engine_type],
        )

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim([0, 1.2 * x_limit])
    ax.set_xticks([])
    ax.set_ylim([-0.5, len(engines_results_cpu) - 0.5])
    plt.title("Core-hour required to process 100 hours of audio (lower is better)", fontsize=12)
    plot_path = os.path.join(save_path, "cpu_usage_comparison.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    if show:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    dataset_name = args.dataset
    sorted_engines = sorted(ENGINES, key=lambda e: (ENGINE_ORDER_KEYS.get(e, 1), ENGINE_PRINT_NAMES.get(e, e.value)))

    save_path = os.path.join(RESULTS_FOLDER, "plots")

    result_dataset_path = os.path.join(RESULTS_FOLDER, dataset_name)
    _plot_accuracy(sorted_engines, result_dataset_path, os.path.join(save_path, dataset_name), args.show)
    _plot_cpu(sorted_engines, result_dataset_path, save_path, args.show)


if __name__ == "__main__":
    main()

__all__ = [
    "Color",
    "plot_results",
    "rgb_from_hex",
]
