from enum import Enum
from typing import Sequence

import numpy as np


class Metrics(Enum):
    EER = "EER"


class Metric(object):
    def compute(self, positives: Sequence[float], negatives: Sequence[float]) -> float:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, metric: Metrics) -> "Metric":
        children = {
            Metrics.EER: EERMetric,
        }

        if metric not in children:
            raise ValueError(f"Cannot create `{cls.__name__}` of type `{metric.value}`")

        return children[metric]()


class EERMetric(Metric):
    def compute(self, positives: Sequence[float], negatives: Sequence[float]) -> float:
        positives = np.asarray(positives, dtype=np.double)
        negatives = np.asarray(negatives, dtype=np.double)

        scores = np.concatenate([positives, negatives])
        labels = \
            np.concatenate([np.ones(positives.size, dtype=np.int32), np.zeros(negatives.size, dtype=np.int32)])
        labels = labels[np.argsort(scores)[::-1]]

        tp = np.cumsum(labels == 1)
        fp = np.cumsum(labels == 0)

        fn = positives.size - tp

        frr = fn / positives.size
        far = fp / negatives.size

        idx = np.argmin(np.abs(far - frr))

        return float(0.5 * (far[idx] + frr[idx]))

    def __str__(self) -> str:
        return f"🧪[{Metrics.EER.value}]"


__all__ = [
    "Metric",
    "Metrics",
]
