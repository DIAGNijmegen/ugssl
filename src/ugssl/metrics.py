"""Various ensemble-based uncertainty metrics.

TODO think about what is smart here, maybe better to have single
separate function that output min/mean+min/mean in dict, so that I can
use it in other scripts more easily.
I can then just wrap that in a class.

This way I can also add more detailed stats about dsc, such as which
organ is worst etc.

At the same time, I can also implement MPV, entropy, etc. with the same
abstract class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
import numpy as np


class UncertaintyMetric(ABC):
    @abstractmethod
    def __call__(self, predictions: List[np.ndarray]):
        pass


class PairwiseUncertaintyMetric(ABC):
    def __call__(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        n_predictions = len(predictions)
        metric_values = []
        for i in range(n_predictions):
            for j in range(i + 1, n_predictions):
                metric_values.append(self.metric(predictions[i], predictions[j]))

        return {"mean": np.mean(metric_values), "min": np.mean(metric_values)}

    @abstractmethod
    def metric(self, prediction_a: np.ndarray, prediction_b: np.ndarray) -> float:
        pass


class PairwiseDSC(PairwiseUncertaintyMetric):
    def __init__(self, labels: List[int], mode: str = "mean"):
        self.labels = labels
        self.mode = mode

    def metric(self, prediction_a: np.ndarray, prediction_b: np.ndarray) -> float:
        dsc_values = []
        for label in self.labels:
            prediction_a_single = prediction_a == label
            prediction_b_single = prediction_b == label
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(
                prediction_a_single, prediction_b_single
            )
            dsc = 2 * tp / (2 * tp + fp + fn)
            dsc_values.append(dsc)

        if self.mode == "mean":
            return np.mean(dsc_values)

        if self.mode == "min":
            return np.min(dsc_values)

    def compute_tp_fp_fn_tn(
        self, label: np.ndarray, prediction: np.ndarray, ignore_mask: np.ndarray = None
    ) -> Tuple[int, int, int, int]:
        """This function is taken from nnunet
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/evaluation/evaluate_predictions.py
        """
        if ignore_mask is None:
            use_mask = np.ones_like(label, dtype=bool)
        else:
            use_mask = ~ignore_mask
        tp = np.sum((label & prediction) & use_mask)
        fp = np.sum(((~label) & prediction) & use_mask)
        fn = np.sum((label & (~prediction)) & use_mask)
        tn = np.sum(((~label) & (~prediction)) & use_mask)
        return tp, fp, fn, tn
