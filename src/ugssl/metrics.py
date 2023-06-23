"""Various ensemble-based uncertainty metrics.

All metrics take a list of numpy arrays as input and output a dictionary
with the various calculated metrics.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
import numpy as np
import multiprocessing
from pathlib import Path
import pickle


class UncertaintyMetric(ABC):
    @abstractmethod
    def __call__(self, predictions: List[np.ndarray]) -> Dict:
        pass

    def _apply_pairwise(
        self, predictions: List[np.ndarray], function: callable
    ) -> Dict:
        n_predictions = len(predictions)
        metric_values = []
        for i in range(n_predictions):
            for j in range(i + 1, n_predictions):
                metric_values.append(self.metric(predictions[i], predictions[j]))

        return {"mean": np.mean(metric_values), "min": np.mean(metric_values)}


class PairwiseDSC(UncertaintyMetric):
    def __init__(self, labels: List[int], n_workers=8):
        self.labels = labels
        self.n_workers = n_workers
        self.format = "segmentation"

    def __call__(self, predictions: List[np.ndarray], pkl_path: Path = None) -> Dict:
        n_predictions = len(predictions)
        pool = multiprocessing.Pool(processes=self.n_workers)
        results = []
        for i in range(n_predictions):
            for j in range(i + 1, n_predictions):
                result = pool.apply_async(self.dsc, (predictions[i], predictions[j]))
                results.append(result)
        pool.close()
        pool.join()

        results = [result.get() for result in results]

        if pkl_path:
            self._pickle_raw_results(results, pkl_path)

        return self._reduce_results(results)

    def dsc(self, prediction_a: np.ndarray, prediction_b: np.ndarray) -> float:
        dsc_values = {}
        for label in self.labels:
            prediction_a_single = prediction_a == label
            prediction_b_single = prediction_b == label
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(
                prediction_a_single, prediction_b_single
            )
            if tp == 0 and fp == 0 and fn == 0:
                dsc = 1
            else:
                dsc = 2 * tp / (2 * tp + fp + fn)
            dsc_values[label] = dsc

        return dsc_values

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

    def _reduce_results(self, results):
        reduced = {}

        # Group dsc scores by label
        dsc_per_label = {}
        for result in results:
            for label in result:
                if label not in dsc_per_label:
                    dsc_per_label[label] = []

                dsc_per_label[label].append(result[label])

        # Reduce over pairs
        mean_dsc_per_label = []
        min_dsc_per_label = []
        for label in dsc_per_label:
            mean_dsc_per_label.append(np.mean(dsc_per_label[label]))
            min_dsc_per_label.append(np.min(dsc_per_label[label]))

        # Reduce over labels
        reduced["pair_mean_label_mean"] = np.mean(mean_dsc_per_label)
        reduced["pair_min_label_mean"] = np.mean(min_dsc_per_label)
        reduced["pair_mean_label_min"] = np.min(mean_dsc_per_label)
        reduced["pair_min_label_min"] = np.min(min_dsc_per_label)

        return reduced

    def _pickle_raw_results(self, results, pkl_path):
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(results, pkl_file)


class MaxProbability(UncertaintyMetric):
    def __init__(self):
        self.format = "softmax"

    def __call__(self, predictions: List[np.ndarray], pkl_path: Path = None) -> Dict:
        # Ensemble predictions
        ensemble = sum(predictions) / len(predictions)

        # Take max class prediction for each voxel
        ensemble = np.max(ensemble, axis=0)

        # Return the average over all pixels
        return {"max_probability": np.mean(ensemble)}


class Entropy(UncertaintyMetric):
    def __init__(self):
        self.format = "softmax"

    def __call__(self, predictions: List[np.ndarray], pkl_path: Path = None) -> Dict:
        # Ensemble predictions
        ensemble = sum(predictions) / len(predictions) + 1e-6

        # Take max class prediction for each voxel
        ensemble = -ensemble * np.log(ensemble)
        ensemble = np.sum(ensemble, axis=0)

        # Return the average over all pixels
        return {"entropy": np.mean(ensemble)}


class Variance(UncertaintyMetric):
    def __init__(self):
        self.format = "softmax"

    def __call__(self, predictions: List[np.ndarray], pkl_path: Path = None) -> Dict:
        variance = 0
        for c in range(predictions[0].shape[0]):
            stacked = np.stack([prediction[c] for prediction in predictions])
            variance += np.mean(np.var(stacked, axis=0)) / predictions[0].shape[0]

        # Return the average over all pixels
        return {"variance": variance}
