"""Provides support for metrics calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class Metrics:

    """
    Store metrics of a model prediction.

    Attributes
    ----------
    confusion_matrix: A matrix containing label predictions vs true values
    accuracy: The percentage of correct predictions

    """

    __slots__ = (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "accuracy",
        "jaccard",
        "average_jaccard",
    )

    def __init__(
        self: Metrics,
        truth: Iterable[int],
        predictions: Iterable[int],
        *,
        quantum_neighbors: np.ndarray[int] | None = None,
        classical_neighbors: np.ndarray[int] | None = None,
    ) -> None:
        """
        Populate confusion matrix values.

        Args:
        ----
        truth: Known labels
        predictions: Predicted labels
        quantum_neighbors: The quantum generated nearest neighbors labels
        classical_neighbors: The classically generated nearest neighbors labels

        """
        self._populate_confusion_matrix(predictions, truth)
        self._calculate_accuracy()
        if isinstance(quantum_neighbors, np.ndarray):
            self._calculate_jaccard(quantum_neighbors, classical_neighbors)
            self._calculate_average_jaccard(quantum_neighbors, classical_neighbors)
        else:
            self.jaccard = None
            self.average_jaccard = None

    def __repr__(self: Metrics) -> str:
        """
        Return all confusion matrix values as one string.

        Return:
        ------
        The confusion matrix contents

        """
        if not self.jaccard:
            return (
                f"Accuracy: {self.accuracy}\n"
                f"True Positives: {self.true_positive}\n"
                f"False Positives: {self.false_positive}\n"
                f"False Negatives: {self.false_negative}\n"
                f"True Negatives: {self.true_negative}"
            )

        return (
            f"Accuracy: {self.accuracy}\n"
            f"Jaccard: {self.jaccard}\n"
            f"Average Jaccard: {self.average_jaccard}\n"
            f"True Positives: {self.true_positive}\n"
            f"False Positives: {self.false_positive}\n"
            f"False Negatives: {self.false_negative}\n"
            f"True Negatives: {self.true_negative}"
        )

    def _populate_confusion_matrix(
        self: Metrics,
        predictions: Iterable[int],
        truth: Iterable[int],
    ) -> None:
        """
        Store raw prediction analysis in ConfusionMatrix object.

        Args:
        ----
        predictions: Float-valued probabilities, between 0 and 1
        truth: Truth data, valued 0 or 1

        """
        self.true_positive: int = int(np.sum(np.logical_and(predictions, truth)))
        self.true_negative: int = int(
            np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(truth))),
        )
        self.false_positive: int = int(
            np.sum(np.logical_and(predictions, np.logical_not(truth))),
        )
        self.false_negative: int = int(
            np.sum(np.logical_and(np.logical_not(predictions), truth)),
        )

    def _calculate_accuracy(self: Metrics) -> None:
        """Store accuracy of model results in class attribute."""
        try:
            trues: int = self.true_positive + self.true_negative
            self.accuracy = trues / (trues + self.false_positive + self.false_negative)
        except ZeroDivisionError:
            self.accuracy = 0

    def _calculate_jaccard(
        self: Metrics,
        quantum_neighbors: np.ndarray[int],
        classical_neighbors: np.ndarray[int],
    ) -> None:
        total: float = 0
        for quantum_result, classical_result in zip(
            quantum_neighbors,
            classical_neighbors,
        ):
            jaccard_index: float = np.sum(quantum_result == classical_result) / len(
                classical_result,
            )
            total += jaccard_index

        self.jaccard = total / len(classical_neighbors)

    def _calculate_average_jaccard(
        self: Metrics,
        quantum_neighbors: np.ndarray[int],
        classical_neighbors: np.ndarray[int],
    ) -> None:
        total: float = 0
        for k in range(1, quantum_neighbors.shape[1] + 1):
            subtotal: float = 0
            for quantum_slice, classical_slice in zip(
                quantum_neighbors[:, :k],
                classical_neighbors[:, :k],
            ):
                jaccard_index: float = np.sum(quantum_slice == classical_slice) / len(
                    classical_slice,
                )
                subtotal += jaccard_index
            total += subtotal

        self.average_jaccard = total / (
            classical_neighbors.shape[0] * classical_neighbors.shape[1]
        )
