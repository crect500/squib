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
    )

    def __init__(
        self: Metrics,
        truth: Iterable[int],
        predictions: Iterable[int],
    ) -> None:
        """
        Populate confusion matrix values.

        Args:
        ----
        truth: Known labels
        predictions: Predicted labels

        """
        self._populate_confusion_matrix(predictions, truth)
        self._calculate_accuracy()

    def __repr__(self: Metrics) -> str:
        """
        Return all confusion matrix values as one string.

        Return:
        ------
        The confusion matrix contents

        """
        return (
            f"Accuracy: {self.accuracy}\n"
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
