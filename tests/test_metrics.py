from __future__ import annotations

from unittest import mock

import numpy as np
from hypothesis import given
from hypothesis.strategies import integers

from squib.evaluation.metrics import Metrics


@given(integers(min_value=1, max_value=7), integers(min_value=1, max_value=10))
def test_jaccard(k: int, feature_quantity: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    q_neighbors: np.ndarray[int] = random_generator.integers(
        0, 2, (feature_quantity, k),
    )
    k_neighbors: np.ndarray[int] = random_generator.integers(
        0, 2, (feature_quantity, k),
    )
    jaccard: float = 0
    for q_result, k_result in zip(q_neighbors, k_neighbors):
        correct: int = 0
        for q, c in zip(q_result, k_result):
            if q == c:
                correct += 1
        jaccard += correct / k

    jaccard /= len(q_neighbors)
    with mock.patch(
        "squib.evaluation.metrics.Metrics._populate_confusion_matrix",
    ), mock.patch("squib.evaluation.metrics.Metrics._calculate_accuracy"):
        metric = Metrics([], [])
        metric._calculate_jaccard(q_neighbors, k_neighbors)
        assert metric.jaccard == jaccard


@given(integers(min_value=1, max_value=7), integers(min_value=1, max_value=10))
def test_average_jaccard(k: int, feature_quantity: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    q_neighbors: np.ndarray[int] = random_generator.integers(
        0, 2, (feature_quantity, k),
    )
    k_neighbors: np.ndarray[int] = random_generator.integers(
        0, 2, (feature_quantity, k),
    )
    average_jaccard: float = 0
    for m in range(1, k + 1):
        jaccard: float = 0
        for q_result, k_result in zip(q_neighbors[:, :m], k_neighbors[:, :m]):
            correct: int = 0
            for q, c in zip(q_result, k_result):
                if q == c:
                    correct += 1
            jaccard += correct / m
        average_jaccard += jaccard

    average_jaccard /= feature_quantity * k
    with mock.patch(
        "squib.evaluation.metrics.Metrics._populate_confusion_matrix",
    ), mock.patch("squib.evaluation.metrics.Metrics._calculate_accuracy"):
        metric = Metrics([], [])
        metric._calculate_average_jaccard(q_neighbors, k_neighbors)
        assert metric.average_jaccard == average_jaccard
