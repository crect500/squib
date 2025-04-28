"""Wrapper for KNN execution."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold

from squib.evaluation.metrics import Metrics

logger: logging.Logger = logging.getLogger(__name__)


def assign_label(
    results: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
) -> tuple[int, np.ndarray[int]]:
    """
    Assign result to label based on k-nearest neighbors.

    Args:
    ----
    results: The scaled inner products from processed QKNN circuit results
    labels: The labels corresponding to the results
    k: The number of neighbors to check in the k-nearest neighbors algorithm

    Returns:
    -------
    The determined label

    """
    k_indices: np.ndarray[int] = np.argpartition(results, -k)[:k]
    k_greatest: np.ndarray[int] = labels[k_indices]
    logger.warning(k_greatest)
    values, counts = np.unique(k_greatest, return_counts=True)
    return int(values[np.argmax(counts)]), k_greatest


def run_knn(
    training_set: np.ndarray,
    test_set: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    """
    Label each test feature using the knn algorithm.

    Args:
    ----
    training_set: The training features
    test_set: The test features to be labeled
    labels: The labels corresponding to the training features
    k: The number of neighbors to check in the k-nearest neighbors algorithm

    Returns:
    -------
    The labels for the test features and the sorted q- and k-nearest neighbors

    """
    working_labels: np.ndarray = deepcopy(labels)
    new_labels: np.ndarray = np.ndarray((len(test_set),), dtype=int)

    for iteration, test_feature in enumerate(test_set):
        elementwise_difference: np.ndarray[float] = np.subtract(
            training_set,
            test_feature,
        )
        new_label = assign_label(
            np.sum(elementwise_difference * elementwise_difference, axis=1),
            labels=working_labels,
            k=k,
        )[0]
        training_set = np.append(
            training_set,
            np.expand_dims(test_feature, 0),
            axis=0,
        )
        working_labels = np.append(working_labels, new_label)
        new_labels[iteration] = new_label

    return new_labels


def cross_validate_knn(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    seed: int = 1,
) -> list[Metrics]:
    """
    Run classical k-nearest neighbors.

    Args:
    ----
    features: The feature set
    labels: The labels for the feature set
    k: The k value for knn
    seed: Seed for cross-validation

    Returns:
    -------
    The metrics for the run

    """
    index_generator: StratifiedKFold = StratifiedKFold(shuffle=True, random_state=seed)
    metrics: list[Metrics] = []
    for iteration, (train_index, test_index) in enumerate(
        index_generator.split(features, labels),
    ):
        logger.warning(f"Training fold {iteration + 1} / 5")
        new_labels: np.ndarray = run_knn(
            features[train_index],
            features[test_index],
            labels[train_index],
            k=k,
        )
        logger.warning(f"Truth {labels[test_index]}")
        logger.warning(f"Predictions {new_labels}")
        metrics.append(Metrics(truth=labels[test_index], predictions=new_labels))
        logger.warning(metrics[-1])

    return metrics
