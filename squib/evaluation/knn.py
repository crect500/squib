"""Wrapper for KNN execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from squib.evaluation.metrics import Metrics

if TYPE_CHECKING:
    import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


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
        neigbhors = KNeighborsClassifier(k)
        neigbhors.fit(features[train_index], labels[train_index])
        new_labels: np.ndarray = neigbhors.predict(features[test_index])
        metrics.append(Metrics(truth=labels[test_index], predictions=new_labels))
        logger.warning(metrics[-1])

    return metrics
