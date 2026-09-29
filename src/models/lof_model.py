"""
src/models/lof_model.py
========================
Local Outlier Factor (LOF) Anomaly Detector

Algorithm Overview
------------------
LOF (Breunig et al., 2000) is a density-based anomaly detection algorithm.
Unlike Isolation Forest, which is global and tree-based, LOF measures
LOCAL density relative to the k-nearest neighbours.

Key Concepts
------------
1. k-Nearest Neighbours (kNN)
   For each point p, find its k nearest neighbours using Euclidean distance.

2. Reachability Distance
   reachability_dist(p, o) = max(k-dist(o), dist(p, o))
   where k-dist(o) is the distance from o to its k-th neighbour.
   Smooths out small distance fluctuations in dense regions.

3. Local Reachability Density (LRD)
   lrd(p) = 1 / (mean reachability distance from p to its k-neighbours)
   Higher LRD → point is in a DENSE neighbourhood (likely normal).

4. Local Outlier Factor
   LOF(p) = mean(lrd(o) for o in kNN(p)) / lrd(p)
   - LOF ≈ 1 → point has similar density to its neighbours (normal).
   - LOF >> 1 → point is in a much sparser region than its neighbours (anomaly).

How LOF Differs from Isolation Forest
--------------------------------------
| Property        | Isolation Forest          | LOF                         |
|-----------------|---------------------------|-----------------------------|
| Approach        | Global (tree-based)       | Local (density-based)       |
| Complexity      | O(n log n) per tree       | O(n²) for exact kNN         |
| Cluster shape   | Any shape                 | Irregular / complex shapes  |
| Novelty support | Yes (built-in)            | novelty=True flag needed    |
| Score type      | Path length               | Density ratio               |
| Best for        | Global, high-dim data     | Local cluster anomalies     |

Novelty Detection vs. Transductive LOF
----------------------------------------
By default, sklearn's LOF operates TRANSDUCTIVELY (fit_predict on same data).
For our use-case (separate train / val / test sets), we need NOVELTY DETECTION:
  - Set ``novelty=True``
  - Call ``.fit()`` on training data only
  - Call ``.predict()`` / ``.decision_function()`` on unseen val/test data.

Score Convention
----------------
``decision_function()`` returns negative values where more negative = more anomalous.
We negate the raw score so that HIGHER values = MORE anomalous (consistent with IF).

Parameters
----------
n_neighbors  : int
    Number of neighbours for density estimation.  Larger k → smoother density
    estimates but may miss fine-grained local clusters.
contamination : float
    Expected fraction of outliers.  Used internally to set the decision threshold.
novelty : bool
    Must be True to support predict/decision_function on new (unseen) data.
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


class LOFDetector:
    """
    Wrapper around sklearn's LocalOutlierFactor for the attendance anomaly project.

    Sets novelty=True to support separate train/val/test workflows.
    Score direction is standardised: higher score = more anomalous.
    """

    def __init__(
        self,
        n_neighbors:   int   = 20,
        contamination: float = 0.08,
        metric:        str   = "euclidean",
        n_jobs:        int   = -1,
    ) -> None:
        """
        Initialise the LOF detector.

        Parameters
        ----------
        n_neighbors   : k-nearest neighbours for density estimation.
        contamination : Expected fraction of outliers (sets decision threshold).
        metric        : Distance metric.  Euclidean works well for scaled features.
        n_jobs        : Parallel jobs for kNN computation (-1 = all cores).
        """
        self.n_neighbors   = n_neighbors
        self.contamination = contamination
        self.metric        = metric
        self.n_jobs        = n_jobs
        self._model: LocalOutlierFactor | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray) -> "LOFDetector":
        """
        Fit the LOF model on the training feature matrix.

        novelty=True is required so that the fitted model can later
        score previously unseen validation and test records.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        logger.info(
            "Training LOF: n_neighbors=%d, contamination=%.3f",
            self.n_neighbors, self.contamination,
        )
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,          # enables predict/decision_function on new data
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_train)
        logger.info("LOF training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for X.

        Returns
        -------
        np.ndarray of int
            1 = anomaly,  0 = normal
        """
        self._check_fitted()
        raw = self._model.predict(X)
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores (higher = more anomalous).

        sklearn's decision_function() returns negative LOF values where more
        negative means higher LOF score (more anomalous).
        We negate to make the direction intuitive.
        """
        self._check_fitted()
        raw = self._model.decision_function(X)
        return -raw  # negate: higher = more anomalous

    def predict_with_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns predictions and anomaly scores together.

        Returns
        -------
        predictions : np.ndarray  – 1=anomaly, 0=normal
        scores      : np.ndarray  – higher = more anomalous
        """
        return self.predict(X), self.score_samples(X)

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Serialise the fitted model to disk."""
        self._check_fitted()
        joblib.dump(self._model, filepath)
        logger.info("LOF model saved to '%s'.", filepath)

    def load(self, filepath: str) -> "LOFDetector":
        """Load a previously saved LOF model."""
        self._model = joblib.load(filepath)
        logger.info("LOF model loaded from '%s'.", filepath)
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    @property
    def model(self) -> LocalOutlierFactor:
        self._check_fitted()
        return self._model
