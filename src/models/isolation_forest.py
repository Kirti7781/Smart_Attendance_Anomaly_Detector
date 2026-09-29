"""
src/models/isolation_forest_model.py
=====================================
Isolation Forest Anomaly Detector

Algorithm Overview
------------------
Isolation Forest (Liu et al., 2008) is an ensemble anomaly detection algorithm
that works by explicitly ISOLATING anomalies rather than profiling normal points.

Key Idea
--------
Normal points require many random splits to isolate because they cluster together.
Anomalous points are rare and lie in sparse regions, so they are isolated in
very few splits (short path lengths).

The anomaly score is derived from the average path length across all trees:
  - Score close to 1   → very likely anomaly (short isolation path)
  - Score around 0.5   → uncertain / borderline
  - Score much less than 0.5 → likely normal

Parameters
----------
n_estimators : int
    Number of isolation trees.  More trees → more stable scores but slower training.
contamination : float
    Expected fraction of anomalies in the training set.
    Used internally by sklearn to set the decision threshold that separates
    the predict() output into -1 (anomaly) and +1 (normal).
    Setting contamination correctly is important:
      - Too low  → many real anomalies missed (high false negatives)
      - Too high → many normal records flagged (high false positives)
max_samples : int or "auto"
    Number of samples drawn to build each tree.
    "auto" uses min(256, n_samples) — a good default.
random_state : int
    Seed for reproducibility.

Score Convention
----------------
``decision_function`` returns NEGATIVE scores where more negative = more anomalous.
We NEGATE the raw score so that HIGHER values mean MORE anomalous.
This makes the score intuitive and consistent with LOF's convention.

Novelty vs. Transductive
-------------------------
Isolation Forest supports ``fit`` + ``predict``/``decision_function`` (novelty mode),
so it can be fitted on training data and applied to unseen validation/test data.
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Wrapper around sklearn's IsolationForest for the attendance anomaly project.

    The wrapper standardises:
    - Score direction (higher = more anomalous)
    - Prediction labels  (-1 / +1 converted to 1 / 0)
    - Save / load functionality
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        contamination: float = 0.08,
        max_samples:   str | int = "auto",
        random_state:  int   = 42,
    ) -> None:
        """
        Initialise the Isolation Forest detector.

        Parameters
        ----------
        n_estimators  : Number of trees in the ensemble.
        contamination : Expected fraction of anomalies (guides sklearn's threshold).
        max_samples   : Samples per tree ("auto" → min(256, n_samples)).
        random_state  : Reproducibility seed.
        """
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.max_samples   = max_samples
        self.random_state  = random_state
        self._model: IsolationForest | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on the training feature matrix.

        Note: The label vector y is intentionally NOT passed here because
        Isolation Forest is an UNSUPERVISED algorithm.  Labels are only used
        later for quantitative evaluation.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        logger.info(
            "Training Isolation Forest: n_estimators=%d, contamination=%.3f",
            self.n_estimators, self.contamination,
        )
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train)
        logger.info("Isolation Forest training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for X.

        Returns
        -------
        np.ndarray of int
            1 = anomaly,  0 = normal
            (sklearn returns -1 for anomaly, +1 for normal; we convert.)
        """
        self._check_fitted()
        raw = self._model.predict(X)
        # Convert sklearn convention: -1 → 1 (anomaly), +1 → 0 (normal)
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for each sample.

        Higher score → more anomalous.
        (sklearn's decision_function returns negative scores; we negate.)

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray of float
            Anomaly scores (higher = more suspicious).
        """
        self._check_fitted()
        # decision_function: more negative = more anomalous in sklearn convention
        raw_scores = self._model.decision_function(X)
        # Negate so that higher value = more anomalous
        return -raw_scores

    def predict_with_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method returning both labels and scores.

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
        """Serialise the fitted model to disk with joblib."""
        self._check_fitted()
        joblib.dump(self._model, filepath)
        logger.info("Isolation Forest saved to '%s'.", filepath)

    def load(self, filepath: str) -> "IsolationForestDetector":
        """Load a previously saved model from disk."""
        self._model = joblib.load(filepath)
        logger.info("Isolation Forest loaded from '%s'.", filepath)
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    @property
    def model(self) -> IsolationForest:
        self._check_fitted()
        return self._model
