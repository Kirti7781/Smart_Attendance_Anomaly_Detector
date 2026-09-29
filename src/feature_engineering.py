"""
src/feature_engineering.py
============================
Feature engineering pipeline for the Smart Attendance Anomaly Detector.

This module transforms raw cleaned attendance records into a numeric feature
matrix suitable for anomaly-detection algorithms (Isolation Forest, LOF).

Features Created
----------------
1.  hour                       – clock hour of check-in (already present)
2.  minute                     – clock minute (already present)
3.  day_of_week                – weekday index (already present)
4.  hour_deviation             – absolute difference from student's median check-in hour
5.  minute_sin / minute_cos    – cyclic encoding of minute-of-hour
6.  hour_sin / hour_cos        – cyclic encoding of hour-of-day
7.  weekly_attendance_rate     – already present
8.  monthly_attendance_rate    – already present
9.  attendance_rate_diff       – |weekly_rate – monthly_rate| (consistency check)
10. previous_attendance_gap    – hours since last attendance event
11. gap_log                    – log1p(previous_attendance_gap) for skew reduction
12. absence_ratio              – absence_count / total_classes
13. late_ratio                 – late_count / total_classes
14. historical_deviation       – absolute deviation from long-term baseline
15. location_frequency         – how often this student uses this location (normalised)
16. device_frequency           – how often this device appears across ALL students (proxy signal)
17. ip_frequency               – how often this IP appears across ALL students (proxy signal)
18. device_student_count       – number of unique students sharing this device
19. ip_student_count           – number of unique students sharing this IP
20. proxy_risk_indicator       – binary flag from generator

Note
----
*  ``student_id`` is NOT passed as a raw numeric feature.
   Instead, student-level statistics (e.g., median hour, location frequency)
   are computed and merged in.
*  ``is_anomaly`` and ``anomaly_type`` are excluded from the feature matrix.
*  All features are scaled with RobustScaler (robust to outliers).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Columns excluded from the feature matrix
# ─────────────────────────────────────────────
EXCLUDE_COLS = [
    "student_id", "date", "attendance_status", "attendance_time",
    "ip_address", "device_id", "class_id",
    "is_anomaly", "anomaly_type",
    # raw counts replaced by ratios:
    "total_classes", "classes_attended", "absence_count", "late_count",
]


def _cyclic_encode(series: pd.Series, max_val: float) -> Tuple[pd.Series, pd.Series]:
    """
    Cyclic (sine/cosine) encoding for periodic features.

    Why cyclic encoding?
    --------------------
    Hour 23 and hour 0 are only one hour apart, but numerically they are
    far apart (23 vs 0).  Sine/cosine encoding preserves this circularity.
    """
    sin_enc = np.sin(2 * np.pi * series / max_val)
    cos_enc = np.cos(2 * np.pi * series / max_val)
    return pd.Series(sin_enc, index=series.index), pd.Series(cos_enc, index=series.index)


def _compute_student_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-student statistics that describe the student's normal behaviour.

    These statistics are used to create relative features (e.g., how far this
    record deviates from the student's own median check-in hour).

    Only statistics derivable from the feature columns (not the label) are
    computed here to prevent data leakage.
    """
    student_stats = (
        df.groupby("student_id")
        .agg(
            student_median_hour=("hour", "median"),
            student_median_gap=("previous_attendance_gap", "median"),
            student_location_mode=("location_id", lambda x: x.mode()[0] if len(x) > 0 else -1),
        )
        .reset_index()
    )
    return student_stats


def _compute_sharing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute device/IP sharing statistics across ALL students.

    These capture proxy-attendance signals:
    - A device used by many different students is suspicious.
    - An IP address used by many different students is suspicious.
    """
    # Number of unique students per device
    device_stats = (
        df.groupby("device_id")["student_id"]
        .nunique()
        .rename("device_student_count")
        .reset_index()
    )

    # Number of unique students per IP
    ip_stats = (
        df.groupby("ip_address")["student_id"]
        .nunique()
        .rename("ip_student_count")
        .reset_index()
    )

    # Overall frequency of each device (as a fraction of total records)
    device_freq = (
        df["device_id"].value_counts(normalize=True)
        .rename("device_frequency")
        .reset_index()
        .rename(columns={"index": "device_id"})
    )

    # Overall frequency of each IP
    ip_freq = (
        df["ip_address"].value_counts(normalize=True)
        .rename("ip_frequency")
        .reset_index()
        .rename(columns={"index": "ip_address"})
    )

    return device_stats, ip_stats, device_freq, ip_freq


def engineer_features(
    df: pd.DataFrame,
    student_stats: pd.DataFrame | None = None,
    sharing_stats: tuple | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build the full numeric feature matrix from cleaned attendance records.

    Parameters
    ----------
    df            : pd.DataFrame
        Cleaned attendance data (may contain ``is_anomaly`` but it will be dropped).
    student_stats : pd.DataFrame or None
        Pre-computed student-level statistics.  If None and ``fit=True``,
        computed from ``df``.  Pass training stats when transforming val/test
        to avoid leakage.
    sharing_stats : tuple or None
        Pre-computed sharing statistics tuple from ``_compute_sharing_stats``.
        Same leakage-prevention logic as ``student_stats``.
    fit           : bool
        If True, compute statistics from ``df`` (use for training data).
        If False, ``student_stats`` and ``sharing_stats`` must be provided.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix (no label columns).
    stats : dict
        Dictionary of computed statistics for use on val/test splits.
    """
    df = df.copy()

    # ── Step 1: Compute or receive lookup tables ─────────────────────────
    if fit:
        student_stats = _compute_student_stats(df)
        sharing_stats = _compute_sharing_stats(df)
    else:
        assert student_stats is not None and sharing_stats is not None, \
            "Must provide pre-computed stats when fit=False."

    device_stats, ip_stats, device_freq, ip_freq = sharing_stats

    # ── Step 2: Merge lookup statistics ─────────────────────────────────
    df = df.merge(student_stats, on="student_id", how="left")
    df = df.merge(device_stats,  on="device_id",  how="left")
    df = df.merge(ip_stats,      on="ip_address", how="left")
    df = df.merge(device_freq,   on="device_id",  how="left")
    df = df.merge(ip_freq,       on="ip_address", how="left")

    # Fill NaN for unseen devices/IPs (new in val/test)
    for col in ["device_student_count", "ip_student_count",
                "device_frequency", "ip_frequency"]:
        df[col] = df[col].fillna(df[col].median())

    # ── Step 3: Derived features ─────────────────────────────────────────

    # 3a. Cyclic time encoding
    df["hour_sin"], df["hour_cos"]     = _cyclic_encode(df["hour"],   24)
    df["minute_sin"], df["minute_cos"] = _cyclic_encode(df["minute"], 60)

    # 3b. Deviation from student's own normal attendance time
    #     A high value means the student is checking in at an unusual hour.
    df["hour_deviation"] = (df["hour"] - df["student_median_hour"]).abs()

    # 3c. Log-transform of gap (right-skewed distribution)
    df["gap_log"] = np.log1p(df["previous_attendance_gap"])

    # 3d. Attendance efficiency ratios
    safe_total = df["total_classes"].replace(0, 1)  # avoid divide-by-zero
    df["absence_ratio"] = df["absence_count"] / safe_total
    df["late_ratio"]    = df["late_count"]    / safe_total

    # 3e. Consistency between weekly and monthly rates
    df["attendance_rate_diff"] = (
        df["weekly_attendance_rate"] - df["monthly_attendance_rate"]
    ).abs()

    # ── Step 4: Drop non-feature columns ─────────────────────────────────
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    # Also drop intermediate helper columns that are already encoded
    drop_cols += ["student_median_hour", "student_median_gap",
                  "student_location_mode", "location_id"]
    drop_cols = list(set(drop_cols))
    X = df.drop(columns=drop_cols, errors="ignore")

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.debug("Dropping non-numeric columns: %s", non_numeric)
        X = X.drop(columns=non_numeric)

    # ── Step 5: Final NaN fill (safety net) ──────────────────────────────
    X = X.fillna(X.median())

    stats = {
        "student_stats": student_stats,
        "sharing_stats": sharing_stats,
        "feature_columns": list(X.columns),
    }

    logger.info("Feature matrix shape: %s", X.shape)
    return X, stats


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Fit a RobustScaler on the training data and transform all splits.

    RobustScaler is preferred over StandardScaler for anomaly detection
    because it uses the median and IQR, making it robust to the very
    outliers we are trying to detect (StandardScaler's mean and variance
    would be distorted by extreme anomalous values).

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices from the train/val/test splits.

    Returns
    -------
    X_train_s, X_val_s, X_test_s : np.ndarray
        Scaled feature arrays.
    scaler : RobustScaler
        Fitted scaler (for later use in the Streamlit app).
    """
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def get_feature_names(stats: dict) -> List[str]:
    """Return the list of feature column names from the engineering stats dict."""
    return stats.get("feature_columns", [])
