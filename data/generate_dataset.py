"""
Smart Attendance Anomaly Detector
===================================
Dataset Generator
-----------------
Generates a synthetic student attendance dataset with realistic normal
records AND deliberately injected anomalous records for ground-truth evaluation.

Usage:
    python data/generate_dataset.py

Outputs:
    data/attendance_dataset.csv  – full dataset (10 000 records)
    data/demo_attendance.csv     – smaller demo slice (1 000 records)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
NUM_STUDENTS   = 200          # number of unique students
NUM_RECORDS    = 10_000       # total attendance events
ANOMALY_RATE   = 0.08         # fraction of records that are anomalies (~8 %)
RANDOM_SEED    = 42           # reproducibility

NUM_CLASSES    = 20           # different class sessions
NUM_LOCATIONS  = 10           # possible classrooms / labs
NUM_DEVICES    = 150          # unique device IDs (slightly fewer than students -> sharing possible)
NUM_IPS        = 80           # unique IP addresses  (even fewer -> more sharing)

START_DATE     = datetime(2024, 1, 15)
END_DATE       = datetime(2024, 6, 30)

# Normal attendance hour range  (8 AM – 6 PM)
NORMAL_HOUR_MIN = 8
NORMAL_HOUR_MAX = 18

# ─────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86_400)
    return start + timedelta(days=random_days, seconds=random_seconds)


def build_student_profiles(n_students: int) -> dict:
    """
    Create a 'normal behaviour profile' for each student.
    Each student has a preferred attendance hour, preferred location,
    preferred device, preferred IP, and a stable weekly attendance rate.
    """
    profiles = {}
    for sid in range(1, n_students + 1):
        profiles[sid] = {
            "preferred_hour":     np.random.randint(NORMAL_HOUR_MIN, NORMAL_HOUR_MAX),
            "preferred_location": np.random.randint(1, NUM_LOCATIONS + 1),
            "preferred_device":   np.random.randint(1, NUM_DEVICES + 1),
            "preferred_ip":       np.random.randint(1, NUM_IPS + 1),
            # weekly_rate is the student's normal fraction of classes attended
            "weekly_rate":        np.clip(np.random.beta(a=8, b=2), 0.5, 1.0),
        }
    return profiles


# ─────────────────────────────────────────────
# NORMAL RECORD GENERATOR
# ─────────────────────────────────────────────

def generate_normal_record(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    Generate a single NORMAL attendance record that follows the student's profile.
    Small Gaussian noise is added to simulate realistic variation.
    """
    # Hour: student's preferred hour ± 1 hr std-dev
    hour   = int(np.clip(
        np.random.normal(profile["preferred_hour"], 1),
        NORMAL_HOUR_MIN, NORMAL_HOUR_MAX - 1
    ))
    minute = np.random.randint(0, 60)

    # Location: 85% chance of preferred location, 15% nearby location
    if np.random.rand() < 0.85:
        location = profile["preferred_location"]
    else:
        location = np.random.randint(1, NUM_LOCATIONS + 1)

    # Device: 90% chance of own device
    if np.random.rand() < 0.90:
        device = profile["preferred_device"]
    else:
        device = np.random.randint(1, NUM_DEVICES + 1)

    # IP: 85% chance of own IP (same campus network zone)
    if np.random.rand() < 0.85:
        ip = profile["preferred_ip"]
    else:
        ip = np.random.randint(1, NUM_IPS + 1)

    weekly_rate   = float(np.clip(np.random.normal(profile["weekly_rate"], 0.05), 0, 1))
    monthly_rate  = float(np.clip(np.random.normal(profile["weekly_rate"], 0.08), 0, 1))
    total_classes = np.random.randint(20, 50)
    classes_att   = int(total_classes * weekly_rate)
    gap           = float(np.clip(np.random.exponential(scale=2.0), 0.1, 24))  # hours

    return {
        "student_id":              f"STU{student_id:04d}",
        "date":                    record_date.strftime("%Y-%m-%d"),
        "day_of_week":             record_date.weekday(),          # 0=Mon … 6=Sun
        "class_id":                np.random.randint(1, NUM_CLASSES + 1),
        "attendance_status":       "present",
        "attendance_time":         f"{hour:02d}:{minute:02d}",
        "hour":                    hour,
        "minute":                  minute,
        "location_id":             location,
        "ip_address":              f"192.168.{ip}.1",
        "device_id":               f"DEV{device:04d}",
        "previous_attendance_gap": round(gap, 2),
        "weekly_attendance_rate":  round(weekly_rate, 4),
        "monthly_attendance_rate": round(monthly_rate, 4),
        "total_classes":           total_classes,
        "classes_attended":        classes_att,
        "late_count":              np.random.randint(0, 5),
        "absence_count":           total_classes - classes_att,
        "proxy_risk_indicator":    0,
        "historical_deviation":    round(abs(np.random.normal(0, 0.05)), 4),
        "is_anomaly":              0,
        "anomaly_type":            "normal",
    }


# ─────────────────────────────────────────────
# ANOMALY INJECTORS
# ─────────────────────────────────────────────

def inject_proxy_anomaly(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 1 – Proxy / shared-device attack.
    Same device / IP used by many students; unrealistically short gap.
    """
    rec = generate_normal_record(student_id, profile, record_date)
    # Force a commonly shared device and IP (low-numbered = high sharing)
    rec["device_id"]               = f"DEV{np.random.randint(1, 5):04d}"
    rec["ip_address"]              = f"192.168.{np.random.randint(1, 4)}.1"
    rec["previous_attendance_gap"] = round(np.random.uniform(0.01, 0.05), 4)  # < 3 min
    rec["proxy_risk_indicator"]    = 1
    rec["is_anomaly"]              = 1
    rec["anomaly_type"]            = "proxy"
    return rec


def inject_time_anomaly(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 2 – Attendance submitted at an unusual hour (late night / very early morning).
    """
    rec = generate_normal_record(student_id, profile, record_date)
    unusual_hour = int(np.random.choice(
        list(range(0, 5)) + list(range(22, 24))  # midnight–5 AM or 10 PM–midnight
    ))
    rec["hour"]             = unusual_hour
    rec["minute"]           = np.random.randint(0, 60)
    rec["attendance_time"]  = f"{unusual_hour:02d}:{rec['minute']:02d}"
    rec["is_anomaly"]       = 1
    rec["anomaly_type"]     = "time_anomaly"
    return rec


def inject_location_anomaly(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 3 – Attendance from an entirely different location than the student's norm.
    """
    rec = generate_normal_record(student_id, profile, record_date)
    # Pick a location far from the preferred location
    alt_location = profile["preferred_location"]
    while alt_location == profile["preferred_location"]:
        alt_location = np.random.randint(1, NUM_LOCATIONS + 1)
    rec["location_id"]  = alt_location
    rec["is_anomaly"]   = 1
    rec["anomaly_type"] = "location_anomaly"
    return rec


def inject_behavioral_anomaly(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 4 – Sudden behavioral change: a usually diligent student
    shows very low attendance rates or vice-versa.
    """
    rec = generate_normal_record(student_id, profile, record_date)
    if profile["weekly_rate"] > 0.7:
        # High-attendance student suddenly shows very low rate
        new_rate = np.random.uniform(0.1, 0.3)
    else:
        # Low-attendance student suddenly shows suspiciously perfect attendance
        new_rate = np.random.uniform(0.95, 1.0)
    rec["weekly_attendance_rate"]  = round(new_rate, 4)
    rec["monthly_attendance_rate"] = round(new_rate + np.random.uniform(-0.1, 0.1), 4)
    rec["historical_deviation"]    = round(abs(new_rate - profile["weekly_rate"]), 4)
    rec["is_anomaly"]              = 1
    rec["anomaly_type"]            = "behavioral_anomaly"
    return rec


def inject_attendance_burst(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 5 – Attendance burst: multiple check-ins in an impossibly short window.
    """
    rec = generate_normal_record(student_id, profile, record_date)
    rec["previous_attendance_gap"] = round(np.random.uniform(0.001, 0.03), 5)  # seconds-level gap
    rec["is_anomaly"]              = 1
    rec["anomaly_type"]            = "burst"
    return rec


def inject_historical_deviation(student_id: int, profile: dict, record_date: datetime) -> dict:
    """
    TYPE 6 – Historical deviation: extreme deviation from the student's own long-term baseline.
    """
    rec = generate_normal_record(student_id, profile, record_date)
    rec["historical_deviation"] = round(np.random.uniform(0.5, 1.0), 4)
    rec["is_anomaly"]           = 1
    rec["anomaly_type"]         = "historical_deviation"
    return rec


# Map anomaly type -> generator function
ANOMALY_GENERATORS = [
    inject_proxy_anomaly,
    inject_time_anomaly,
    inject_location_anomaly,
    inject_behavioral_anomaly,
    inject_attendance_burst,
    inject_historical_deviation,
]


# ─────────────────────────────────────────────
# MAIN DATASET BUILDER
# ─────────────────────────────────────────────

def generate_dataset(
    num_students:  int = NUM_STUDENTS,
    num_records:   int = NUM_RECORDS,
    anomaly_rate:  float = ANOMALY_RATE,
    random_seed:   int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build the complete synthetic attendance dataset.

    Parameters
    ----------
    num_students : int
        Number of unique students.
    num_records : int
        Total attendance events to generate.
    anomaly_rate : float
        Fraction of records that will be labelled anomalies.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined normal + anomalous records, shuffled.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    profiles    = build_student_profiles(num_students)
    student_ids = list(profiles.keys())

    n_anomalies = int(num_records * anomaly_rate)
    n_normal    = num_records - n_anomalies

    records = []

    # ── Normal records ──────────────────────────────────────────────────
    print(f"  Generating {n_normal:,} normal records …")
    for _ in range(n_normal):
        sid    = random.choice(student_ids)
        date   = random_date(START_DATE, END_DATE)
        records.append(generate_normal_record(sid, profiles[sid], date))

    # ── Anomalous records ────────────────────────────────────────────────
    print(f"  Injecting  {n_anomalies:,} anomalous records …")
    for _ in range(n_anomalies):
        sid       = random.choice(student_ids)
        date      = random_date(START_DATE, END_DATE)
        generator = random.choice(ANOMALY_GENERATORS)
        records.append(generator(sid, profiles[sid], date))

    df = pd.DataFrame(records).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"  Total records : {len(df):,}")
    print(f"  Anomalies     : {df['is_anomaly'].sum():,}  ({df['is_anomaly'].mean()*100:.1f} %)")
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = script_dir   # same folder as this script

    print("=" * 60)
    print("Smart Attendance Anomaly Detector – Dataset Generator")
    print("=" * 60)

    # ── Full dataset
    df_full = generate_dataset()
    full_path = os.path.join(data_dir, "attendance_dataset.csv")
    df_full.to_csv(full_path, index=False)
    print(f"\n  Saved full dataset -> {full_path}")

    # ── Demo dataset (first 1 000 rows, balanced anomaly fraction)
    n_demo_normal  = 920
    n_demo_anomaly = 80
    demo_normal    = df_full[df_full["is_anomaly"] == 0].head(n_demo_normal)
    demo_anomaly   = df_full[df_full["is_anomaly"] == 1].head(n_demo_anomaly)
    df_demo = (
        pd.concat([demo_normal, demo_anomaly])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    demo_path = os.path.join(data_dir, "demo_attendance.csv")
    df_demo.to_csv(demo_path, index=False)
    print(f"  Saved demo dataset -> {demo_path}")

    print("\n  Done.")
