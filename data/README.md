# Dataset README

## Smart Attendance Anomaly Detector – Synthetic Dataset

### Overview

This folder contains a synthetically generated student attendance dataset created specifically for academic experimentation. No real student data is used.

---

### Files

| File | Description |
|------|-------------|
| `attendance_dataset.csv` | Full dataset (10,000 records, ~200 students) |
| `demo_attendance.csv` | Smaller demo slice (1,000 records) for quick evaluation |
| `generate_dataset.py` | Script to regenerate the dataset |

---

### Dataset Fields

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | string | Unique student identifier (e.g., STU0001) |
| `date` | string | Attendance date (YYYY-MM-DD) |
| `day_of_week` | int | 0=Monday … 6=Sunday |
| `class_id` | int | Class/session identifier |
| `attendance_status` | string | "present" for all records |
| `attendance_time` | string | HH:MM format |
| `hour` | int | Hour component (0–23) |
| `minute` | int | Minute component (0–59) |
| `location_id` | int | Classroom/lab identifier |
| `ip_address` | string | IP address string |
| `device_id` | string | Device identifier |
| `previous_attendance_gap` | float | Hours since last attendance event |
| `weekly_attendance_rate` | float | Fraction of weekly classes attended (0–1) |
| `monthly_attendance_rate` | float | Fraction of monthly classes attended (0–1) |
| `total_classes` | int | Total classes in the period |
| `classes_attended` | int | Number of classes attended |
| `late_count` | int | Number of late arrivals |
| `absence_count` | int | Number of absences |
| `proxy_risk_indicator` | int | 1 if record was generated as proxy anomaly |
| `historical_deviation` | float | Absolute deviation from student's historical rate |
| **`is_anomaly`** | **int** | **Ground truth: 0=normal, 1=anomaly** |
| `anomaly_type` | string | "normal" or one of the 6 anomaly types |

---

### Statistics

| Metric | Value |
|--------|-------|
| Total records | 10,000 |
| Number of students | ~200 |
| Anomaly records | ~800 (8%) |
| Normal records | ~9,200 (92%) |
| Date range | 2024-01-15 to 2024-06-30 |

---

### Anomaly Types Injected

| Type | Description | Label |
|------|-------------|-------|
| Proxy | Shared device/IP, very short gap | `proxy` |
| Time Anomaly | Attendance at midnight/3AM | `time_anomaly` |
| Location Anomaly | Different classroom than normal | `location_anomaly` |
| Behavioral Anomaly | Sudden rate change | `behavioral_anomaly` |
| Attendance Burst | Multiple check-ins in seconds | `burst` |
| Historical Deviation | Large deviation from long-term average | `historical_deviation` |

---

### Important Notes

- `is_anomaly` is the **ground-truth label** used ONLY for evaluation – it must NOT be given to the ML algorithms as an input feature.
- All records with `attendance_status = "present"` because anomalies represent suspicious present-markings, not absences.
- The dataset is fully reproducible using `RANDOM_SEED = 42`.
