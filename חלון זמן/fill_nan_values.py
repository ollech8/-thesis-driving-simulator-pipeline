"""
fill_nan_values.py
------------------
Fills NaN values in BOTH window datasets:
  - windows_Xs.csv          → windows_Xs_clean.csv
  - sliding_windows_Xs.csv  → sliding_windows_Xs_clean.csv

Run this AFTER build_window_datasets.py / build_sliding_window_datasets.py
and BEFORE run_ml_models.py / run_ml_models_sliding.py.

NaN fill rules:
  - relevant_object_name NaN     → DELETE the row
  - TrafficLight_prev/current    → fill with "Green"
  - prev_phase / current_phase   → "Approaching" if is_partial_window=1
                                   "LeavingJunction" if is_partial_window=0
  - TTC_min, TTC_mean, TTC_delta → fill with 999 (no collision risk)
  - Speed_mean, Speed_max        → fill with 0 (assume stopped)
  - Speed_std, Speed_delta       → fill with 0
  - is_stopped                   → fill with 1
  - Brake_max, Braking_rate      → fill with 0
  - delta_distance               → fill with 0
  - yellow_transition            → fill with 0
  - transition_phase             → fill with 0
  - max_time_since_event_start   → fill with 0
  - Accelerating_rate            → fill with 0
  - Acceleration_max/min         → fill with training median
  - SteeringAngle_absmax/std     → fill with training median
  - min_distance_from_relevant_object → fill with training mean
"""

import pandas as pd
import numpy as np
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\דיבאג\חלון זמן"
WINDOW_SIZES = [1, 2, 3, 4, 5]
BIG_NUMBER   = 999

# ─── Fill rules ──────────────────────────────────────────────────────────────
# Columns filled with a fixed value (independent of data distribution)
FIXED_FILLS = {
    # Traffic light
    "TrafficLight_prev":             "Green",
    "TrafficLight_current":          "Green",
    # TTC — no collision risk
    "TTC_min":                       BIG_NUMBER,
    "TTC_mean":                      BIG_NUMBER,
    "TTC_delta":                     BIG_NUMBER,
    # Speed — assume stopped
    "Speed_mean":                    0,
    "Speed_max":                     0,
    "Speed_std":                     0,
    "Speed_delta":                   0,
    "is_stopped":                    1,
    # Braking — assume no braking
    "Brake_max":                     0,
    "Braking_rate":                  0,
    # Distance change — assume no change
    "delta_distance":                0,
    # Flags — assume no event
    "yellow_transition":             0,
    "transition_phase":              0,
    "max_time_since_event_start":    0,
    "Accelerating_rate":             0,
}

# Columns filled with training set median
MEDIAN_FILLS = [
    "Acceleration_max",
    "Acceleration_min",
    "SteeringAngle_absmax",
    "SteeringAngle_std",
]

# Columns filled with training set mean
MEAN_FILLS = [
    "min_distance_from_relevant_object",
]


# ─── Main cleaning function ───────────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Apply all NaN filling rules to a dataframe.
    Prints a detailed report.
    Returns the cleaned dataframe.
    """
    df = df.copy()
    report = []   # (column, fill_value, n_filled_or_deleted)

    print(f"\n{'─' * 65}")
    print(f"  Dataset: {label}  |  Rows before cleaning: {len(df):,}")
    print(f"{'─' * 65}")

    # ── 1. Delete rows where relevant_object_name is NaN ─────────────────────
    if "relevant_object_name" in df.columns:
        n_before = len(df)
        df = df[df["relevant_object_name"].notna()].reset_index(drop=True)
        n_deleted = n_before - len(df)
        if n_deleted > 0:
            report.append(("relevant_object_name", "ROW DELETED", n_deleted))

    # ── 2. Phase columns: fill based on is_partial_window ────────────────────
    for phase_col in ["prev_phase", "current_phase"]:
        if phase_col not in df.columns:
            continue
        nan_mask = df[phase_col].isna()
        n = nan_mask.sum()
        if n > 0:
            df.loc[nan_mask & (df["is_partial_window"] == 1), phase_col] = "Approaching"
            df.loc[nan_mask & (df["is_partial_window"] == 0), phase_col] = "LeavingJunction"
            # Handle rows where is_partial_window itself might be NaN
            still_nan = df[phase_col].isna()
            if still_nan.any():
                df.loc[still_nan, phase_col] = "LeavingJunction"
            report.append((phase_col,
                            "Approaching (partial=1) / LeavingJunction (partial=0)",
                            n))

    # ── 3. Fixed value fills ──────────────────────────────────────────────────
    for col, fill_val in FIXED_FILLS.items():
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna(fill_val)
            report.append((col, str(fill_val), n))

    # ── 4. Median fills (computed on this dataset — used as proxy for train) ──
    # Note: in the modeling script, medians are recomputed on train set only.
    # Here we use the full dataset median as a reasonable approximation.
    for col in MEDIAN_FILLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append((col, f"median={median_val:.4f}", n))

    # ── 5. Mean fills ─────────────────────────────────────────────────────────
    for col in MEAN_FILLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            report.append((col, f"mean={mean_val:.4f}", n))

    # ── 6. Check for any remaining NaNs in numeric columns ───────────────────
    remaining_nan_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        n = df[col].isna().sum()
        if n > 0:
            remaining_nan_cols.append((col, n))

    # ── Print report ──────────────────────────────────────────────────────────
    if report:
        print(f"\n  {'Column':<48} {'Fill value':<45} {'Count':>8}")
        print(f"  {'-'*48} {'-'*45} {'-'*8}")
        for col, val, n in report:
            action = "deleted" if val == "ROW DELETED" else "filled"
            print(f"  {col:<48} {str(val):<45} {n:>8,}  ({action})")
    else:
        print("  No NaN values found.")

    if remaining_nan_cols:
        print(f"\n  ⚠️  Remaining NaNs after filling (check these manually):")
        for col, n in remaining_nan_cols:
            print(f"    {col:<48} {n:>8,}")
    else:
        print(f"\n  ✅ No remaining NaN values in numeric columns.")

    print(f"\n  Rows after cleaning: {len(df):,}")
    return df


# ─── Main loop ───────────────────────────────────────────────────────────────
# Process both regular and sliding window files
DATASETS = [
    ("windows_{w}s.csv",         "windows_{w}s_clean.csv"),
    ("sliding_windows_{w}s.csv", "sliding_windows_{w}s_clean.csv"),
]

for in_template, out_template in DATASETS:
    print(f"\n{'═' * 65}")
    print(f"  Processing: {in_template}")
    print(f"{'═' * 65}")

    for w in WINDOW_SIZES:
        in_path  = os.path.join(DATA_DIR, in_template.format(w=w))
        out_path = os.path.join(DATA_DIR, out_template.format(w=w))

        if not os.path.exists(in_path):
            print(f"\n  ⚠️  File not found, skipping: {in_path}")
            continue

        df = pd.read_csv(in_path)
        df_clean = clean_dataset(df, label=in_template.format(w=w))

        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Saved: {out_path}")

print("\n✨ NaN filling complete!")