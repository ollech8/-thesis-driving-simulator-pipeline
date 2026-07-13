"""
build_dataset.py
----------------
Builds windowed datasets from correct_All_participants.csv.
Identical to run_on_correct_participants.py except:
  - TTC columns (TTC_min, TTC_mean, TTC_delta) are kept as NaN
    when no valid TTC exists — they are NOT filled with 999.
"""

import pandas as pd
import numpy as np
import re
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\claude code\event relavent feedback"
INPUT_PATH = os.path.join(BASE_DIR, "correct_All_participants.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "univariate_feedback_prediction", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZES = [1]

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — BUILD WINDOW DATASETS
# ══════════════════════════════════════════════════════════════════════════════

print("Loading CSV file...")
df = pd.read_csv(INPUT_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

REQUIRED_COLS = [
    "Id", "Condition", "Map", "WorldTime",
    "time_since_event_start_world",
    "Acceleration", "Speed", "SteeringAngle", "Brake",
    "Braking", "Accelerating",
    "distance_from_relevant_object", "relevant_object_name",
    "TrafficLight", "TrafficLight_JunctionPhase",
    "time_to_collision", "gap_acceptance",
    "SpacialEvent", "first_feedback_relavet_to_event"
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    print(f"\nWARNING: Missing columns: {missing}")
    print("   Continuing with available columns...\n")

# ─── Convert numeric columns ─────────────────────────────────────────────────
df["WorldTime"] = pd.to_numeric(df["WorldTime"], errors="coerce")
for col in ["time_since_event_start_world", "Acceleration", "Speed",
            "SteeringAngle", "Brake", "distance_from_relevant_object",
            "time_to_collision", "gap_acceptance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["Id", "Condition", "WorldTime"]).reset_index(drop=True)

# ─── Clean event name ─────────────────────────────────────────────────────────
def clean_event_name(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    val = re.sub(r'^(start|end)\s+', '', val, flags=re.IGNORECASE)
    return val.strip()

df["_clean_event"] = df["SpacialEvent"].apply(clean_event_name)
df["_is_tl"] = df["_clean_event"].str.lower().str.contains("traffic light", na=False)
df["_event_key"] = np.where(
    df["_is_tl"],
    df["Id"].astype(str) + "__" +
    df["Condition"].astype(str) + "__" +
    df["_clean_event"].astype(str),
    None
)

# ─── Fix leading-space typo variants from source ──────────────────────────────
for _typo, _fixed in [
    ("  first_feedback_relavet_to_event", "first_feedback_relavet_to_event"),
]:
    if _typo in df.columns:
        df = df.rename(columns={_typo: _fixed})

# ─── first_feedback_time_for_event: timestamp of the feedback row ─────────────
if "first_feedback_relavet_to_event" in df.columns:
    fb_flag = pd.to_numeric(df["first_feedback_relavet_to_event"], errors="coerce")
    fb_time_rows = df[df["_is_tl"] & (fb_flag == 1)].copy()
    first_fb_map = (
        fb_time_rows.groupby("_event_key")["time_since_event_start_world"].min()
    )
    df["_first_fb_event_time"] = df["_event_key"].map(first_fb_map)
else:
    df["_first_fb_event_time"] = np.nan

tl_df = df[df["_is_tl"]].copy()
print(f"\n  Traffic light rows: {len(tl_df):,}")
print(f"  Unique TL events:   {tl_df['_event_key'].nunique():,}")

# ─── Window builder ───────────────────────────────────────────────────────────
def build_windows_for_event(event_df: pd.DataFrame, window_sec: float) -> list[dict]:
    event_df = event_df.sort_values("time_since_event_start_world").reset_index(drop=True)
    if len(event_df) == 0:
        return []

    t_start  = event_df["time_since_event_start_world"].iloc[0]
    t_end_ev = event_df["time_since_event_start_world"].iloc[-1]

    first_fb_event_time = event_df["_first_fb_event_time"].iloc[0] \
        if "_first_fb_event_time" in event_df.columns else None
    if pd.isna(first_fb_event_time):
        first_fb_event_time = None

    first_fb_relat = first_fb_event_time

    windows = []
    t       = t_start
    win_idx = 0

    while t < t_end_ev:
        t_end_win = t + window_sec
        mask = (
            (event_df["time_since_event_start_world"] >= t) &
            (event_df["time_since_event_start_world"] <  t_end_win)
        )
        win = event_df[mask]
        is_partial = int(t_end_win > t_end_ev + 0.05)

        if len(win) == 0:
            t = t_end_win
            win_idx += 1
            continue

        row = {}
        row["Id"]                = win["Id"].iloc[0]
        row["Condition"]         = win["Condition"].iloc[0] if "Condition" in win.columns else None
        row["Map"]               = win["Map"].iloc[0] if "Map" in win.columns else None
        row["event_id"]          = win["_event_key"].iloc[0]
        row["window_index"]      = win_idx
        row["is_partial_window"] = is_partial
        row["start_time"]        = round(t, 4)
        row["end_time"]          = round(min(t_end_win, t_end_ev), 4)
        row["window_size_sec"]   = window_sec

        if "event_category" in win.columns:
            row["event_category"] = win["event_category"].iloc[0]

        if "time_since_event_start_world" in win.columns:
            vals = win["time_since_event_start_world"].dropna()
            row["max_time_since_event_start"] = vals.max() if not vals.empty else None

        if "relevant_object_name" in win.columns:
            mode_vals = win["relevant_object_name"].dropna()
            row["relevant_object_name"] = mode_vals.mode().iloc[0] if not mode_vals.empty else None

        if "distance_from_relevant_object" in win.columns:
            dist_vals = win["distance_from_relevant_object"].dropna()
            row["min_distance_from_relevant_object"] = dist_vals.min() if not dist_vals.empty else None
            if len(dist_vals) >= 2:
                row["delta_distance"] = round(float(dist_vals.iloc[0]) - float(dist_vals.iloc[-1]), 4)
            else:
                row["delta_distance"] = None

        if "Brake" in win.columns:
            row["Brake_max"]    = win["Brake"].max()
            row["Braking_rate"] = (win["Brake"] > 0).mean()

        if "Acceleration" in win.columns:
            row["Acceleration_max"]  = win["Acceleration"].max()
            row["Acceleration_min"]  = win["Acceleration"].min()
            row["Accelerating_rate"] = (win["Acceleration"] > 0).mean()

        if "Speed" in win.columns:
            row["Speed_mean"]  = win["Speed"].mean()
            row["Speed_max"]   = win["Speed"].max()
            row["Speed_std"]   = win["Speed"].std()
            row["Speed_delta"] = round(float(win["Speed"].iloc[-1]) - float(win["Speed"].iloc[0]), 4)
            row["is_stopped"]  = int(win["Speed"].mean() < 0.5)

        if "SteeringAngle" in win.columns:
            row["SteeringAngle_absmax"] = win["SteeringAngle"].abs().max()
            row["SteeringAngle_std"]    = win["SteeringAngle"].std()

        if "time_to_collision" in win.columns:
            ttc_vals = win["time_to_collision"].replace([np.inf, -np.inf], np.nan).dropna()
            # TTC stays NaN when no valid values exist — not filled downstream
            row["TTC_min"]   = ttc_vals.min()  if not ttc_vals.empty else np.nan
            row["TTC_mean"]  = ttc_vals.mean() if not ttc_vals.empty else np.nan
            if len(ttc_vals) >= 2:
                row["TTC_delta"] = round(float(ttc_vals.iloc[-1]) - float(ttc_vals.iloc[0]), 4)
            else:
                row["TTC_delta"] = np.nan

        if "TrafficLight" in win.columns:
            row["TrafficLight_prev"]    = win["TrafficLight"].iloc[0]
            row["TrafficLight_current"] = win["TrafficLight"].iloc[-1]
            tl_vals = win["TrafficLight"].dropna().unique().tolist()
            row["yellow_transition"] = int(
                "Yellow" in tl_vals and win["TrafficLight"].iloc[0] != "Yellow"
            )

        if "TrafficLight_JunctionPhase" in win.columns:
            row["prev_phase"]       = win["TrafficLight_JunctionPhase"].iloc[0]
            row["current_phase"]    = win["TrafficLight_JunctionPhase"].iloc[-1]
            row["transition_phase"] = int(row["prev_phase"] != row["current_phase"])

        t_win_end = row["end_time"]

        if first_fb_event_time is not None:
            fb_in_window = int(row["start_time"] <= first_fb_event_time <= t_win_end)
            row["first_feedback_relavet_to_event"] = fb_in_window
            row["feedback_next_1s"] = int(
                (first_fb_event_time > t_win_end) and
                (first_fb_event_time <= t_win_end + 1.0)
            )
        else:
            fb_in_window = 0
            row["first_feedback_relavet_to_event"] = 0
            row["feedback_next_1s"] = 0

        windows.append(row)

        if fb_in_window == 1:
            break

        t = t_end_win
        win_idx += 1

    return windows


# ─── Main window loop ─────────────────────────────────────────────────────────
all_results = {}
print()
for w in WINDOW_SIZES:
    print(f"Building {w}s windows...")
    all_windows = []
    for ekey, event_df in tl_df.groupby("_event_key"):
        wins = build_windows_for_event(event_df, window_sec=w)
        all_windows.extend(wins)

    result_df = pd.DataFrame(all_windows)
    all_results[w] = result_df

    n_events      = result_df["event_id"].nunique()
    n_with_target = (result_df.groupby("event_id")["feedback_next_1s"].sum() > 0).sum()
    pct_partial   = result_df["is_partial_window"].mean() * 100
    pct_positive  = result_df["feedback_next_1s"].mean() * 100

    print(f"  Total windows:            {len(result_df):,}")
    print(f"  Unique TL events:         {n_events:,}")
    print(f"  Events with target=1:     {n_with_target:,} / {n_events:,}")
    print(f"  Positive rate (target=1): {pct_positive:.1f}%")
    print(f"  Partial windows:          {pct_partial:.1f}%")

    out_path = os.path.join(OUTPUT_DIR, f"windows_{w}s.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_path}\n")

# ─── Combined file ────────────────────────────────────────────────────────────
print("Combining all window sizes...")
combined = pd.concat(
    [res_df.assign(window_size_sec=w) for w, res_df in all_results.items()],
    ignore_index=True
)
combined_path = os.path.join(OUTPUT_DIR, "windows_ALL_combined.csv")
combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"Combined file saved: {combined_path}")
print(f"   Total rows: {len(combined):,}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — FILL NaN VALUES  (TTC columns intentionally left as NaN)
# ══════════════════════════════════════════════════════════════════════════════

BIG_NUMBER = 999

FIXED_FILLS = {
    # TTC_min, TTC_mean, TTC_delta intentionally omitted — kept as NaN
    "TrafficLight_prev":          "Green",
    "TrafficLight_current":       "Green",
    "Speed_mean":                 0,
    "Speed_max":                  0,
    "Speed_std":                  0,
    "Speed_delta":                0,
    "is_stopped":                 1,
    "Brake_max":                  0,
    "Braking_rate":               0,
    "delta_distance":             0,
    "yellow_transition":          0,
    "transition_phase":           0,
    "max_time_since_event_start": 0,
    "Accelerating_rate":          0,
}

MEDIAN_FILLS = [
    "Acceleration_max",
    "Acceleration_min",
    "SteeringAngle_absmax",
    "SteeringAngle_std",
]

MEAN_FILLS = [
    "min_distance_from_relevant_object",
]

TTC_COLS = ["TTC_min", "TTC_mean", "TTC_delta"]


def clean_dataset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    report = []

    print(f"\n{'-' * 65}")
    print(f"  Dataset: {label}  |  Rows before cleaning: {len(df):,}")
    print(f"{'-' * 65}")

    if "relevant_object_name" in df.columns:
        n_before = len(df)
        df = df[df["relevant_object_name"].notna()].reset_index(drop=True)
        n_deleted = n_before - len(df)
        if n_deleted > 0:
            report.append(("relevant_object_name", "ROW DELETED", n_deleted))

    for phase_col in ["prev_phase", "current_phase"]:
        if phase_col not in df.columns:
            continue
        nan_mask = df[phase_col].isna()
        n = nan_mask.sum()
        if n > 0:
            df.loc[nan_mask & (df["is_partial_window"] == 1), phase_col] = "Approaching"
            df.loc[nan_mask & (df["is_partial_window"] == 0), phase_col] = "LeavingJunction"
            still_nan = df[phase_col].isna()
            if still_nan.any():
                df.loc[still_nan, phase_col] = "LeavingJunction"
            report.append((phase_col,
                            "Approaching (partial=1) / LeavingJunction (partial=0)",
                            n))

    for col, fill_val in FIXED_FILLS.items():
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna(fill_val)
            report.append((col, str(fill_val), n))

    for col in MEDIAN_FILLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append((col, f"median={median_val:.4f}", n))

    for col in MEAN_FILLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            report.append((col, f"mean={mean_val:.4f}", n))

    # Report TTC NaN counts without filling
    for col in TTC_COLS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            report.append((col, "kept as NaN (intentional)", n))

    remaining_nan_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in TTC_COLS:
            continue
        n = df[col].isna().sum()
        if n > 0:
            remaining_nan_cols.append((col, n))

    if report:
        print(f"\n  {'Column':<48} {'Fill value':<45} {'Count':>8}")
        print(f"  {'-'*48} {'-'*45} {'-'*8}")
        for col, val, n in report:
            action = "deleted" if val == "ROW DELETED" else "filled" if "NaN" not in val else "kept"
            print(f"  {col:<48} {str(val):<45} {n:>8,}  ({action})")
    else:
        print("  No NaN values found.")

    if remaining_nan_cols:
        print(f"\n  WARNING: Remaining NaNs after filling (check these manually):")
        for col, n in remaining_nan_cols:
            print(f"    {col:<48} {n:>8,}")
    else:
        print(f"\n  OK: No remaining NaN values in non-TTC numeric columns.")

    print(f"\n  Rows after cleaning: {len(df):,}")
    return df


print(f"\n\n{'=' * 65}")
print("  PART 2 -- NaN filling  (TTC kept as NaN)")
print(f"{'=' * 65}")

for w in [1]:
    in_path  = os.path.join(OUTPUT_DIR, f"windows_{w}s.csv")
    out_path = os.path.join(OUTPUT_DIR, f"windows_{w}s_clean.csv")

    if not os.path.exists(in_path):
        print(f"\n  File not found, skipping: {in_path}")
        continue

    df_win   = pd.read_csv(in_path)
    df_clean = clean_dataset(df_win, label=f"windows_{w}s.csv")
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_path}")

print("\nAll done!")
print(f"   Output folder: {OUTPUT_DIR}")
