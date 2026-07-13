# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:22:18 2026

@author: ASUS VIVOBOOK
"""

"""
build_sliding_window_datasets.py
---------------------------------
Builds overlapping (sliding) time-window datasets from raw participant analysis file.
Windows: sizes 1s, 2s, 3s, 4s, 5s — all with a FIXED STEP of 1 second.

Key difference from build_window_datasets.py:
  - Step is always 1s regardless of window size.
  - This ensures no feedback is missed between window boundaries.
  - Windows overlap for sizes > 1s (e.g. 2s window: [0-2], [1-3], [2-4], ...)
  - window_index resets to 0 at each new event, counts in 1s steps.
  - Windows stop after the first target=1 window (feedback already fired).

Target variable:
  - feedback_next_1s = 1 if first accompanier feedback occurs in (end_time, end_time+1s]
  - Always a 1s lookahead regardless of window size — consistent across all datasets.

Event identification: SpacialEvent column
  - Clean name: strip leading "Start"/"End" → "traffic light 2"
  - Unique event key = Id + Condition + clean_event_name

Output: one CSV per window size saved to OUTPUT_DIR.
"""

import pandas as pd
import numpy as np
import re
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
INPUT_PATH = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\דיבאג\ALL_participants_analysis_ALL_conditions.csv"
OUTPUT_DIR = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\דיבאג\חלון זמן"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZES = [1, 2, 3, 4, 5]
STEP_SEC     = 1.0          # fixed step — always 1 second
LOOKAHEAD    = 1.0          # always check next 1 second for feedback
FEEDBACK_COL = "start_comment"

# ─── Load ────────────────────────────────────────────────────────────────────
print("Loading raw file...")
df = pd.read_csv(INPUT_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# ─── Convert numeric columns ─────────────────────────────────────────────────
df["WorldTime"] = pd.to_numeric(df["WorldTime"], errors="coerce")
for col in ["time_since_event_start_world", "Acceleration", "Speed",
            "SteeringAngle", "Brake", "distance_from_relevant_object",
            "time_to_collision", "gap_acceptance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ─── Sort ────────────────────────────────────────────────────────────────────
df = df.sort_values(["Id", "Condition", "WorldTime"]).reset_index(drop=True)

# ─── Clean event name ────────────────────────────────────────────────────────
def clean_event_name(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    val = re.sub(r'^(start|end)\s+', '', val, flags=re.IGNORECASE)
    return val.strip()

df["_clean_event"] = df["SpacialEvent"].apply(clean_event_name)
df["_is_tl"]       = df["_clean_event"].str.lower().str.contains("traffic light", na=False)

# ─── Unique event key ────────────────────────────────────────────────────────
df["_event_key"] = np.where(
    df["_is_tl"],
    df["Id"].astype(str) + "__" +
    df["Condition"].astype(str) + "__" +
    df["_clean_event"].astype(str),
    None
)

# ─── First accompanier feedback time per event ───────────────────────────────
def is_accompanier_feedback(speaker_val):
    if pd.isna(speaker_val):
        return True
    s = str(speaker_val).lower()
    return "accompanier" in s

fb_rows = df[
    df["_is_tl"] &
    (df[FEEDBACK_COL].fillna(0).astype(float) == 1) &
    (df["speaker"].apply(is_accompanier_feedback) if "speaker" in df.columns
     else pd.Series([True] * len(df)))
].copy()

first_fb = (
    fb_rows.groupby("_event_key")["time_since_event_start_world"]
    .min()
    .rename("_first_fb_event_time")
)
df = df.merge(first_fb, on="_event_key", how="left")

tl_df = df[df["_is_tl"]].copy()

print(f"\n  Traffic light rows: {len(tl_df):,}")
print(f"  Unique TL events:   {tl_df['_event_key'].nunique():,}")
print(f"\n  Example event keys:")
for k in tl_df["_event_key"].dropna().unique()[:6]:
    print(f"    {k}")
print()

# ─── Sliding window builder ───────────────────────────────────────────────────
def build_sliding_windows(event_df: pd.DataFrame, window_sec: float) -> list[dict]:
    """
    Build overlapping windows with fixed 1s step along time_since_event_start_world.
    - window_index resets to 0 at each event, increments by 1 each step
    - Stops after the first window where feedback_next_1s=1
    - Partial windows at end of event are kept with is_partial_window=1
    """
    event_df = event_df.sort_values("time_since_event_start_world").reset_index(drop=True)
    if len(event_df) == 0:
        return []

    t_start  = event_df["time_since_event_start_world"].iloc[0]
    t_end_ev = event_df["time_since_event_start_world"].iloc[-1]

    first_fb_event_time = event_df["_first_fb_event_time"].iloc[0]
    if pd.isna(first_fb_event_time):
        first_fb_event_time = None

    windows  = []
    t        = t_start
    win_idx  = 0

    while t < t_end_ev:
        t_end_win = t + window_sec

        mask = (
            (event_df["time_since_event_start_world"] >= t) &
            (event_df["time_since_event_start_world"] <  t_end_win)
        )
        win = event_df[mask]

        is_partial = int(t_end_win > t_end_ev + 0.05)

        if len(win) == 0:
            t += STEP_SEC
            win_idx += 1
            continue

        row = {}

        # ── Identifiers ──────────────────────────────────────────────────────
        row["Id"]                = win["Id"].iloc[0]
        row["Condition"]         = win["Condition"].iloc[0] if "Condition" in win.columns else None
        row["Map"]               = win["Map"].iloc[0]       if "Map"       in win.columns else None
        row["event_id"]          = win["_event_key"].iloc[0]
        row["window_index"]      = win_idx
        row["is_partial_window"] = is_partial

        # ── Window boundaries (event-time seconds) ────────────────────────────
        row["start_time"]        = round(t, 4)
        row["end_time"]          = round(min(t_end_win, t_end_ev), 4)
        row["window_size_sec"]   = window_sec

        # ── Event context ─────────────────────────────────────────────────────
        if "event_category" in win.columns:
            row["event_category"] = win["event_category"].iloc[0]

        if "time_since_event_start_world" in win.columns:
            vals = win["time_since_event_start_world"].dropna()
            row["max_time_since_event_start"] = vals.max() if not vals.empty else None

        if "relevant_object_name" in win.columns:
            mode_vals = win["relevant_object_name"].dropna()
            row["relevant_object_name"] = mode_vals.mode().iloc[0] if not mode_vals.empty else None

        # ── Distance ──────────────────────────────────────────────────────────
        if "distance_from_relevant_object" in win.columns:
            dist_vals = win["distance_from_relevant_object"].dropna()
            row["min_distance_from_relevant_object"] = dist_vals.min() if not dist_vals.empty else None
            if len(dist_vals) >= 2:
                row["delta_distance"] = round(float(dist_vals.iloc[0]) - float(dist_vals.iloc[-1]), 4)
            else:
                row["delta_distance"] = None

        # ── Kinematics ────────────────────────────────────────────────────────
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

        # ── Time to collision ─────────────────────────────────────────────────
        if "time_to_collision" in win.columns:
            ttc_vals = win["time_to_collision"].replace([np.inf, -np.inf], np.nan).dropna()
            row["TTC_min"]  = ttc_vals.min()  if not ttc_vals.empty else None
            row["TTC_mean"] = ttc_vals.mean() if not ttc_vals.empty else None
            if len(ttc_vals) >= 2:
                row["TTC_delta"] = round(float(ttc_vals.iloc[-1]) - float(ttc_vals.iloc[0]), 4)
            else:
                row["TTC_delta"] = None

        # ── Traffic light state ───────────────────────────────────────────────
        if "TrafficLight" in win.columns:
            row["TrafficLight_prev"]    = win["TrafficLight"].iloc[0]
            row["TrafficLight_current"] = win["TrafficLight"].iloc[-1]
            tl_vals = win["TrafficLight"].dropna().unique().tolist()
            row["yellow_transition"]    = int(
                "Yellow" in tl_vals and win["TrafficLight"].iloc[0] != "Yellow"
            )

        # ── Junction phase ────────────────────────────────────────────────────
        if "TrafficLight_JunctionPhase" in win.columns:
            row["prev_phase"]       = win["TrafficLight_JunctionPhase"].iloc[0]
            row["current_phase"]    = win["TrafficLight_JunctionPhase"].iloc[-1]
            row["transition_phase"] = int(row["prev_phase"] != row["current_phase"])

        # ── Target variable ───────────────────────────────────────────────────
        # feedback_next_1s = 1 if first feedback occurs in (end_time, end_time+1s]
        t_win_end = row["end_time"]

        if first_fb_event_time is not None:
            row["feedback_next_1s"] = int(
                first_fb_event_time > t_win_end and
                first_fb_event_time <= t_win_end + LOOKAHEAD
            )
        else:
            row["feedback_next_1s"] = 0

        windows.append(row)

        # Stop after the first target=1 window
        if row["feedback_next_1s"] == 1:
            break

        t += STEP_SEC
        win_idx += 1

    return windows


# ─── Main loop ───────────────────────────────────────────────────────────────
all_results = {}

for w in WINDOW_SIZES:
    print(f"\n⏱  Building sliding windows — size={w}s, step=1s...")
    all_windows = []

    for ekey, event_df in tl_df.groupby("_event_key"):
        wins = build_sliding_windows(event_df, window_sec=w)
        all_windows.extend(wins)

    result_df = pd.DataFrame(all_windows)
    all_results[w] = result_df

    n_events      = result_df["event_id"].nunique()
    n_with_target = (result_df.groupby("event_id")["feedback_next_1s"].sum() > 0).sum()
    pct_positive  = result_df["feedback_next_1s"].mean() * 100
    pct_partial   = result_df["is_partial_window"].mean() * 100

    print(f"  Total windows:             {len(result_df):,}")
    print(f"  Unique TL events:          {n_events:,}")
    print(f"  Events with target=1:      {n_with_target:,} / {n_events:,}")
    print(f"  Positive rate (target=1):  {pct_positive:.2f}%")
    print(f"  Partial windows:           {pct_partial:.1f}%")
    print(f"  Avg windows per event:     {len(result_df) / max(n_events, 1):.1f}")

    out_path = os.path.join(OUTPUT_DIR, f"sliding_windows_{w}s.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {out_path}")

print("\n✨ Done!")