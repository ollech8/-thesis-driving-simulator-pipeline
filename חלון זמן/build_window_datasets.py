"""
build_window_datasets.py
------------------------
Builds time-window datasets from raw participant analysis file.
Windows: 1s, 2s, 3s, 4s, 5s (non-overlapping, step = window size)

Window logic:
  - Windows are built per event using time_since_event_start_world as the time axis.
  - window_index resets to 0 at the start of each event.
  - start_time / end_time are time_since_event_start values (seconds into the event).
  - Partial windows at the end of an event are KEPT with is_partial_window=1.

Event identification (SpacialEvent column):
  - Clean name: strip leading "Start " / "End "  →  "traffic light 2"
  - All rows are expected to be non-NaN.
  - Unique event key = Id + Condition + clean_event_name

Target variable:
  - feedback_next_1s = 1 if the first start_comment=1 in this event occurs
    within 1 second AFTER the end of the current window (end_time, end_time+1].
    This is independent of window boundaries.
  - censored = 1 if:
      (a) no feedback exists anywhere in this event, OR
      (b) the feedback already occurred before or inside this window
          (i.e. the event is "over" from a survival standpoint).

time_since_event_start columns:
  - start_time  = window start in event-time  (time_since_event_start of first row)
  - end_time    = window end   in event-time  (start_time + window_sec)
  - time_since_event_start = MAX of time_since_event_start_world within the window

Output: one CSV per window size + one combined CSV, saved to INPUT_PATH directory.
"""

import pandas as pd
import numpy as np
import re
import os

# ─── Paths ──────────────────────────────────────────────────────────────────
INPUT_PATH = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\דיבאג\ALL_participants_analysis_ALL_conditions.csv"
OUTPUT_DIR = os.path.dirname(INPUT_PATH)

WINDOW_SIZES = [1, 2, 3, 4, 5]
FEEDBACK_COL = "start_comment"

# ─── Load ────────────────────────────────────────────────────────────────────
print("Loading raw file...")
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
    "SpacialEvent", FEEDBACK_COL
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    print(f"\n⚠️  Missing columns: {missing}")
    print("   Continuing with available columns...\n")

# ─── Convert numeric columns ─────────────────────────────────────────────────
df["WorldTime"] = pd.to_numeric(df["WorldTime"], errors="coerce")

for col in ["time_since_event_start_world", "Acceleration", "Speed",
            "SteeringAngle", "Brake", "distance_from_relevant_object",
            "time_to_collision", "gap_acceptance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ─── Sort ────────────────────────────────────────────────────────────────────
df = df.sort_values(["Id", "Condition", "WorldTime"]).reset_index(drop=True)

# ─── Step 1: Clean event name ────────────────────────────────────────────────
# "Start traffic light 2"  →  "traffic light 2"
# "End traffic light 2"    →  "traffic light 2"
# "traffic light 2"        →  "traffic light 2"

def clean_event_name(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    val = re.sub(r'^(start|end)\s+', '', val, flags=re.IGNORECASE)
    return val.strip()

df["_clean_event"] = df["SpacialEvent"].apply(clean_event_name)

# ─── Step 2: Flag traffic light rows ─────────────────────────────────────────
df["_is_tl"] = df["_clean_event"].str.lower().str.contains("traffic light", na=False)

# ─── Step 3: Unique event key (TL rows only) ─────────────────────────────────
# Format: "D01__B__traffic light 2"
df["_event_key"] = np.where(
    df["_is_tl"],
    df["Id"].astype(str) + "__" +
    df["Condition"].astype(str) + "__" +
    df["_clean_event"].astype(str),
    None
)

# ─── Step 4: First feedback time (in event-time) per event ───────────────────
# Only count feedback from the accompanier.
# Exclude rows where speaker is "young driver" alone.
# Include rows where speaker contains "accompanier" (even if also lists other roles).

def is_accompanier_feedback(speaker_val):
    """Return True if this feedback row should be attributed to the accompanier.
    - Include if speaker is NaN (unknown speaker, assumed valid)
    - Include if speaker contains 'accompanier'
    - Exclude only if speaker is solely 'young driver' with no other role
    """
    if pd.isna(speaker_val):
        return True
    s = str(speaker_val).lower()
    return "accompanier" in s

fb_rows = df[
    df["_is_tl"] &
    (df[FEEDBACK_COL].fillna(0).astype(float) == 1) &
    df["speaker"].apply(is_accompanier_feedback)
].copy() if "speaker" in df.columns else df[
    df["_is_tl"] &
    (df[FEEDBACK_COL].fillna(0).astype(float) == 1)
].copy()

first_fb = (
    fb_rows.groupby("_event_key")["time_since_event_start_world"]
    .min()
    .rename("_first_fb_event_time")   # seconds into the event
)
df = df.merge(first_fb, on="_event_key", how="left")

# Keep only TL rows
tl_df = df[df["_is_tl"]].copy()

print(f"\n  Traffic light rows: {len(tl_df):,}")
print(f"  Unique TL events:   {tl_df['_event_key'].nunique():,}")
print("\n  Example event keys:")
for k in tl_df["_event_key"].dropna().unique()[:6]:
    print(f"    {k}")
print()

# ─── Window builder per event ─────────────────────────────────────────────────
def build_windows_for_event(event_df: pd.DataFrame, window_sec: float) -> list[dict]:
    """
    Build non-overlapping windows along the time_since_event_start_world axis.
    window_index resets to 0 for each event.
    Partial windows at the end are kept with is_partial_window=1.

    Target: feedback_next_1s = 1 if first_feedback_event_time is in
            (end_time, end_time + 1]  — i.e. within 1 second after this window.
    """
    event_df = event_df.sort_values("time_since_event_start_world").reset_index(drop=True)
    if len(event_df) == 0:
        return []

    t_start  = event_df["time_since_event_start_world"].iloc[0]   # usually ~0
    t_end_ev = event_df["time_since_event_start_world"].iloc[-1]  # end of event

    # First feedback in event-time (None if no feedback)
    first_fb_event_time = event_df["_first_fb_event_time"].iloc[0]
    if pd.isna(first_fb_event_time):
        first_fb_event_time = None

    windows = []
    t        = t_start
    win_idx  = 0

    while t < t_end_ev:
        t_end_win = t + window_sec

        mask = (
            (event_df["time_since_event_start_world"] >= t) &
            (event_df["time_since_event_start_world"] <  t_end_win)
        )
        win = event_df[mask]

        # Keep partial window (end of event)
        is_partial = int(t_end_win > t_end_ev + 0.05)

        if len(win) == 0:
            t = t_end_win
            win_idx += 1
            continue

        row = {}

        # ── Identifiers ──────────────────────────────────────────────────────
        row["Id"]                = win["Id"].iloc[0]
        row["Condition"]         = win["Condition"].iloc[0] if "Condition" in win.columns else None
        row["Map"]               = win["Map"].iloc[0] if "Map" in win.columns else None
        row["event_id"]          = win["_event_key"].iloc[0]
        row["window_index"]      = win_idx
        row["is_partial_window"] = is_partial

        # ── Window boundaries (in event-time seconds) ─────────────────────────
        row["start_time"]        = round(t, 4)
        row["end_time"]          = round(min(t_end_win, t_end_ev), 4)
        row["window_size_sec"]   = window_sec

        # ── Event context ─────────────────────────────────────────────────────
        if "event_category" in win.columns:
            row["event_category"] = win["event_category"].iloc[0]

        # time_since_event_start = max within window (how far into event we reached)
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
                # positive = approaching the traffic light
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
            # TTC_delta: negative = situation getting more dangerous
            if len(ttc_vals) >= 2:
                row["TTC_delta"] = round(float(ttc_vals.iloc[-1]) - float(ttc_vals.iloc[0]), 4)
            else:
                row["TTC_delta"] = None

        # ── Traffic light state ───────────────────────────────────────────────
        if "TrafficLight" in win.columns:
            row["TrafficLight_prev"]    = win["TrafficLight"].iloc[0]
            row["TrafficLight_current"] = win["TrafficLight"].iloc[-1]
            tl_vals = win["TrafficLight"].dropna().unique().tolist()
            row["yellow_transition"] = int(
                "Yellow" in tl_vals and win["TrafficLight"].iloc[0] != "Yellow"
            )
        # ── Junction phase ────────────────────────────────────────────────────
        if "TrafficLight_JunctionPhase" in win.columns:
            row["prev_phase"]       = win["TrafficLight_JunctionPhase"].iloc[0]
            row["current_phase"]    = win["TrafficLight_JunctionPhase"].iloc[-1]
            row["transition_phase"] = int(row["prev_phase"] != row["current_phase"])

        # ── Target variable ───────────────────────────────────────────────────
        # feedback_next_1s = 1 if the first feedback in this event occurs
        # in the 1-second window AFTER this window ends: (end_time, end_time + 1]
        # Independent of window boundaries.
        t_win_end = row["end_time"]   # actual end (capped at event end if partial)

        if first_fb_event_time is not None:
            in_next_sec = (first_fb_event_time > t_win_end) and \
                          (first_fb_event_time <= t_win_end + 1.0)
            row["feedback_next_1s"] = int(in_next_sec)
        else:
            row["feedback_next_1s"] = 0

        windows.append(row)

        # Stop after the window where target fires — post-feedback windows
        # are not meaningful for prediction and should not be modelled.
        if row["feedback_next_1s"] == 1:
            break

        t = t_end_win
        win_idx += 1

    return windows


# ─── Main loop ───────────────────────────────────────────────────────────────
all_results = {}

for w in WINDOW_SIZES:
    print(f"\n⏱  Building {w}s windows...")
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

    print(f"  Total windows:             {len(result_df):,}")
    print(f"  Unique TL events:          {n_events:,}")
    print(f"  Events with target=1:      {n_with_target:,} / {n_events:,}")
    print(f"  Positive rate (target=1):  {pct_positive:.1f}%")
    print(f"  Partial windows:           {pct_partial:.1f}%")
    print(f"  Avg windows per event:     {len(result_df) / max(n_events, 1):.1f}")

    out_path = os.path.join(OUTPUT_DIR, f"windows_{w}s.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Saved: {out_path}")


# ─── Combined file ────────────────────────────────────────────────────────────
print("\n📦 Combining all window sizes...")
combined = pd.concat(
    [res_df.assign(window_size_sec=w) for w, res_df in all_results.items()],
    ignore_index=True
)
combined_path = os.path.join(OUTPUT_DIR, "windows_ALL_combined.csv")
combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"✅ Combined file saved: {combined_path}")
print(f"   Total rows: {len(combined):,}")

# ─── Output column list ───────────────────────────────────────────────────────
print("\n── Output columns ──")
for col in all_results[1].columns:
    print(f"  {col}")

print("\n✨ Done!")