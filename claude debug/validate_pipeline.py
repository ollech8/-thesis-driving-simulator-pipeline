# -*- coding: utf-8 -*-
"""
Read-only validation pass over ALL_participants_analysis_ALL_conditions.csv.

Does NOT modify the source CSV or the pipeline scripts (all_analysis_functions_Debug.py,
run_analysis_Debug.py). Produces two report files under outputs/:
    - validation_report.csv     one row per check (machine-sortable)
    - validation_summary.md     same data, grouped by column, human-readable

Run from this folder:
    python validate_pipeline.py
"""

import os
import re
import csv
import math
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# -------------------
# CONFIG
# -------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV = os.path.join(
    os.path.dirname(THIS_DIR),
    "ALL_participants_analysis_ALL_conditions.csv",
)
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
REPORT_CSV = os.path.join(OUTPUT_DIR, "validation_report.csv")
SUMMARY_MD = os.path.join(OUTPUT_DIR, "validation_summary.md")

HEADER = [
    "Id", "Condition", "Map", "FrameID", "WorldTime", "SimulationTime",
    "Longitude", "Latitude", "PositionX", "PositionY", "Acceleration", "Speed",
    "SteeringAngle", "Brake", "Braking", "Accelerating", "TurnRight", "TurnLeft",
    "SpacialEvent", "Reason", "BaseEvent", "time_since_event_start_world",
    "event_category", "Overtake", "distance_from_relevant_object",
    "relevant_object_name", "Pedestrian", "TrafficLight",
    "TrafficLight_JunctionPhase", "gap_acceptance", "time_to_collision",
    "ttc_object_name", "text", "speaker", "transcript_type", "comment_flag",
    "start_comment", "first_feedback_in_event",
]
EXTRA_COL = "_extra_39th_field"  # placeholder name for the genuinely unnamed 39th field

REPORT_ROWS = []  # accumulates dict rows matching the schema below


def add_result(column_name, check_id, check_description, check_type, status,
               n_rows_checked, n_rows_failed, example_failing_rows="", notes=""):
    pct_failed = round(100.0 * n_rows_failed / n_rows_checked, 4) if n_rows_checked else 0.0
    REPORT_ROWS.append({
        "column_name": column_name,
        "check_id": check_id,
        "check_description": check_description,
        "check_type": check_type,  # structural | semantic | cross-column
        "status": status,  # PASS | WARN | FAIL
        "n_rows_checked": n_rows_checked,
        "n_rows_failed": n_rows_failed,
        "pct_failed": pct_failed,
        "example_failing_rows": example_failing_rows,
        "notes": notes,
    })


def sample_indices(bool_mask, n=5):
    idx = list(bool_mask[bool_mask].index[:n])
    return "; ".join(str(i) for i in idx)


# -------------------
# STEP 1 — tolerant raw load (row-shape aware, before any typed parsing)
# -------------------
def load_raw_rows_with_field_counts():
    """
    Reads the CSV with the csv module so every row is captured regardless of
    field count (a strict pandas read throws ParserError on this file).
    Returns: list of raw string-field lists, and a Counter of field counts.
    """
    field_count_counter = Counter()
    rows = []
    with open(SOURCE_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            field_count_counter[len(row)] += 1
            rows.append(row)
    return rows, field_count_counter


def build_typed_dataframe(rows):
    """
    Builds a DataFrame where every row is padded/truncated to 39 columns
    (HEADER + EXTRA_COL), so both aligned (38-field) and shifted (39-field)
    rows can be analyzed uniformly. Values are kept as strings; numeric
    coercion happens per-check so a bad value doesn't silently disappear.
    """
    n_cols = len(HEADER) + 1
    padded = []
    for row in rows:
        if len(row) < n_cols:
            row = row + [None] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]
        padded.append(row)
    df = pd.DataFrame(padded, columns=HEADER + [EXTRA_COL])
    for col in df.columns:
        df[col] = df[col].replace("", np.nan)
    return df


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# -------------------
# STEP 2 — structural checks
# -------------------
def check_structural(df, field_count_counter):
    total = len(df)

    aligned = field_count_counter.get(len(HEADER), 0)
    shifted = field_count_counter.get(len(HEADER) + 1, 0)
    malformed = total - aligned - shifted
    add_result(
        "(structural)", "field_count_classification",
        "Classify every raw row by delimited field count: 38=aligned, 39=shifted (column-shift bug), other=malformed",
        "structural",
        "FAIL" if shifted > 0 else "PASS",
        total, shifted + malformed,
        notes=(f"aligned(38 fields)={aligned} ({100*aligned/total:.1f}%), "
               f"shifted(39 fields)={shifted} ({100*shifted/total:.1f}%), "
               f"malformed(other)={malformed} ({100*malformed/total:.1f}%). "
               "Root cause: add_first_feedback_in_event() always adds SpacialEvent_core, "
               "but process_transcription_pipeline()'s empty-transcript branch does not -> "
               "append_output() appends 38-col and 39-col DataFrames into the same CSV with no "
               "schema alignment (all_analysis_functions_Debug.py:2362-2397,2454-2492; "
               "run_analysis_Debug.py:58-60)."),
    )

    ffie_raw = df["first_feedback_in_event"]
    non_null = ffie_raw.dropna()
    numeric_like = non_null.apply(lambda v: str(v).strip() in ("0", "1", "0.0", "1.0"))
    n_non_binary = (~numeric_like).sum()
    value_counts = non_null[~numeric_like].value_counts()
    add_result(
        "first_feedback_in_event", "shift_signature_non_binary_values",
        "first_feedback_in_event should be strictly {0,1,NaN}; any other string value is the column-shift signature (actually a SpacialEvent_core name)",
        "structural",
        "FAIL" if n_non_binary > 0 else "PASS",
        len(non_null), n_non_binary,
        example_failing_rows=sample_indices(~numeric_like.reindex(df.index, fill_value=False)),
        notes=f"{value_counts.shape[0]} distinct non-binary string values found, e.g.: {dict(list(value_counts.items())[:10])}",
    )

    extra_raw = df[EXTRA_COL]
    extra_non_null = extra_raw.dropna()
    extra_numeric = to_numeric(extra_non_null)
    extra_binary_ok = extra_numeric.isin([0, 1])
    add_result(
        EXTRA_COL, "extra_field_is_recoverable_binary_flag",
        "The genuinely unnamed 39th field should be a clean binary flag (0/1) -- this is the true first_feedback_in_event value for shifted rows",
        "structural",
        "PASS" if extra_binary_ok.all() else "WARN",
        len(extra_non_null), int((~extra_binary_ok).sum()),
        notes=f"non-null count={len(extra_non_null)}, distinct values={extra_numeric.value_counts(dropna=False).to_dict()}",
    )

    ids = df["Id"]
    rows_per_id = ids.value_counts()
    add_result(
        "Id", "row_count_sanity",
        "Report total rows and rows-per-participant distribution to spot truncated/duplicated trips",
        "structural", "PASS" if rows_per_id.min() > 0 else "WARN",
        total, 0,
        notes=(f"total_rows={total}, unique_participants={ids.nunique()}, "
               f"rows_per_id min={rows_per_id.min()} max={rows_per_id.max()} "
               f"mean={rows_per_id.mean():.0f} std={rows_per_id.std():.0f}"),
    )


# -------------------
# STEP 3 — per-column semantic checks
# -------------------
def per_trip_groups(df):
    return df.groupby(["Id", "Condition"], sort=False)


def check_Id(df):
    pattern = re.compile(r"^C\d+_\d+$")
    ids = df["Id"].dropna().astype(str)
    bad = ~ids.apply(lambda s: bool(pattern.match(s.strip())))
    add_result("Id", "id_format", "Id must match ^C<digits>_<digits>$", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(ids), int(bad.sum()),
               example_failing_rows=sample_indices(bad.reindex(df.index, fill_value=False)))


def check_Condition(df):
    allowed = {"Conventional", "Avatar", "Remote"}
    vals = df["Condition"].dropna().astype(str)
    bad = ~vals.isin(allowed)
    add_result("Condition", "condition_enum", f"Condition must be one of {sorted(allowed)}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")


def check_Map(df):
    allowed = {"A", "B", "C"}
    vals = df["Map"].dropna().astype(str)
    bad = ~vals.isin(allowed)
    add_result("Map", "map_enum", f"Map must be one of {sorted(allowed)}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")


def check_FrameID(df):
    frame_id = to_numeric(df["FrameID"])
    sim_time = to_numeric(df["SimulationTime"])
    n_bad = 0
    n_checked = 0
    for _, g in per_trip_groups(df):
        g_sorted = g.assign(_ft=to_numeric(g["FrameID"]), _st=to_numeric(g["SimulationTime"])).sort_values("_st")
        diffs = g_sorted["_ft"].diff().dropna()
        n_checked += len(diffs)
        n_bad += int((diffs < 0).sum())
    add_result("FrameID", "monotonic_within_trip",
               "FrameID should be mostly non-decreasing within a trip once sorted by SimulationTime",
               "semantic", "PASS" if n_bad == 0 else "WARN", n_checked, n_bad,
               notes=f"non-null FrameID rows={frame_id.notna().sum()}/{len(df)}, dup rate n/a (frame ids repeat across GPS/telemetry merge by design)")


def check_WorldTime(df):
    vals = df["WorldTime"].dropna().astype(str)
    pattern = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")
    bad = ~vals.apply(lambda s: bool(pattern.match(s.strip())))
    add_result("WorldTime", "parseable_timestamp", "WorldTime must match HH:MM:SS[.ffffff]", "semantic",
               "PASS" if bad.sum() == 0 else "WARN", len(vals), int(bad.sum()),
               example_failing_rows=sample_indices(bad.reindex(df.index, fill_value=False)))


def check_SimulationTime(df):
    n_bad = 0
    n_checked = 0
    tol = -0.01
    for _, g in per_trip_groups(df):
        st = to_numeric(g["SimulationTime"]).sort_index()
        # sort by original row order (assumed chronological within a trip's append)
        diffs = st.diff().dropna()
        n_checked += len(diffs)
        n_bad += int((diffs < tol).sum())
    add_result("SimulationTime", "monotonic_within_trip",
               "SimulationTime should be non-decreasing within (Id,Condition) (primary join key across the pipeline)",
               "semantic", "PASS" if n_bad == 0 else "WARN", n_checked, n_bad)


def check_lon_lat_position(df):
    for col in ["Longitude", "Latitude", "PositionX", "PositionY"]:
        vals = to_numeric(df[col])
        non_null = vals.dropna()
        if non_null.empty:
            continue
        lo, hi = non_null.quantile(0.01), non_null.quantile(0.99)
        outside = (non_null < lo - abs(lo) * 2) | (non_null > hi + abs(hi) * 2)
        add_result(col, "empirical_bounding_box",
                   f"{col} should fall within an empirically plausible range (per simulator, not real-world geo bounds)",
                   "semantic", "PASS" if outside.sum() == 0 else "WARN", len(non_null), int(outside.sum()),
                   notes=f"min={non_null.min():.6g}, max={non_null.max():.6g}, p1={lo:.6g}, p99={hi:.6g}, missing={vals.isna().sum()}")

    lon = to_numeric(df["Longitude"])
    lat = to_numeric(df["Latitude"])
    px = to_numeric(df["PositionX"])
    py = to_numeric(df["PositionY"])
    corr_rows = []
    for _, g in per_trip_groups(df):
        idx = g.index
        d_lon = lon.loc[idx].diff()
        d_px = px.loc[idx].diff()
        d_lat = lat.loc[idx].diff()
        d_py = py.loc[idx].diff()
        try:
            c1 = pd.concat([d_lon, d_px], axis=1).dropna().corr().iloc[0, 1]
            c2 = pd.concat([d_lat, d_py], axis=1).dropna().corr().iloc[0, 1]
            if not (math.isnan(c1) or math.isnan(c2)):
                corr_rows.append(min(abs(c1), abs(c2)))
        except Exception:
            continue
    if corr_rows:
        low_corr = sum(1 for c in corr_rows if c < 0.8)
        add_result("Longitude/Latitude vs PositionX/Y", "gps_position_correlation",
                   "Per-trip, delta GPS and delta Position should correlate strongly (both derive from the same underlying vehicle position)",
                   "cross-column", "PASS" if low_corr == 0 else "WARN", len(corr_rows), low_corr,
                   notes=f"min |corr| observed across trips: {min(corr_rows):.3f}")


def check_Acceleration(df):
    vals = to_numeric(df["Acceleration"])
    non_null = vals.dropna()
    bad = non_null < 0
    add_result("Acceleration", "non_negative_magnitude",
               "Acceleration is a vector magnitude sqrt(x^2+y^2+z^2) by construction -> must be >= 0",
               "semantic", "PASS" if bad.sum() == 0 else "FAIL", len(non_null), int(bad.sum()),
               example_failing_rows=sample_indices(bad.reindex(df.index, fill_value=False)))


def check_Speed(df):
    vals = to_numeric(df["Speed"])
    non_null = vals.dropna()
    negative = non_null < -1e-6
    near_zero_nonzero = (non_null > 0) & (non_null < 1e-4)
    hi_cut = non_null.quantile(0.999)
    implausible = non_null > max(hi_cut * 3, 50)
    add_result("Speed", "non_negative", "Speed should be >= 0 (allow tiny float noise)", "semantic",
               "PASS" if negative.sum() == 0 else "WARN", len(non_null), int(negative.sum()),
               notes=f"min={non_null.min():.6g}")
    add_result("Speed", "near_zero_nonzero_flag",
               "Rows with 0 < Speed < 1e-4 are the direct denominator-noise source for time_to_collision blow-ups (relative_movement dead-code bug)",
               "semantic", "WARN" if near_zero_nonzero.sum() > 0 else "PASS", len(non_null), int(near_zero_nonzero.sum()),
               notes="Quantifies exposure only; TTC fix itself is out of scope for this pass")
    add_result("Speed", "implausible_high", "Flag speeds far beyond the 99.9th percentile as implausible for manual review",
               "semantic", "PASS" if implausible.sum() == 0 else "WARN", len(non_null), int(implausible.sum()),
               notes=f"p99.9={hi_cut:.3g}, max={non_null.max():.3g}")


def check_SteeringAngle(df):
    vals = to_numeric(df["SteeringAngle"]).dropna()
    add_result("SteeringAngle", "range_report",
               "No hardcoded bound found in the pipeline code; report distribution for manual scale confirmation",
               "semantic", "WARN", len(vals), 0,
               notes=f"min={vals.min():.4g}, max={vals.max():.4g}, p1={vals.quantile(0.01):.4g}, p99={vals.quantile(0.99):.4g}")


def check_Brake(df):
    vals = to_numeric(df["Brake"]).dropna()
    negative = vals < 0
    above_one = vals > 1
    status = "FAIL" if negative.sum() > 0 else ("WARN" if above_one.sum() > 0 else "PASS")
    add_result("Brake", "pedal_range", "Brake likely a [0,1] pedal-pressure value; negative is a hard fail, >1 is a scale warning",
               "semantic", status, len(vals), int(negative.sum() + above_one.sum()),
               notes=f"min={vals.min():.4g}, max={vals.max():.4g}")


def check_binary_columns(df):
    for col in ["Braking", "Accelerating", "TurnRight", "TurnLeft", "comment_flag", "start_comment"]:
        raw = df[col].dropna()
        numeric = to_numeric(raw)
        bad = ~numeric.isin([0, 1])
        add_result(col, "strict_binary", f"{col} must be strictly {{0,1}}", "semantic",
                   "PASS" if bad.sum() == 0 else "FAIL", len(raw), int(bad.sum()),
                   notes=f"value_counts={numeric.value_counts(dropna=False).to_dict()}")


def check_SpacialEvent(df):
    vals = df["SpacialEvent"].dropna().astype(str)
    lowered = vals.str.strip().str.lower()
    case_variants = vals.groupby(lowered).nunique()
    n_case_dupe_groups = int((case_variants > 1).sum())
    literal_nan = vals.str.strip().str.lower().isin(["nan", "none"])
    add_result("SpacialEvent", "frequency_and_case_variants",
               "Frequency table for manual review; flag case-variant duplicates and literal 'nan'/'none' strings (vs true null)",
               "semantic", "WARN" if (n_case_dupe_groups > 0 or literal_nan.sum() > 0) else "PASS",
               len(vals), int(literal_nan.sum()),
               notes=f"top_values={vals.value_counts().head(10).to_dict()}, case_variant_groups={n_case_dupe_groups}, literal_nan_or_none_strings={int(literal_nan.sum())}")


def check_Reason(df):
    reason = df["Reason"]
    spacial_event = df["SpacialEvent"]
    bad = reason.notna() & (spacial_event.astype(str).str.strip() != "Termination")
    add_result("Reason", "only_on_termination", "Reason must be non-null only where SpacialEvent == 'Termination'",
               "cross-column", "PASS" if bad.sum() == 0 else "FAIL", int(reason.notna().sum()), int(bad.sum()),
               example_failing_rows=sample_indices(bad))


_SENTINELS = ("StartPoint", "Start", "None", "Termination", "EndPoint")


def clean_event_name_reimpl(event):
    """Faithful re-implementation of clean_event_name() (all_analysis_functions_Debug.py:466-477)."""
    if pd.isna(event):
        return event
    event = str(event)
    if event in _SENTINELS:
        return event  # sentinel check is case-sensitive and pre-lowering, exactly as in production
    event = event.lower().strip()
    event = re.sub(r"\b(start|end)\b", "", event)
    event = re.sub(r"[^a-zA-Z0-9\s]", "", event)
    event = re.sub(r"\s+", " ", event)
    return event.strip()


def check_BaseEvent(df):
    spacial = df["SpacialEvent"]
    base = df["BaseEvent"].astype(str)
    recomputed = spacial.apply(clean_event_name_reimpl).astype(str)
    both_present = df["BaseEvent"].notna() & spacial.notna()
    mismatch = both_present & (base != recomputed)
    stale_none = mismatch & (base.str.strip() == "None")
    add_result("BaseEvent", "deterministic_function_of_spacialevent",
               "BaseEvent should equal clean_event_name(SpacialEvent) exactly (faithful re-implementation of "
               "all_analysis_functions_Debug.py:466-477, including its case-sensitive sentinel list). ROOT CAUSE "
               "of most mismatches: process_spacial_events() (line 751-772) calls prepare_spacial_events() -- which "
               "computes BaseEvent from SpacialEvent -- BEFORE fill_intermediate_events() fills in SpacialEvent for "
               "rows between paired start/end events. So BaseEvent is stale ('None') for every row whose "
               "SpacialEvent value was only populated by the later fill step; it was never recomputed after the fill.",
               "cross-column", "PASS" if mismatch.sum() == 0 else "FAIL",
               int(both_present.sum()), int(mismatch.sum()),
               example_failing_rows=sample_indices(mismatch),
               notes=f"of {int(mismatch.sum())} mismatches, {int(stale_none.sum())} are the stale-'None' pattern "
                     "(BaseEvent never recomputed after SpacialEvent was filled); remainder may be other cases, "
                     "spot-check before assuming they're all the same root cause")


def check_time_since_event_start_world(df):
    vals = to_numeric(df["time_since_event_start_world"]).dropna()
    negative = vals < 0
    add_result("time_since_event_start_world", "non_negative",
               "Should be >= 0 where present (elapsed time since event start)", "semantic",
               "PASS" if negative.sum() == 0 else "FAIL", len(vals), int(negative.sum()))


def check_event_category(df):
    allowed = {"SimulationPoints", "Pedestrians", "Overtake", "TrafficLights", "GapAcceptance"}
    vals = df["event_category"].dropna().astype(str)
    bad = ~vals.isin(allowed)
    add_result("event_category", "enum_by_construction",
               f"categorize_event() can only return one of {sorted(allowed)} or None -- any other value implies function drift",
               "semantic", "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")


def check_Overtake_column(df):
    allowed = {"start drive", "start brake", "stop", "restart drive"}
    vals = df["Overtake"].dropna().astype(str).str.strip()
    bad = ~vals.isin(allowed)
    add_result("Overtake", "enum_phase_labels",
               f"Overtake (sudden-stop phase label) must be one of {sorted(allowed)}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")
    dup_counts = df.assign(_ov=df["Overtake"]).dropna(subset=["_ov"]).groupby(["Id", "Condition", "_ov"]).size()
    dup = dup_counts[dup_counts > 1]
    add_result("Overtake", "no_duplicate_labels_per_trip",
               "Each phase label should appear at most once per (Id,Condition) trip", "cross-column",
               "PASS" if len(dup) == 0 else "WARN", len(dup_counts), len(dup))


def check_distance_and_relevant_object(df):
    dist = to_numeric(df["distance_from_relevant_object"])
    obj = df["relevant_object_name"]
    mismatch = dist.notna() != obj.notna()
    add_result("distance_from_relevant_object / relevant_object_name", "null_pattern_consistency",
               "distance_from_relevant_object must be non-null iff relevant_object_name is non-null", "cross-column",
               "PASS" if mismatch.sum() == 0 else "FAIL", len(df), int(mismatch.sum()),
               example_failing_rows=sample_indices(mismatch))
    negative = dist.dropna() < 0
    add_result("distance_from_relevant_object", "non_negative", "Haversine distance must be >= 0", "semantic",
               "PASS" if negative.sum() == 0 else "FAIL", int(dist.notna().sum()), int(negative.sum()))
    names = obj.dropna().astype(str)
    add_result("relevant_object_name", "frequency_report", "Frequency table for manual sanity check of naming conventions",
               "semantic", "PASS", len(names), 0, notes=f"top_values={names.value_counts().head(10).to_dict()}")


def check_Pedestrian(df):
    vals = df["Pedestrian"].dropna().astype(str)
    pattern = re.compile(
        r"^(start walking|start crossing|end crossing): .+"
        r"( \| (start walking|start crossing|end crossing): .+)*$"
    )
    bad = ~vals.apply(lambda s: bool(pattern.match(s.strip())))
    add_result("Pedestrian", "compound_marker_format",
               "Non-null values must match '<start walking|start crossing|end crossing>: <name>' possibly ' | '-joined",
               "semantic", "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               example_failing_rows=sample_indices(bad.reindex(df.index, fill_value=False)),
               notes=f"non-null count={len(vals)}/{len(df)} ({100*len(vals)/len(df):.2f}%)")

    # Walker2 / Avatar gap flag (quantify only, no fix).
    # NOTE: must match the standalone "Walker2" stage token exactly -- a substring/contains
    # match would wrongly match "section walker2" (a common, unrelated ~14k-row stage).
    is_pedestrian_event = df["event_category"] == "Pedestrians"
    for cond in sorted(df["Condition"].dropna().unique()):
        sub = df[(df["Condition"] == cond) & is_pedestrian_event]
        spacial_vals = sub["SpacialEvent"].dropna().astype(str)
        has_walker2 = spacial_vals.str.strip().str.fullmatch("Walker2", case=False).fillna(False).any()
        add_result("SpacialEvent (Pedestrians category)", f"walker2_presence_{cond}",
                   f"Count of Pedestrians-category rows and presence of a Walker2 stage for Condition={cond}",
                   "semantic", "PASS" if has_walker2 else "WARN", len(sub), 0 if has_walker2 else len(sub),
                   notes=f"n_pedestrian_rows={len(sub)}, distinct_SpacialEvent={sorted(spacial_vals.unique().tolist())}, "
                         f"has_walker2_stage={has_walker2}. Root-cause investigation (design difference vs "
                         "object-matching bug in find_matching_objects) deferred to a follow-up pass.")


def check_TrafficLight(df):
    vals = df["TrafficLight"].dropna().astype(str)
    add_result("TrafficLight", "vocabulary_report",
               "True CurrentState vocabulary lives in raw per-participant objects JSON, not derivable from this CSV alone -- reporting observed values instead of assuming Red/Yellow/Green",
               "semantic", "WARN", len(vals), 0, notes=f"value_counts={vals.value_counts().to_dict()}")
    mismatch = df["TrafficLight"].notna() & (df["event_category"] != "TrafficLights")
    add_result("TrafficLight", "category_alignment",
               "TrafficLight is merged by relevant_object_name across ALL rows in the pipeline, not filtered to event_category=='TrafficLights' -- flagging as WARN, not FAIL, since this may be expected incidental matching",
               "cross-column", "PASS" if mismatch.sum() == 0 else "WARN", len(df), int(mismatch.sum()),
               example_failing_rows=sample_indices(mismatch))


def check_TrafficLight_JunctionPhase(df):
    allowed = ["Approaching", "AtStopLine", "InsideJunction", "LeavingJunction"]
    vals = df["TrafficLight_JunctionPhase"].dropna().astype(str)
    bad = ~vals.isin(allowed)
    add_result("TrafficLight_JunctionPhase", "enum", f"Must be one of {allowed}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")

    order = {name: i for i, name in enumerate(allowed)}
    n_checked = 0
    n_bad = 0
    for _, g in per_trip_groups(df):
        phases = g["TrafficLight_JunctionPhase"].dropna()
        if len(phases) < 2:
            continue
        ranks = phases.map(order)
        n_checked += len(ranks) - 1
        n_bad += int((ranks.diff().dropna() < 0).sum())
    add_result("TrafficLight_JunctionPhase", "forward_only_phase_order",
               "Within a trip, phases should progress Approaching -> AtStopLine -> InsideJunction -> LeavingJunction (no backward transitions)",
               "cross-column", "PASS" if n_bad == 0 else "WARN", n_checked, n_bad)


def check_gap_acceptance(df):
    vals = to_numeric(df["gap_acceptance"]).dropna()
    negative = vals < 0
    add_result("gap_acceptance", "non_negative", "gap_acceptance = abs(dist2-dist1) so must be >= 0", "semantic",
               "PASS" if negative.sum() == 0 else "FAIL", len(vals), int(negative.sum()))

    non_null_mask = df["gap_acceptance"].notna() & to_numeric(df["gap_acceptance"]).notna()
    pattern = re.compile(r"(?i)gap\s*acceptance")
    spacial = df.loc[non_null_mask, "SpacialEvent"].astype(str)
    matches = spacial.apply(lambda s: bool(pattern.search(s)))
    add_result("gap_acceptance", "spacialevent_alignment",
               "Non-null gap_acceptance rows should have a SpacialEvent containing 'gap acceptance'", "cross-column",
               "PASS" if (~matches).sum() == 0 else "WARN", len(spacial), int((~matches).sum()))


def check_time_to_collision(df):
    ttc = to_numeric(df["time_to_collision"])
    non_null = ttc.dropna()
    ceiling = 10000.0
    extreme = non_null > ceiling
    add_result("time_to_collision", "sanity_ceiling",
               f"Flag values above a non-physical ceiling ({ceiling}s) for a driving-sim scenario", "semantic",
               "FAIL" if extreme.sum() > 0 else "PASS", len(non_null), int(extreme.sum()),
               example_failing_rows=sample_indices(extreme.reindex(df.index, fill_value=False)),
               notes=f"max={non_null.max():.6g}" if len(non_null) else "no non-null values")

    speed = to_numeric(df["Speed"])
    near_zero_speed = (speed > 0) & (speed < 1e-4)
    ttc_with_near_zero_speed = ttc.notna() & near_zero_speed
    add_result("time_to_collision", "relative_movement_deadcode_exposure",
               "Rows where time_to_collision was computed while Speed is near-zero-nonzero: known root cause is "
               "calculate_time_to_collision_with_police() reading a 'relative_movement' field that df_unique_events_objects "
               "never has, so relative_speed always falls back to raw ego Speed (dead code for same/opposite/crossing "
               "branches) -- quantifying exposure only, fix deferred to a follow-up pass",
               "semantic", "WARN" if ttc_with_near_zero_speed.sum() > 0 else "PASS",
               int(ttc.notna().sum()), int(ttc_with_near_zero_speed.sum()))


def check_ttc_object_name(df):
    ttc = df["time_to_collision"]
    name = df["ttc_object_name"]
    mismatch = ttc.notna() != name.notna()
    add_result("ttc_object_name", "null_pattern_consistency",
               "ttc_object_name must be non-null iff time_to_collision is non-null", "cross-column",
               "PASS" if mismatch.sum() == 0 else "FAIL", len(df), int(mismatch.sum()),
               example_failing_rows=sample_indices(mismatch))


def check_text_encoding(df):
    vals = df["text"].dropna().astype(str)
    has_replacement_char = vals.str.contains("�", na=False)
    literal_nan = vals.str.strip().str.lower().isin(["nan", ""])
    add_result("text", "encoding_and_literal_nan",
               "Check for UTF-8 mojibake (replacement char) and literal 'nan'/empty strings masquerading as null",
               "semantic", "WARN" if (has_replacement_char.sum() + literal_nan.sum()) > 0 else "PASS",
               len(vals), int(has_replacement_char.sum() + literal_nan.sum()),
               notes=f"replacement_char_rows={int(has_replacement_char.sum())}, literal_nan_or_empty={int(literal_nan.sum())}")


def check_speaker(df):
    vals = df["speaker"].dropna().astype(str)
    stripped_lower = vals.str.strip().str.lower()
    variant_groups = vals.groupby(stripped_lower).nunique()
    n_variant_groups = int((variant_groups > 1).sum())
    literal_nan = stripped_lower.isin(["nan"])
    add_result("speaker", "frequency_and_whitespace_variants",
               "Frequency table; flag whitespace/case-variant duplicates and literal 'nan' strings",
               "semantic", "WARN" if (n_variant_groups > 0 or literal_nan.sum() > 0) else "PASS",
               len(vals), int(literal_nan.sum()),
               notes=f"top_values={vals.value_counts().head(10).to_dict()}, variant_groups={n_variant_groups}, literal_nan={int(literal_nan.sum())}")

    text_notna = df["text"].notna()
    speaker_notna = df["speaker"].notna()
    mismatch = speaker_notna & ~text_notna
    add_result("speaker", "null_iff_text_null",
               "speaker should be null wherever text is null", "cross-column",
               "PASS" if mismatch.sum() == 0 else "WARN", len(df), int(mismatch.sum()))


def check_transcript_type(df):
    allowed = {"manual", "whisper"}
    vals = df["transcript_type"].dropna().astype(str)
    bad = ~vals.isin(allowed)
    add_result("transcript_type", "enum", f"Must be one of {sorted(allowed)}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()),
               notes=f"value_counts={vals.value_counts().to_dict()}")

    mixed_trips = 0
    n_trips = 0
    for _, g in per_trip_groups(df):
        n_trips += 1
        types = set(g["transcript_type"].dropna().unique())
        if len(types) > 1:
            mixed_trips += 1
    add_result("transcript_type", "single_type_per_trip",
               "A trip should never mix 'manual' and 'whisper' (selection happens once per trip at load time)",
               "cross-column", "PASS" if mixed_trips == 0 else "FAIL", n_trips, mixed_trips)


def add_comment_flag_reimpl(text_val):
    if pd.isna(text_val):
        return 0
    return 1 if str(text_val).strip() != "" else 0


def check_comment_flag(df):
    recomputed = df["text"].apply(add_comment_flag_reimpl)
    stored = to_numeric(df["comment_flag"]).fillna(-1).astype(int)
    mismatch = recomputed != stored
    add_result("comment_flag", "deterministic_function_of_text",
               "comment_flag should equal an independently recomputed function of text (1 iff text non-empty/non-null)",
               "cross-column", "PASS" if mismatch.sum() == 0 else "FAIL", len(df), int(mismatch.sum()),
               example_failing_rows=sample_indices(mismatch))


def check_start_comment(df):
    vals = df["start_comment"].dropna()
    numeric = to_numeric(vals)
    bad = ~numeric.isin([0, 1])
    add_result("start_comment", "strict_binary", "start_comment must be strictly {0,1}", "semantic",
               "PASS" if bad.sum() == 0 else "FAIL", len(vals), int(bad.sum()))


def check_first_feedback_in_event_quantified(df):
    ffie = df["first_feedback_in_event"].dropna()
    numeric_like = ffie.apply(lambda v: str(v).strip() in ("0", "1", "0.0", "1.0"))
    add_result("first_feedback_in_event", "current_state_quantified",
               "As-is, this column is corrupted by the column-shift bug (see structural checks); reported here as FAIL "
               "to keep the per-column table complete. The fix (realigning shifted rows using the 39th field) is a "
               "separate follow-up pass, not part of this read-only validation.",
               "semantic", "FAIL" if (~numeric_like).sum() > 0 else "PASS",
               len(ffie), int((~numeric_like).sum()))


# -------------------
# MAIN
# -------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading source CSV (read-only): {SOURCE_CSV}")
    rows, field_count_counter = load_raw_rows_with_field_counts()
    print(f"  {len(rows)} data rows read. Field-count distribution: {dict(field_count_counter)}")

    df = build_typed_dataframe(rows)

    check_structural(df, field_count_counter)

    check_Id(df)
    check_Condition(df)
    check_Map(df)
    check_FrameID(df)
    check_WorldTime(df)
    check_SimulationTime(df)
    check_lon_lat_position(df)
    check_Acceleration(df)
    check_Speed(df)
    check_SteeringAngle(df)
    check_Brake(df)
    check_binary_columns(df)
    check_SpacialEvent(df)
    check_Reason(df)
    check_BaseEvent(df)
    check_time_since_event_start_world(df)
    check_event_category(df)
    check_Overtake_column(df)
    check_distance_and_relevant_object(df)
    check_Pedestrian(df)
    check_TrafficLight(df)
    check_TrafficLight_JunctionPhase(df)
    check_gap_acceptance(df)
    check_time_to_collision(df)
    check_ttc_object_name(df)
    check_text_encoding(df)
    check_speaker(df)
    check_transcript_type(df)
    check_comment_flag(df)
    check_start_comment(df)
    check_first_feedback_in_event_quantified(df)

    report_df = pd.DataFrame(REPORT_ROWS)
    report_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {REPORT_CSV} ({len(report_df)} check rows)")

    write_summary_md(report_df, df, field_count_counter)
    print(f"Wrote {SUMMARY_MD}")


def write_summary_md(report_df, df, field_count_counter):
    total = len(df)
    aligned = field_count_counter.get(len(HEADER), 0)
    shifted = field_count_counter.get(len(HEADER) + 1, 0)
    status_counts = report_df["status"].value_counts().to_dict()

    lines = []
    lines.append("# Validation Summary — ALL_participants_analysis_ALL_conditions.csv\n")
    lines.append("## Executive summary\n")
    lines.append(f"- Total data rows: **{total}**")
    lines.append(f"- Aligned rows (38 fields): **{aligned}** ({100*aligned/total:.1f}%)")
    lines.append(f"- Shifted rows (39 fields, column-shift bug): **{shifted}** ({100*shifted/total:.1f}%)")
    lines.append(f"- Check verdicts: {status_counts}")
    lines.append("")
    lines.append("This report is read-only: the source CSV and pipeline scripts were not modified. "
                  "Fixing the column-shift bug, and investigating the Walker2/Avatar gap and the "
                  "time_to_collision `relative_movement` dead-code issue, are deferred to follow-up passes.\n")

    lines.append("## Per-column detail\n")
    for col, group in report_df.groupby("column_name", sort=False):
        lines.append(f"### {col}\n")
        for _, r in group.iterrows():
            lines.append(f"- **[{r['status']}]** `{r['check_id']}` ({r['check_type']}): {r['check_description']}")
            lines.append(f"  - checked={r['n_rows_checked']}, failed={r['n_rows_failed']} ({r['pct_failed']}%)")
            if r["notes"]:
                lines.append(f"  - notes: {r['notes']}")
            if r["example_failing_rows"]:
                lines.append(f"  - example failing row indices: {r['example_failing_rows']}")
        lines.append("")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
