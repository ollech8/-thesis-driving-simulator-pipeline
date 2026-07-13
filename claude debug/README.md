# Claude debug — validation pass

This folder contains a **read-only** validation pass over
`ALL_participants_analysis_ALL_conditions.csv`. It does not modify that file,
nor `all_analysis_functions_Debug.py`, nor `run_analysis_Debug.py`.

## How to run

```
cd "claude debug"
python validate_pipeline.py
```

It reads the CSV from the parent folder, runs a set of checks, and writes two
files into `outputs/`:

- **`validation_report.csv`** — one row per check, with columns:
  `column_name, check_id, check_description, check_type, status, n_rows_checked,
  n_rows_failed, pct_failed, example_failing_rows, notes`.
  `check_type` is one of `structural` / `semantic` / `cross-column`.
  `status` is `PASS`, `WARN`, or `FAIL`. Open this in Excel/pandas to sort/filter.

- **`validation_summary.md`** — the same data, grouped by column, with an
  executive summary at the top (total rows, % of rows affected by the
  column-shift bug, and a tally of PASS/WARN/FAIL across all checks).

## What this pass covers

1. **Structural integrity** — quantifies the column-shift bug: `add_first_feedback_in_event()`
   always adds a `SpacialEvent_core` column, but the empty-transcript branch of
   `process_transcription_pipeline()` doesn't, so `append_output()` in
   `run_analysis_Debug.py` silently appends 38-column and 39-column trip
   DataFrames into the same CSV with no schema alignment. ~61% of rows end up
   shifted one column to the right starting at `first_feedback_in_event`.
2. **Per-column semantic checks** — for all 38 header columns: value-domain
   checks (enums, binary columns, non-negativity), monotonicity checks
   (`SimulationTime`, `FrameID`, `TrafficLight_JunctionPhase` order), and
   cross-column consistency checks (e.g. `distance_from_relevant_object`
   non-null iff `relevant_object_name` non-null; `BaseEvent` should equal a
   recomputed function of `SpacialEvent`; `comment_flag` should equal a
   recomputed function of `text`).
3. **Flag-only quantification of two known issues** (no fix attempted here):
   - The Avatar condition has zero `Walker2`-stage rows under
     `event_category=="Pedestrians"`, unlike Conventional/Remote.
   - `time_to_collision` computations that used the generic branch of
     `calculate_time_to_collision_with_police()` are exposed to a dead-code bug
     (`relative_movement` is read from a table that never has that column, so
     relative speed always falls back to raw ego `Speed`) — this pass reports
     how many `time_to_collision` values coincide with near-zero `Speed`
     (the direct cause of extreme outliers), without attempting a fix.

## Deferred to follow-up passes (not built yet)

- `fix_structural_issues.py` — repairs the column shift by using the 39th
  (currently unnamed) field as the true `first_feedback_in_event` value and
  recovering `SpacialEvent_core` as a legitimate new column; writes a
  **new** cleaned CSV into this folder (never overwrites the original).
- `investigate_walker2_avatar.py` — traces whether the Avatar `Walker2` gap is
  a genuine scenario-design difference or an object-matching bug (checking
  raw objects JSON and `objectPoints.csv` against `find_matching_objects()`).
- `investigate_ttc_outliers.py` — confirms the `relative_movement` dead-code
  mechanism against the actual extreme-value rows.

Build these once the validation report above has been reviewed.
