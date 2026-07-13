# Validation Summary — ALL_participants_analysis_ALL_conditions.csv

## Executive summary

- Total data rows: **469849**
- Aligned rows (38 fields): **183282** (39.0%)
- Shifted rows (39 fields, column-shift bug): **286567** (61.0%)
- Check verdicts: {'PASS': 39, 'WARN': 14, 'FAIL': 5}

This report is read-only: the source CSV and pipeline scripts were not modified. Fixing the column-shift bug, and investigating the Walker2/Avatar gap and the time_to_collision `relative_movement` dead-code issue, are deferred to follow-up passes.

## Per-column detail

### (structural)

- **[FAIL]** `field_count_classification` (structural): Classify every raw row by delimited field count: 38=aligned, 39=shifted (column-shift bug), other=malformed
  - checked=469849, failed=286567 (60.9913%)
  - notes: aligned(38 fields)=183282 (39.0%), shifted(39 fields)=286567 (61.0%), malformed(other)=0 (0.0%). Root cause: add_first_feedback_in_event() always adds SpacialEvent_core, but process_transcription_pipeline()'s empty-transcript branch does not -> append_output() appends 38-col and 39-col DataFrames into the same CSV with no schema alignment (all_analysis_functions_Debug.py:2362-2397,2454-2492; run_analysis_Debug.py:58-60).

### first_feedback_in_event

- **[FAIL]** `shift_signature_non_binary_values` (structural): first_feedback_in_event should be strictly {0,1,NaN}; any other string value is the column-shift signature (actually a SpacialEvent_core name)
  - checked=286567, failed=286567 (100.0%)
  - notes: 16 distinct non-binary string values found, e.g.: {'none': 141424, 'traffic light 1': 33114, 'egocar gap acceptance': 25532, 'overtake': 22875, 'traffic light 3': 21045, 'traffic light 2': 13713, 'section walker1': 9057, 'section walker2': 8669, 'gap acceptance': 4539, 'section walker4': 3430}
  - example failing row indices: 0; 1; 2; 3; 4
- **[FAIL]** `current_state_quantified` (semantic): As-is, this column is corrupted by the column-shift bug (see structural checks); reported here as FAIL to keep the per-column table complete. The fix (realigning shifted rows using the 39th field) is a separate follow-up pass, not part of this read-only validation.
  - checked=286567, failed=286567 (100.0%)

### _extra_39th_field

- **[PASS]** `extra_field_is_recoverable_binary_flag` (structural): The genuinely unnamed 39th field should be a clean binary flag (0/1) -- this is the true first_feedback_in_event value for shifted rows
  - checked=260587, failed=0 (0.0%)
  - notes: non-null count=260587, distinct values={0: 260331, 1: 256}

### Id

- **[PASS]** `row_count_sanity` (structural): Report total rows and rows-per-participant distribution to spot truncated/duplicated trips
  - checked=469849, failed=0 (0.0%)
  - notes: total_rows=469849, unique_participants=39, rows_per_id min=4317 max=23230 mean=12047 std=4843
- **[PASS]** `id_format` (semantic): Id must match ^C<digits>_<digits>$
  - checked=469849, failed=0 (0.0%)

### Condition

- **[PASS]** `condition_enum` (semantic): Condition must be one of ['Avatar', 'Conventional', 'Remote']
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={'Conventional': 170681, 'Avatar': 152434, 'Remote': 146734}

### Map

- **[PASS]** `map_enum` (semantic): Map must be one of ['A', 'B', 'C']
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={'A': 192766, 'C': 145946, 'B': 131137}

### FrameID

- **[WARN]** `monotonic_within_trip` (semantic): FrameID should be mostly non-decreasing within a trip once sorted by SimulationTime
  - checked=469762, failed=10410 (2.216%)
  - notes: non-null FrameID rows=469849/469849, dup rate n/a (frame ids repeat across GPS/telemetry merge by design)

### WorldTime

- **[PASS]** `parseable_timestamp` (semantic): WorldTime must match HH:MM:SS[.ffffff]
  - checked=469849, failed=0 (0.0%)

### SimulationTime

- **[PASS]** `monotonic_within_trip` (semantic): SimulationTime should be non-decreasing within (Id,Condition) (primary join key across the pipeline)
  - checked=469762, failed=0 (0.0%)

### Longitude

- **[PASS]** `empirical_bounding_box` (semantic): Longitude should fall within an empirically plausible range (per simulator, not real-world geo bounds)
  - checked=469762, failed=0 (0.0%)
  - notes: min=0.00172393, max=0.00489228, p1=0.00174942, p99=0.00479257, missing=87

### Latitude

- **[PASS]** `empirical_bounding_box` (semantic): Latitude should fall within an empirically plausible range (per simulator, not real-world geo bounds)
  - checked=469762, failed=0 (0.0%)
  - notes: min=-0.00447022, max=-0.00143434, p1=-0.00445812, p99=-0.00147686, missing=87

### PositionX

- **[PASS]** `empirical_bounding_box` (semantic): PositionX should fall within an empirically plausible range (per simulator, not real-world geo bounds)
  - checked=469762, failed=0 (0.0%)
  - notes: min=-5.25415, max=347.445, p1=-2.41597, p99=336.341, missing=87

### PositionY

- **[PASS]** `empirical_bounding_box` (semantic): PositionY should fall within an empirically plausible range (per simulator, not real-world geo bounds)
  - checked=469762, failed=0 (0.0%)
  - notes: min=-329.952, max=332.236, p1=-327.335, p99=330.888, missing=87

### Longitude/Latitude vs PositionX/Y

- **[WARN]** `gps_position_correlation` (cross-column): Per-trip, delta GPS and delta Position should correlate strongly (both derive from the same underlying vehicle position)
  - checked=87, failed=3 (3.4483%)
  - notes: min |corr| observed across trips: 0.053

### Acceleration

- **[PASS]** `non_negative_magnitude` (semantic): Acceleration is a vector magnitude sqrt(x^2+y^2+z^2) by construction -> must be >= 0
  - checked=469762, failed=0 (0.0%)

### Speed

- **[WARN]** `non_negative` (semantic): Speed should be >= 0 (allow tiny float noise)
  - checked=469849, failed=673 (0.1432%)
  - notes: min=-0.0700212
- **[WARN]** `near_zero_nonzero_flag` (semantic): Rows with 0 < Speed < 1e-4 are the direct denominator-noise source for time_to_collision blow-ups (relative_movement dead-code bug)
  - checked=469849, failed=23138 (4.9246%)
  - notes: Quantifies exposure only; TTC fix itself is out of scope for this pass
- **[PASS]** `implausible_high` (semantic): Flag speeds far beyond the 99.9th percentile as implausible for manual review
  - checked=469849, failed=0 (0.0%)
  - notes: p99.9=77, max=80.6

### SteeringAngle

- **[WARN]** `range_report` (semantic): No hardcoded bound found in the pipeline code; report distribution for manual scale confirmation
  - checked=469762, failed=0 (0.0%)
  - notes: min=-0.5, max=0.4625, p1=-0.1936, p99=0.2736

### Brake

- **[WARN]** `pedal_range` (semantic): Brake likely a [0,1] pedal-pressure value; negative is a hard fail, >1 is a scale warning
  - checked=469762, failed=21105 (4.4927%)
  - notes: min=0, max=10

### Braking

- **[PASS]** `strict_binary` (semantic): Braking must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 411127, 1: 58722}

### Accelerating

- **[PASS]** `strict_binary` (semantic): Accelerating must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 423288, 1: 46561}

### TurnRight

- **[PASS]** `strict_binary` (semantic): TurnRight must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 418925, 1: 50924}

### TurnLeft

- **[PASS]** `strict_binary` (semantic): TurnLeft must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 425807, 1: 44042}

### comment_flag

- **[PASS]** `strict_binary` (semantic): comment_flag must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 413548, 1: 56301}
- **[PASS]** `deterministic_function_of_text` (cross-column): comment_flag should equal an independently recomputed function of text (1 iff text non-empty/non-null)
  - checked=469849, failed=0 (0.0%)

### start_comment

- **[PASS]** `strict_binary` (semantic): start_comment must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)
  - notes: value_counts={0: 467959, 1: 1890}
- **[PASS]** `strict_binary` (semantic): start_comment must be strictly {0,1}
  - checked=469849, failed=0 (0.0%)

### SpacialEvent

- **[WARN]** `frequency_and_case_variants` (semantic): Frequency table for manual review; flag case-variant duplicates and literal 'nan'/'none' strings (vs true null)
  - checked=469849, failed=228363 (48.6035%)
  - notes: top_values={'None': 228363, 'traffic light 1': 54537, 'egocar gap acceptance': 50441, 'overtake': 37757, 'traffic light 3': 29132, 'traffic light 2': 23789, 'section walker2': 14366, 'section walker1': 12974, 'gap acceptance': 6116, 'section walker3': 4930}, case_variant_groups=0, literal_nan_or_none_strings=228363

### Reason

- **[PASS]** `only_on_termination` (cross-column): Reason must be non-null only where SpacialEvent == 'Termination'
  - checked=167, failed=0 (0.0%)

### BaseEvent

- **[FAIL]** `deterministic_function_of_spacialevent` (cross-column): BaseEvent should equal clean_event_name(SpacialEvent) exactly (faithful re-implementation of all_analysis_functions_Debug.py:466-477, including its case-sensitive sentinel list). ROOT CAUSE of most mismatches: process_spacial_events() (line 751-772) calls prepare_spacial_events() -- which computes BaseEvent from SpacialEvent -- BEFORE fill_intermediate_events() fills in SpacialEvent for rows between paired start/end events. So BaseEvent is stale ('None') for every row whose SpacialEvent value was only populated by the later fill step; it was never recomputed after the fill.
  - checked=469849, failed=234498 (49.9092%)
  - notes: of 234498 mismatches, 234320 are the stale-'None' pattern (BaseEvent never recomputed after SpacialEvent was filled); remainder may be other cases, spot-check before assuming they're all the same root cause
  - example failing row indices: 777; 778; 779; 780; 781

### time_since_event_start_world

- **[PASS]** `non_negative` (semantic): Should be >= 0 where present (elapsed time since event start)
  - checked=241031, failed=0 (0.0%)

### event_category

- **[PASS]** `enum_by_construction` (semantic): categorize_event() can only return one of ['GapAcceptance', 'Overtake', 'Pedestrians', 'SimulationPoints', 'TrafficLights'] or None -- any other value implies function drift
  - checked=241189, failed=0 (0.0%)
  - notes: value_counts={'TrafficLights': 108360, 'GapAcceptance': 56859, 'Overtake': 38096, 'Pedestrians': 37716, 'SimulationPoints': 158}

### Overtake

- **[PASS]** `enum_phase_labels` (semantic): Overtake (sudden-stop phase label) must be one of ['restart drive', 'start brake', 'start drive', 'stop']
  - checked=257, failed=0 (0.0%)
  - notes: value_counts={'start drive': 85, 'stop': 81, 'start brake': 78, 'restart drive': 13}
- **[WARN]** `no_duplicate_labels_per_trip` (cross-column): Each phase label should appear at most once per (Id,Condition) trip
  - checked=256, failed=1 (0.3906%)

### distance_from_relevant_object / relevant_object_name

- **[PASS]** `null_pattern_consistency` (cross-column): distance_from_relevant_object must be non-null iff relevant_object_name is non-null
  - checked=469849, failed=0 (0.0%)

### distance_from_relevant_object

- **[PASS]** `non_negative` (semantic): Haversine distance must be >= 0
  - checked=238872, failed=0 (0.0%)

### relevant_object_name

- **[PASS]** `frequency_report` (semantic): Frequency table for manual sanity check of naming conventions
  - checked=238872, failed=0 (0.0%)
  - notes: top_values={'traffic.traffic_light 60': 54488, 'traffic.traffic_light 69': 29340, 'traffic.traffic_light 80': 23973, 'vehicle.tesla.model3 258': 10782, 'walker.pedestrian.0002 253': 10700, 'walker.pedestrian.0001 256': 10544, 'vehicle.tesla.model3 253': 10543, 'vehicle.tesla.model3 263': 7000, 'vehicle.tesla.model3 262': 6793, 'vehicle.tesla.model3 264': 5832}

### Pedestrian

- **[PASS]** `compound_marker_format` (semantic): Non-null values must match '<start walking|start crossing|end crossing>: <name>' possibly ' | '-joined
  - checked=230, failed=0 (0.0%)
  - notes: non-null count=230/469849 (0.05%)

### SpacialEvent (Pedestrians category)

- **[WARN]** `walker2_presence_Avatar` (semantic): Count of Pedestrians-category rows and presence of a Walker2 stage for Condition=Avatar
  - checked=13413, failed=13413 (100.0%)
  - notes: n_pedestrian_rows=13413, distinct_SpacialEvent=['End section walker1', 'End section walker2', 'End section walker3', 'End section walker4', 'Start section walker1', 'Start section walker2', 'Start section walker3', 'Start section walker4', 'Walker1', 'section walker1', 'section walker2', 'section walker3', 'section walker4'], has_walker2_stage=False. Root-cause investigation (design difference vs object-matching bug in find_matching_objects) deferred to a follow-up pass.
- **[PASS]** `walker2_presence_Conventional` (semantic): Count of Pedestrians-category rows and presence of a Walker2 stage for Condition=Conventional
  - checked=10254, failed=0 (0.0%)
  - notes: n_pedestrian_rows=10254, distinct_SpacialEvent=['End section walker1', 'End section walker2', 'End section walker3', 'End section walker4', 'Start section walker1', 'Start section walker2', 'Start section walker3', 'Start section walker4', 'Walker1', 'Walker2', 'section walker1', 'section walker2', 'section walker3', 'section walker4'], has_walker2_stage=True. Root-cause investigation (design difference vs object-matching bug in find_matching_objects) deferred to a follow-up pass.
- **[PASS]** `walker2_presence_Remote` (semantic): Count of Pedestrians-category rows and presence of a Walker2 stage for Condition=Remote
  - checked=14049, failed=0 (0.0%)
  - notes: n_pedestrian_rows=14049, distinct_SpacialEvent=['End section walker1', 'End section walker2', 'End section walker3', 'End section walker4', 'Start section walker1', 'Start section walker2', 'Start section walker3', 'Start section walker4', 'Walker1', 'Walker2', 'section walker1', 'section walker2', 'section walker3', 'section walker4'], has_walker2_stage=True. Root-cause investigation (design difference vs object-matching bug in find_matching_objects) deferred to a follow-up pass.

### TrafficLight

- **[WARN]** `vocabulary_report` (semantic): True CurrentState vocabulary lives in raw per-participant objects JSON, not derivable from this CSV alone -- reporting observed values instead of assuming Red/Yellow/Green
  - checked=107801, failed=0 (0.0%)
  - notes: value_counts={'Green': 75941, 'Red': 24880, 'Yellow': 6980}
- **[PASS]** `category_alignment` (cross-column): TrafficLight is merged by relevant_object_name across ALL rows in the pipeline, not filtered to event_category=='TrafficLights' -- flagging as WARN, not FAIL, since this may be expected incidental matching
  - checked=469849, failed=0 (0.0%)

### TrafficLight_JunctionPhase

- **[PASS]** `enum` (semantic): Must be one of ['Approaching', 'AtStopLine', 'InsideJunction', 'LeavingJunction']
  - checked=106011, failed=0 (0.0%)
  - notes: value_counts={'Approaching': 85617, 'InsideJunction': 18529, 'LeavingJunction': 1752, 'AtStopLine': 113}
- **[WARN]** `forward_only_phase_order` (cross-column): Within a trip, phases should progress Approaching -> AtStopLine -> InsideJunction -> LeavingJunction (no backward transitions)
  - checked=105936, failed=38 (0.0359%)

### gap_acceptance

- **[PASS]** `non_negative` (semantic): gap_acceptance = abs(dist2-dist1) so must be >= 0
  - checked=49532, failed=0 (0.0%)
- **[PASS]** `spacialevent_alignment` (cross-column): Non-null gap_acceptance rows should have a SpacialEvent containing 'gap acceptance'
  - checked=49532, failed=0 (0.0%)

### time_to_collision

- **[FAIL]** `sanity_ceiling` (semantic): Flag values above a non-physical ceiling (10000.0s) for a driving-sim scenario
  - checked=193411, failed=17125 (8.8542%)
  - notes: max=3.73627e+12
  - example failing row indices: 877; 884; 885; 886; 887
- **[WARN]** `relative_movement_deadcode_exposure` (semantic): Rows where time_to_collision was computed while Speed is near-zero-nonzero: known root cause is calculate_time_to_collision_with_police() reading a 'relative_movement' field that df_unique_events_objects never has, so relative_speed always falls back to raw ego Speed (dead code for same/opposite/crossing branches) -- quantifying exposure only, fix deferred to a follow-up pass
  - checked=193411, failed=14821 (7.663%)

### ttc_object_name

- **[PASS]** `null_pattern_consistency` (cross-column): ttc_object_name must be non-null iff time_to_collision is non-null
  - checked=469849, failed=0 (0.0%)

### text

- **[WARN]** `encoding_and_literal_nan` (semantic): Check for UTF-8 mojibake (replacement char) and literal 'nan'/empty strings masquerading as null
  - checked=56301, failed=752 (1.3357%)
  - notes: replacement_char_rows=752, literal_nan_or_empty=0

### speaker

- **[WARN]** `frequency_and_whitespace_variants` (semantic): Frequency table; flag whitespace/case-variant duplicates and literal 'nan' strings
  - checked=56301, failed=13861 (24.6195%)
  - notes: top_values={'accompanier': 40852, 'nan': 13861, 'young driver': 1489, 'young driver | accompanier': 44, 'accompanier | young driver': 26, 'מעביר הניסוי': 12, 'both': 4, 'both | accompanier': 2, 'experimenter': 2, 'young driver | accompanier | young driver | accompanier': 2}, variant_groups=0, literal_nan=13861
- **[PASS]** `null_iff_text_null` (cross-column): speaker should be null wherever text is null
  - checked=469849, failed=0 (0.0%)

### transcript_type

- **[PASS]** `enum` (semantic): Must be one of ['manual', 'whisper']
  - checked=56301, failed=0 (0.0%)
  - notes: value_counts={'whisper': 52939, 'manual': 3362}
- **[PASS]** `single_type_per_trip` (cross-column): A trip should never mix 'manual' and 'whisper' (selection happens once per trip at load time)
  - checked=87, failed=0 (0.0%)
