# -*- coding: utf-8 -*-
"""
Single-trip test harness for verifying pipeline fixes, without touching the real
output CSV or running all 39 participants.

Runs the exact same per-trip call sequence as run_analysis_Debug.py, for one
hardcoded trip (C1_036248 / Remote / Map A), and saves the resulting DataFrame
to claude debug/outputs/test_trip_output.csv (or a name passed via --tag) so
before/after fixes can be compared.

Usage (from the "claude debug" folder, with the Anaconda Python interpreter):
    python test_single_trip.py --tag baseline
    python test_single_trip.py --tag after_fix
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import all_analysis_functions_Debug as aaf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")

# -------------------
# CONFIG (same paths as run_analysis_Debug.py)
# -------------------
metadata_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_simulator_and_corresponding_physiological_files.csv"
kinematic_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_Accompanied.csv"
spacial_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_with_transcription_Accompanied.csv"
object_points_path = r"H:\האחסון שלי\Ariel Uni\Readme\objectPoints.csv"
transcription_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_with_transcription_Accompanied.csv"

PARTICIPANT_ID = "C1_036248"
CONDITION = "Remote"
MAP_TYPE = "A"
EGOCAR_PATH = r"H:\האחסון שלי\Ariel Uni\C1_036248\Simulation\Accompanied\Far\EgoCar_Parent_with_out_videos_MapA_2024-12-10_12-40-05.json"
OBJECT_FILE_PATH = r"H:\האחסון שלי\Ariel Uni\C1_036248\Simulation\Accompanied\Far\Objects_Parent_with_out_videos_MapA_2024-12-10_12-40-04.json"


def apply_cli_trip_overrides(args):
    """Allows overriding the hardcoded trip via CLI so the same harness can test any trip."""
    global PARTICIPANT_ID, CONDITION, MAP_TYPE, EGOCAR_PATH, OBJECT_FILE_PATH
    if args.participant_id:
        PARTICIPANT_ID = args.participant_id
    if args.condition:
        CONDITION = args.condition
    if args.map_type:
        MAP_TYPE = args.map_type
    if args.egocar_path:
        EGOCAR_PATH = args.egocar_path
    if args.object_file_path:
        OBJECT_FILE_PATH = args.object_file_path


def run_trip(stop_after=None):
    """
    Runs the pipeline for the one hardcoded trip.
    stop_after: optional function-name string to stop after (for isolated testing
    of early steps without needing the whole chain to succeed), e.g. "process_spacial_events".
    Returns the DataFrame at the point execution stopped (or the final one).
    """
    df_final = aaf.process_file(EGOCAR_PATH, PARTICIPANT_ID, CONDITION, MAP_TYPE)
    print("After process_file:", df_final.shape)

    df_final = aaf.process_kinematic_data(kinematic_file_path, df_final)
    print("After process_kinematic_data:", df_final.shape)

    df_final = aaf.process_spacial_data(spacial_file_path, df_final)
    print("After process_spacial_data:", df_final.shape)

    df_final = aaf.process_spacial_events(df_final)
    print("After process_spacial_events:", df_final.shape)
    if stop_after == "process_spacial_events":
        return df_final

    df_categorized_events = aaf.process_event_categorization(df_final)
    print("After process_event_categorization:", df_categorized_events.shape)
    if stop_after == "process_event_categorization":
        return df_categorized_events

    df_objects = aaf.process_objects_data(OBJECT_FILE_PATH)
    print("After process_objects_data:", df_objects.shape)

    df_unique_events_objects = aaf.find_relevant_objects(object_points_path, df_objects, df_categorized_events)
    print("After find_relevant_objects:", df_unique_events_objects.shape)

    df_with_sudden_stop_labels = aaf.add_sudden_stop_phases(df_categorized_events, df_unique_events_objects, df_objects)
    print("After add_sudden_stop_phases:", df_with_sudden_stop_labels.shape)

    df_with_distances = aaf.calculate_all_event_distances(df_with_sudden_stop_labels, df_objects, df_unique_events_objects)
    print("After calculate_all_event_distances:", df_with_distances.shape)
    if stop_after == "calculate_all_event_distances":
        return df_with_distances, df_unique_events_objects, df_objects

    df_events_with_pedestrians = aaf.process_pedestrian_labels(
        df_with_distances, df_unique_events_objects, df_objects, object_points_path
    )
    print("After process_pedestrian_labels:", df_events_with_pedestrians.shape)
    if stop_after == "process_pedestrian_labels":
        return df_events_with_pedestrians, df_unique_events_objects, df_objects

    df_with_distances_light = aaf.add_traffic_light_state(df_events_with_pedestrians, df_objects)
    print("After add_traffic_light_state:", df_with_distances_light.shape)

    df_with_distances_light = aaf.add_traffic_light_junction_phase(
        df_with_distances_light,
        spacial_file_path=spacial_file_path,
        participant_id=PARTICIPANT_ID,
        condition=CONDITION,
        round_to=1,
    )
    print("After add_traffic_light_junction_phase:", df_with_distances_light.shape)

    df_gap_acceptance = aaf.calculate_gap_acceptance_table(df_objects, df_with_distances_light)
    if df_gap_acceptance is None or df_gap_acceptance.empty:
        df_with_distances_gap = df_with_distances_light.copy()
        if "gap_acceptance" not in df_with_distances_gap.columns:
            df_with_distances_gap["gap_acceptance"] = pd.NA
    else:
        df_with_distances_gap = aaf.merge_gap_acceptance_to_main(df_with_distances_light, df_gap_acceptance)
    print("After merge_gap_acceptance_to_main:", df_with_distances_gap.shape)
    if stop_after == "merge_gap_acceptance_to_main":
        return df_with_distances_gap, df_unique_events_objects, df_objects

    df_distance_to_police = aaf.create_distance_to_police_df(df_with_distances_gap, df_objects)
    print("After create_distance_to_police_df:", None if df_distance_to_police is None else df_distance_to_police.shape)

    overtake_sim_time = aaf.detect_overtake_start_simulation_time(df_with_distances_gap)
    print("After detect_overtake_start_simulation_time:", overtake_sim_time)

    df_with_ttc = aaf.calculate_time_to_collision_with_police(
        df_with_distances_gap, df_unique_events_objects, df_objects, df_distance_to_police, overtake_sim_time
    )
    print("After calculate_time_to_collision_with_police:", df_with_ttc.shape)
    if stop_after == "calculate_time_to_collision_with_police":
        return df_with_ttc, df_unique_events_objects, df_objects

    df_with_final_comments, df_transcripts = aaf.process_transcription_pipeline(
        df_with_ttc, transcription_file_path, PARTICIPANT_ID, CONDITION, MAP_TYPE, verbose=False
    )

    df_with_final_comments["Id"] = PARTICIPANT_ID
    df_with_final_comments["Condition"] = CONDITION
    df_with_final_comments["Map"] = MAP_TYPE
    first_cols = ["Id", "Condition", "Map"]
    other_cols = [c for c in df_with_final_comments.columns if c not in first_cols]
    df_with_final_comments = df_with_final_comments[first_cols + other_cols]

    return df_with_final_comments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="test", help="suffix for the output filename")
    parser.add_argument("--stop_after", default=None, help="function name to stop after")
    parser.add_argument("--participant_id", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--map_type", default=None)
    parser.add_argument("--egocar_path", default=None)
    parser.add_argument("--object_file_path", default=None)
    args = parser.parse_args()

    apply_cli_trip_overrides(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = run_trip(stop_after=args.stop_after)
    df_out = result[0] if isinstance(result, tuple) else result

    out_path = os.path.join(OUTPUT_DIR, f"test_trip_{args.tag}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  shape={df_out.shape}")
