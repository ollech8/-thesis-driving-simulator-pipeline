# -*- coding: utf-8 -*-
"""
Run analysis for ALL participants across ALL relevant conditions (Egocar only),
FILTER:
    Id startswith "C"
    Scenario == "Accompanied"
    triggered_by == "Egocar"
Skip Training/Baseline, skip rows with missing files.

OUTPUT:
- ONE combined CSV for all participants (incremental append: mode="a", header only once)
- ONE failures CSV (append)
NO CHECKPOINT.
"""

import os
import pandas as pd
import all_analysis_functions_Debug as aaf


# -------------------
# CONFIG
# -------------------
metadata_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_simulator_and_corresponding_physiological_files.csv"

kinematic_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_Accompanied.csv"
spacial_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_with_transcription_Accompanied.csv"
object_points_path = r"H:\האחסון שלי\Ariel Uni\Readme\objectPoints.csv"
transcription_file_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_events_with_transcription_Accompanied.csv"

# Output files (in current working directory)
output_path = os.path.join(os.getcwd(), "ALL_participants_analysis_ALL_conditions.csv")
failed_path = os.path.join(os.getcwd(), "ALL_participants_failed_conditions.csv")

# Optional: limit for quick tests (set None to run all)
ONLY_PARTICIPANTS = None  # e.g. ["C1_036248", "C2_121241"]

# Fixed schema for the combined output CSV. append_output() reindexes every
# trip's DataFrame to this exact column list before writing, so a future
# accidental column-count mismatch between trips (e.g. one branch adding a
# column the other doesn't) can never again silently column-shift the CSV --
# it would instead show up as an all-NaN column or a visible KeyError.
CANONICAL_COLUMNS = [
    "Id", "Condition", "Map", "FrameID", "WorldTime", "SimulationTime",
    "Longitude", "Latitude", "PositionX", "PositionY", "Acceleration", "Speed",
    "Yaw", "SteeringAngle", "Brake", "Braking", "Accelerating", "TurnRight", "TurnLeft",
    "SpacialEvent", "Reason", "BaseEvent", "time_since_event_start_world",
    "event_category", "Overtake", "distance_from_relevant_object",
    "relevant_object_name", "Pedestrian", "TrafficLight",
    "TrafficLight_JunctionPhase", "gap_acceptance", "time_to_collision",
    "ttc_object_name", "text", "speaker", "transcript_type", "comment_flag",
    "start_comment", "first_feedback_in_event",
]


# -------------------
# HELPERS
# -------------------
def normalize_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def record_fail(pid, condition, map_type, step, err):
    row = pd.DataFrame([{
        "Id": normalize_str(pid),
        "Condition": normalize_str(condition),
        "Map": normalize_str(map_type),
        "FailedStep": step,
        "ErrorMessage": str(err)
    }])
    write_header = not os.path.exists(failed_path)
    row.to_csv(failed_path, mode="a", header=write_header, index=False)

def append_output(df_part):
    extra_cols = [c for c in df_part.columns if c not in CANONICAL_COLUMNS]
    if extra_cols:
        print(f"⚠️ append_output: dropping unexpected column(s) not in CANONICAL_COLUMNS: {extra_cols}")
    df_part = df_part.reindex(columns=CANONICAL_COLUMNS)
    write_header = not os.path.exists(output_path)
    df_part.to_csv(output_path, mode="a", header=write_header, index=False)


# -------------------
# (OPTIONAL) CLEAN START
# -------------------
# If you want a fresh run each time, uncomment these:
# for p in [output_path, failed_path]:
#     if os.path.exists(p):
#         os.remove(p)
#         print("Deleted:", p)


# -------------------
# LOAD META + FILTER
# -------------------
df_meta = pd.read_csv(metadata_path)
df_meta["Id"] = df_meta["Id"].astype(str)

filtered_rows = df_meta[
    df_meta["Id"].str.startswith("C", na=False) &
    (df_meta["Scenario"] == "Accompanied") &
    (df_meta["triggered_by"] == "Egocar")
].copy()

if ONLY_PARTICIPANTS is not None:
    filtered_rows = filtered_rows[filtered_rows["Id"].isin(ONLY_PARTICIPANTS)].copy()

if filtered_rows.empty:
    raise ValueError("לא נמצאו רשומות אחרי הסינון (C + Accompanied + Egocar).")

participant_ids = sorted(filtered_rows["Id"].dropna().unique().tolist())
print(f"Found {len(participant_ids)} participants after filtering.")


# -------------------
# RUN PIPELINE FOR ALL
# -------------------
for participant_id in participant_ids:
    print("\n==============================")
    print(f"🚀 Starting participant: {participant_id}")
    print("==============================")

    participant_rows = filtered_rows[filtered_rows["Id"] == participant_id].copy()

    for _, meta_row in participant_rows.iterrows():
        condition = meta_row.get("Condition", "")
        map_type = meta_row.get("Map", "")

        # Skip Training / Baseline
        cond = normalize_str(condition).lower()
        if cond in ["training", "baseline"]:
            print(f"⏭️ Skip Training/Baseline: {participant_id} | {condition} | Map {map_type}")
            continue

        # Required files
        kin_file = meta_row.get("KinematicFile", None)
        gps_file = meta_row.get("GPSFile", None)

        if pd.isna(kin_file) or normalize_str(kin_file) == "":
            print(f"⏭️ Skip: {participant_id} | {condition} | Missing KinematicFile")
            record_fail(participant_id, condition, map_type, "meta_check", "Missing KinematicFile")
            continue

        if pd.isna(gps_file) or normalize_str(gps_file) == "":
            print(f"⏭️ Skip: {participant_id} | {condition} | Missing GPSFile")
            record_fail(participant_id, condition, map_type, "meta_check", "Missing GPSFile")
            continue

        # Build paths
        egocar_path = str(kin_file).replace("My Drive", "האחסון שלי")
        object_file_path = str(gps_file).replace("My Drive", "האחסון שלי")

        print(f"\n🚗 Running: {participant_id} | {condition} | Map {map_type}")

        # try:
        #     df_final = aaf.process_file(egocar_path, participant_id, condition, map_type)
        #     if df_final is None or df_final.empty:
        #         print(f"⚠️ No GPS/Telemetries: {participant_id} | {condition} | Map {map_type}")
        #         record_fail(participant_id, condition, map_type, "process_file", "Returned empty/None")
        #         continue

        #     df_final = aaf.process_kinematic_data(kinematic_file_path, df_final)
        #     if df_final is None or df_final.empty:
        #         record_fail(participant_id, condition, map_type, "process_kinematic_data", "Returned empty/None")
        #         continue

        #     df_final = aaf.process_spacial_data(spacial_file_path, df_final)
        #     if df_final is None or df_final.empty:
        #         record_fail(participant_id, condition, map_type, "process_spacial_data", "Returned empty/None")
        #         continue

        #     df_final = aaf.process_spacial_events(df_final)
        #     if df_final is None or df_final.empty:
        #         record_fail(participant_id, condition, map_type, "process_spacial_events", "Returned empty/None")
        #         continue

        #     df_categorized_events = aaf.process_event_categorization(df_final)
        #     if df_categorized_events is None or df_categorized_events.empty:
        #         record_fail(participant_id, condition, map_type, "process_event_categorization", "Returned empty/None")
        #         continue

        #     df_objects = aaf.process_objects_data(object_file_path)
        #     if df_objects is None or df_objects.empty:
        #         record_fail(participant_id, condition, map_type, "process_objects_data", "Returned empty/None")
        #         continue

        #     df_unique_events_objects = aaf.find_relevant_objects(object_points_path, df_objects, df_categorized_events)
        #     if df_unique_events_objects is None or (hasattr(df_unique_events_objects, "empty") and df_unique_events_objects.empty):
        #         record_fail(participant_id, condition, map_type, "find_relevant_objects", "Returned empty/None")
        #         continue

        #     df_with_sudden_stop_labels = aaf.add_sudden_stop_phases(df_categorized_events, df_unique_events_objects, df_objects)
        #     if df_with_sudden_stop_labels is None or (hasattr(df_with_sudden_stop_labels, "empty") and df_with_sudden_stop_labels.empty):
        #         record_fail(participant_id, condition, map_type, "add_sudden_stop_phases", "Returned empty/None")
        #         continue

        #     df_with_distances = aaf.calculate_all_event_distances(df_with_sudden_stop_labels, df_objects, df_unique_events_objects)
        #     if df_with_distances is None or (hasattr(df_with_distances, "empty") and df_with_distances.empty):
        #         record_fail(participant_id, condition, map_type, "calculate_all_event_distances", "Returned empty/None")
        #         continue
            
            
        #     df_events_with_pedestrians = aaf.process_pedestrian_labels(df_with_distances,df_unique_events_objects,df_objects,object_points_path)        
        #     print("After process_pedestrian_labels:",df_events_with_pedestrians.shape if df_events_with_pedestrians is not None else "None")


        #     df_with_distances_light = aaf.add_traffic_light_state(df_events_with_pedestrians, df_objects)
        #     if df_with_distances_light is None or (hasattr(df_with_distances_light, "empty") and df_with_distances_light.empty):
        #         record_fail(participant_id, condition, map_type, "add_traffic_light_state", "Returned empty/None")
        #         continue

        #     # GapAcceptance
        #     df_gap_acceptance = aaf.calculate_gap_acceptance_table(df_objects, df_with_distances_light)

        #     if df_gap_acceptance is None or df_gap_acceptance.empty:
        #         df_with_distances_gap = df_with_distances_light.copy()
        #         if "gap_acceptance" not in df_with_distances_gap.columns:
        #             df_with_distances_gap["gap_acceptance"] = pd.NA
        #     else:
        #         df_with_distances_gap = aaf.merge_gap_acceptance_to_main(df_with_distances_light, df_gap_acceptance)

        #     if df_with_distances_gap is None or (hasattr(df_with_distances_gap, "empty") and df_with_distances_gap.empty):
        #         record_fail(participant_id, condition, map_type, "merge_gap_acceptance_to_main", "Returned empty/None")
        #         continue


        #     df_distance_to_police = aaf.create_distance_to_police_df(df_with_distances_gap, df_objects)
            
            # אם אין משטרה (תקין במפה C) – לא מפילים את הנסיעה, פשוט ממשיכים עם DF ריק
        #     if df_distance_to_police is None or (hasattr(df_distance_to_police, "empty") and df_distance_to_police.empty):
        #         if normalize_str(map_type).upper() == "C":
        #             print("ℹ️ Map C: no police (expected) -> continuing without police distances.")
        #             df_distance_to_police = pd.DataFrame(
        #                 columns=["SimulationTime", "distance_to_police", "police_object_name"]
        #             )
        #         else:
        #             record_fail(participant_id, condition, map_type, "create_distance_to_police_df", "Returned empty/None")
        #             continue



        #     overtake_sim_time = aaf.detect_overtake_start_simulation_time(df_with_distances_gap)

        #     df_with_ttc = aaf.calculate_time_to_collision_with_police(
        #         df_with_distances_gap,
        #         df_unique_events_objects,
        #         df_objects,
        #         df_distance_to_police,
        #         overtake_sim_time
        #     )
        #     if df_with_ttc is None or (hasattr(df_with_ttc, "empty") and df_with_ttc.empty):
        #         record_fail(participant_id, condition, map_type, "calculate_time_to_collision_with_police", "Returned empty/None")
        #         continue

        #     df_transcripts = aaf.load_transcription_data(transcription_file_path)
        #     if df_transcripts is None or df_transcripts.empty:
        #         record_fail(participant_id, condition, map_type, "load_transcription_data", "Returned empty/None")
        #         continue

        #     df_with_text = aaf.add_transcription_to_events(df_with_ttc, df_transcripts)
        #     if df_with_text is None or (hasattr(df_with_text, "empty") and df_with_text.empty):
        #         record_fail(participant_id, condition, map_type, "add_transcription_to_events", "Returned empty/None")
        #         continue

        #     df_with_final_comments = aaf.add_comment_flag(df_with_text)
        #     if df_with_final_comments is None or (hasattr(df_with_final_comments, "empty") and df_with_final_comments.empty):
        #         record_fail(participant_id, condition, map_type, "add_comment_flag", "Returned empty/None")
        #         continue

        #     df_with_final_comments = aaf.add_start_comment_column(df_with_final_comments, df_transcripts, round_to=1)
        #     if df_with_final_comments is None or (hasattr(df_with_final_comments, "empty") and df_with_final_comments.empty):
        #         record_fail(participant_id, condition, map_type, "add_start_comment_column", "Returned empty/None")
        #         continue
        try:
            df_final = aaf.process_file(egocar_path, participant_id, condition, map_type)
            
            if df_final is None or df_final.empty:
                print(f"⚠️ אין נתוני GPS/Telemetries עבור {participant_id} | {condition}")
                continue
    
    
            df_final = aaf.process_kinematic_data(kinematic_file_path, df_final)
            print("After process_kinematic_data:", df_final.shape if df_final is not None else "None")
    
            df_final = aaf.process_spacial_data(spacial_file_path, df_final)
            print("After process_spacial_data:", df_final.shape if df_final is not None else "None")
    
            df_final = aaf.process_spacial_events(df_final)
            print("After process_spacial_events:", df_final.shape if df_final is not None else "None")
            
            mask = df_final["SpacialEvent"].astype(str).str.contains("overtake", case=False, na=False)
    
            print("DEBUG overtake count in SpacialEvent:", mask.sum())
            
            print(
                df_final.loc[mask, "SpacialEvent"]
                .value_counts()
                .head(20)
            )
    
    
            df_categorized_events = aaf.process_event_categorization(df_final)
            print("After process_event_categorization:", df_categorized_events.shape if df_categorized_events is not None else "None")
            
            
            print("DEBUG event_category counts:")
            print(df_categorized_events["event_category"].value_counts(dropna=False))
            
            print("\nDEBUG OverTake SpacialEvent values:")
            print(
                df_categorized_events[
                    df_categorized_events["event_category"] == "OverTake"
                ]["SpacialEvent"].value_counts().head(20)
            )
    
    
            # df_objects = aaf.process_objects_data(object_file_path)
            # print("After process_objects_data:", df_objects.shape if df_objects is not None else "None")
            
            df_objects = aaf.process_objects_data(object_file_path)
    
            # 🔍 DEBUG – בדיקת רמזורים בקובץ האובייקטים
            print("DEBUG df_objects Type counts:")
            print(df_objects["Type"].value_counts())
            
            print("\nDEBUG traffic objects by Name:")
            print(
                df_objects[df_objects["Name"].str.contains("traffic", case=False, na=False)][
                    ["Type", "Name"]
                ].head(20)
            )
            print(df_objects["Name"].dropna().unique())
    
    
    
            df_unique_events_objects = aaf.find_relevant_objects(object_points_path, df_objects, df_categorized_events)
            print("After find_relevant_objects:", df_unique_events_objects.shape if df_unique_events_objects is not None else "None")
            
            df_with_sudden_stop_labels = aaf.add_sudden_stop_phases(df_categorized_events, df_unique_events_objects, df_objects)
            print("After add_sudden_stop_phases:", df_with_sudden_stop_labels.shape if df_with_sudden_stop_labels is not None else "None")
    
            df_with_distances = aaf.calculate_all_event_distances(df_with_sudden_stop_labels, df_objects, df_unique_events_objects)
            print("After calculate_all_event_distances:", df_with_distances.shape if df_with_distances is not None else "None")
    
            df_events_with_pedestrians = aaf.process_pedestrian_labels(df_with_distances,df_unique_events_objects,df_objects,object_points_path)        
            print(
                "After process_pedestrian_labels:",
                df_events_with_pedestrians.shape if df_events_with_pedestrians is not None else "None"
            )
    
            
            df_with_distances_light = aaf.add_traffic_light_state(df_events_with_pedestrians, df_objects)
            print("After add_traffic_light_state:", df_with_distances_light.shape if df_with_distances_light is not None else "None")
            
            df_with_distances_light = aaf.add_traffic_light_junction_phase(
                df_with_distances_light,
                spacial_file_path=spacial_file_path,
                participant_id=participant_id,
                condition=condition,
                round_to=1
            )
    
            
            print("DEBUG SpacialEvent counts:",
            df_with_distances_light["SpacialEvent"].value_counts().head(10))
            
    # כרגע קורס בפונקציה הבאה
            df_gap_acceptance = aaf.calculate_gap_acceptance_table(df_objects, df_with_distances_light)
            print("After calculate_gap_acceptance_table:", df_gap_acceptance.shape if df_gap_acceptance is not None else "None")
    
            if df_gap_acceptance is None or df_gap_acceptance.empty:
                print(f"⚠️ אין GapAcceptance עבור {participant_id} | {condition} | Map {map_type} → ממשיך בלי merge")
                df_with_distances_gap = df_with_distances_light.copy()
                # optional: create empty column so downstream code can rely on it
                if "gap_acceptance" not in df_with_distances_gap.columns:
                    df_with_distances_gap["gap_acceptance"] = pd.NA
            else:
                df_with_distances_gap = aaf.merge_gap_acceptance_to_main(df_with_distances_light, df_gap_acceptance)
            print("After merge_gap_acceptance_to_main:", df_with_distances_gap.shape if df_with_distances_gap is not None else "None")
    
            df_distance_to_police = aaf.create_distance_to_police_df(df_with_distances_gap, df_objects)
            print("After create_distance_to_police_df:", df_distance_to_police.shape if df_distance_to_police is not None else "None")
    
            overtake_sim_time = aaf.detect_overtake_start_simulation_time(df_with_distances_gap)
            print("After detect_overtake_start_simulation_time:", overtake_sim_time)
    
            df_with_ttc = aaf.calculate_time_to_collision_with_police(
                df_with_distances_gap,
                df_unique_events_objects,
                df_objects,
                df_distance_to_police,
                overtake_sim_time
            )
            print("After calculate_time_to_collision_with_police:", df_with_ttc.shape if df_with_ttc is not None else "None")
    
    
    
            df_with_final_comments, df_transcripts = aaf.process_transcription_pipeline(
                df_with_ttc,
                transcription_file_path,
                participant_id,
                condition,
                map_type,
                verbose=True
            )

            # Add identifiers + reorder columns
            df_with_final_comments["Id"] = participant_id
            df_with_final_comments["Condition"] = condition
            df_with_final_comments["Map"] = map_type

            first_cols = ["Id", "Condition", "Map"]
            other_cols = [c for c in df_with_final_comments.columns if c not in first_cols]
            df_with_final_comments = df_with_final_comments[first_cols + other_cols]

            # Incremental output write
            append_output(df_with_final_comments)

            print(f"✅ Success: {participant_id} | {condition} | Map {map_type}")

        except Exception as e:
            print(f"❌ Error: {participant_id} | {condition} | Map {map_type} | {e}")
            record_fail(participant_id, condition, map_type, "pipeline_exception", e)
            continue


print("\n🏁 Done. Outputs:")
print(" -", output_path)
print(" -", failed_path)
