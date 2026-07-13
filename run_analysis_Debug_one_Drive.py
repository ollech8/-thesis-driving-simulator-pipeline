# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 10:34:03 2026

@author: ASUS VIVOBOOK
"""

# -*- coding: utf-8 -*-
"""
Run analysis for ONE trip only: (participant_id + condition),
triggered_by == Egocar only,
skip Training/Baseline,
skip if missing files,
write ONE output CSV for that single trip.
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
###### להוסיף לפייפליין המרכזי
ttc_source_path = r"H:\האחסון שלי\Ariel Uni\Readme\calc_simulator_and_corresponding_physiological_files.csv"

participant_id = "C1_072659"
# 'C1_036248'
# "C6_125867"
condition = "Remote"   # Avatar / Remote / Conventional
#Remote
# Avatar
#Conventional

# -------------------
# LOAD META
# -------------------
df_meta = pd.read_csv(metadata_path)

# filter: this participant + this condition + triggered_by Egocar
meta_rows = df_meta[
    (df_meta["Id"] == participant_id) &
    (df_meta["Condition"] == condition) &
    (df_meta["triggered_by"].astype(str).str.strip().str.lower() == "egocar")
].copy()

if meta_rows.empty:
    raise ValueError(f"לא נמצאה נסיעה עבור {participant_id} | {condition} עם triggered_by=Egocar")

# Skip Training/Baseline explicitly (just in case)
cond = str(condition).strip().lower()
if cond in ["training", "baseline"]:
    raise ValueError(f"ה־condition שנבחר הוא {condition} ולכן מדלגים עליו.")

# Output paths
output_path = os.path.join(os.getcwd(), f"{participant_id}_{condition}_analysis.csv")
failed_path = os.path.join(os.getcwd(), f"{participant_id}_{condition}_failed.csv")

all_results = []
failed_rows = []


def record_fail(map_type, step, err):
    failed_rows.append({
        "Id": participant_id,
        "Condition": condition,
        "Map": map_type,
        "FailedStep": step,
        "ErrorMessage": str(err)
    })


# -------------------
# RUN (one condition, maybe multiple maps)
# -------------------
for _, meta_row in meta_rows.iterrows():
    map_type = meta_row.get("Map", "")

    kin_file = meta_row.get("KinematicFile", None)
    gps_file = meta_row.get("GPSFile", None)

    if pd.isna(kin_file) or str(kin_file).strip() == "":
        print(f"⏭️ מדלג: {participant_id} | {condition} | Map {map_type} | חסר KinematicFile")
        record_fail(map_type, "meta_check", "Missing KinematicFile")
        continue

    if pd.isna(gps_file) or str(gps_file).strip() == "":
        print(f"⏭️ מדלג: {participant_id} | {condition} | Map {map_type} | חסר GPSFile")
        record_fail(map_type, "meta_check", "Missing GPSFile")
        continue

    egocar_path = str(kin_file).replace("My Drive", "האחסון שלי")
    object_file_path = str(gps_file).replace("My Drive", "האחסון שלי")

    print(f"\n🚗 מריץ ניתוח עבור {participant_id} - {condition} - Map {map_type}")

    try:
        df_final = aaf.process_file(egocar_path, participant_id, condition, map_type)
        
        if df_final is None or df_final.empty:
            print(f"⚠️ אין נתוני GPS/Telemetries עבור {participant_id} | {condition}")
            continue


        df_final = aaf.process_kinematic_data(kinematic_file_path, df_final)
        print("After process_kinematic_data:", df_final.shape if df_final is not None else "None")

        df_final = aaf.process_spacial_data(spacial_file_path, df_final)
        print("After process_spacial_data:", df_final.shape if df_final is not None else "None")

        # DEBUG: Check TL3 events after spacial merge
        tl3_mask = df_final["SpacialEvent"].astype(str).str.contains("traffic light 3", case=False, na=False)
        print("DEBUG TL3 rows in df_final after process_spacial_data:", tl3_mask.sum())
        print(df_final.loc[tl3_mask, ["SimulationTime", "SpacialEvent"]])

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


        # -------------------
        # Add identifiers + reorder columns
        # -------------------
        df_with_final_comments["Id"] = participant_id
        df_with_final_comments["Condition"] = condition
        df_with_final_comments["Map"] = map_type

        first_cols = ["Id", "Condition", "Map"]
        other_cols = [c for c in df_with_final_comments.columns if c not in first_cols]
        df_with_final_comments = df_with_final_comments[first_cols + other_cols]

        all_results.append(df_with_final_comments)

        print(f"✅ הצלחה: {participant_id} | {condition} | Map {map_type}")

    except Exception as e:
        print(f"❌ שגיאה: {participant_id} | {condition} | Map {map_type} | {e}")
        record_fail(condition, map_type, "pipeline", e)
        continue



# -------------------
# WRITE OUTPUT
# -------------------
if all_results:
    df_out = pd.concat(all_results, ignore_index=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ פלט נשמר: {output_path}")
else:
    print("\n⚠️ לא נוצר פלט (הנסיעה נכשלה או דולגה).")

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    print(f"⚠️ דוח כשלונות נשמר: {failed_path}")

print("🏁 הניתוח הושלם.")
