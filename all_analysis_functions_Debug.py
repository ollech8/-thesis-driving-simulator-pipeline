import os
import re
import pandas as pd
import json
import numpy as np
import Andromeda.tidy as meda

# עיבוד נתוני סימולציה מתוך קובץ JSON

def load_json(file_path):
    """טוען קובץ JSON וממיר אותו ל-DataFrame"""
    try:
        df = pd.read_json(file_path)
        if list(df.columns) == ['Logs']:
            df = pd.DataFrame(df['Logs'].tolist())
        return df
    except Exception as e:
        print("Error reading file:", e)
        return pd.DataFrame()

def normalize_type_column(df):
    """
    מיישר קו לעמודת Type כך שכל הווריאציות ימופו לערכים אחידים:
    - GPS
    - Car_Telemetries
    - Termination
    """

    if "Type" not in df.columns:
        return df

    # נירמול בסיסי
    type_norm = (
        df["Type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )

    # מיפוי לערכים קנוניים
    mapping = {
        "gps": "GPS",
        "cartelemetries": "Car_Telemetries",
        "cartelemetry": "Car_Telemetries",
        "telemetries": "Car_Telemetries",
        "termination": "Termination"
    }

    df["Type"] = type_norm.map(mapping).fillna(df["Type"])

    return df

def ensure_columns(df):
    """מוודא שקיימות עמודות חיוניות"""
    if "Type" not in df.columns:
        df = df.reset_index(drop=True)
        df["Type"] = df.index.to_series().apply(lambda x: "GPS" if x % 2 == 0 else "Car_Telemetries")
    if "FrameID" not in df.columns:
        df["FrameID"] = df.index
    return df

# def extract_simulation_position(df):
    """מחלץ נתוני X ו-Y מתוך עמודת SimulationPosition"""
#     if "SimulationPosition" in df.columns:
#         df["PositionX"] = df["SimulationPosition"].apply(lambda pos: pos.get("x", None) if isinstance(pos, dict) else None)
#         df["PositionY"] = df["SimulationPosition"].apply(lambda pos: pos.get("y", None) if isinstance(pos, dict) else None)
    
#     df["PositionX"] = pd.to_numeric(df["PositionX"], errors='coerce')
#     df["PositionY"] = pd.to_numeric(df["PositionY"], errors='coerce')
#     return df

def extract_simulation_position(df):
    """מחלץ נתוני X ו-Y מתוך עמודת SimulationPosition + מחשב Acceleration Magnitude"""
    
    # --- Position ---
    if "SimulationPosition" in df.columns:
        df["PositionX"] = df["SimulationPosition"].apply(lambda pos: pos.get("x", None) if isinstance(pos, dict) else None)
        df["PositionY"] = df["SimulationPosition"].apply(lambda pos: pos.get("y", None) if isinstance(pos, dict) else None)
    
    df["PositionX"] = pd.to_numeric(df["PositionX"], errors='coerce')
    df["PositionY"] = pd.to_numeric(df["PositionY"], errors='coerce')

    # --- Yaw (heading, degrees) -- needed to decompose Speed into a (vx,vy) velocity
    # vector for the time_to_collision vector-based physics ---
    if "Orientation" in df.columns:
        df["Yaw"] = df["Orientation"].apply(lambda o: o.get("y", None) if isinstance(o, dict) else None)
        df["Yaw"] = pd.to_numeric(df["Yaw"], errors="coerce")

    # --- Acceleration Magnitude ---
    if "Acceleration" in df.columns:
        def calc_magnitude(val):
            if isinstance(val, dict):
                x = val.get("x", 0) or 0
                y = val.get("y", 0) or 0
                z = val.get("z", 0) or 0
                return np.sqrt(x**2 + y**2 + z**2)
            return pd.to_numeric(val, errors="coerce")  # אם כבר מספר - תחזיר אותו

        df["Acceleration"] = df["Acceleration"].apply(calc_magnitude)

    return df

def split_dataframes(df):
    """מחלק את הנתונים ל-GPS, Car Telemetries, ו-Termination"""
    gps_columns = ["FrameID", "WorldTime", "SimulationTime", "Longitude", "Latitude",
                   "Acceleration", "Id", "Condition", "Map", "PositionX", "PositionY"]
    if "LaneID" in df.columns:
        gps_columns.append("LaneID")
    if "Yaw" in df.columns:
        gps_columns.append("Yaw")

    df_gps = df[df["Type"] == "GPS"][gps_columns].copy()

    df_tele = df[df["Type"] == "Car_Telemetries"][["FrameID", "Speed", "SteeringAngle", "Brake"]].copy()

    # ✅ חדש: Termination
    term_cols = [c for c in ["FrameID", "WorldTime", "SimulationTime", "Id", "Condition", "Map"] if c in df.columns]
    if "Reason" in df.columns:
        term_cols.append("Reason")
    if "Event_Name" in df.columns:
        term_cols.append("Event_Name")

    df_term = df[df["Type"] == "Termination"][term_cols].copy()

    return df_gps, df_tele, df_term




def merge_dataframes(df_gps, df_tele):
    """
    ממזג GPS ו-Car_Telemetries בשתי צורות:
    1) אם יש חפיפה ב-FrameID -> merge רגיל על FrameID
    2) אם אין חפיפה והדאטה לסירוגין (odd/even) -> התאמת זוגות באמצעות PairID
    * ללא LaneID בפלט
    """

    final_columns = [
        "Id", "Condition", "Map", "FrameID",
        "WorldTime", "SimulationTime",
        "Longitude", "Latitude",
        "PositionX", "PositionY",
        "Acceleration", "Speed", "SteeringAngle", "Brake",
        "Yaw",
    ]

    if df_gps is None or df_tele is None or df_gps.empty or df_tele.empty:
        return pd.DataFrame(columns=final_columns)

    df_gps = df_gps.copy()
    df_tele = df_tele.copy()

    # ודא FrameID מספרי
    df_gps["FrameID"] = pd.to_numeric(df_gps["FrameID"], errors="coerce")
    df_tele["FrameID"] = pd.to_numeric(df_tele["FrameID"], errors="coerce")
    df_gps = df_gps.dropna(subset=["FrameID"])
    df_tele = df_tele.dropna(subset=["FrameID"])
    df_gps["FrameID"] = df_gps["FrameID"].astype(int)
    df_tele["FrameID"] = df_tele["FrameID"].astype(int)

    # -------------------------
    # 1) נסה merge רגיל על FrameID
    # -------------------------
    common = set(df_gps["FrameID"]).intersection(set(df_tele["FrameID"]))
    if len(common) > 0:
        df_merged = pd.merge(df_gps, df_tele, on="FrameID", how="inner")

        # ודא שאין LaneID (גם אם הגיע מהקבצים)
        if "LaneID" in df_merged.columns:
            df_merged = df_merged.drop(columns=["LaneID"])

        for c in final_columns:
            if c not in df_merged.columns:
                df_merged[c] = pd.NA

        return df_merged[final_columns]

    # -------------------------
    # 2) אין חפיפה -> נסה התאמה לסירוגין (odd/even)
    # -------------------------
    gps_parity = (df_gps["FrameID"] % 2).value_counts(normalize=True)
    tele_parity = (df_tele["FrameID"] % 2).value_counts(normalize=True)

    gps_major = gps_parity.idxmax() if not gps_parity.empty else None
    tele_major = tele_parity.idxmax() if not tele_parity.empty else None

    if gps_major is not None and tele_major is not None and gps_major != tele_major:
        # תרחיש קלאסי: Tele = Gps + 1
        df_gps["PairID"] = df_gps["FrameID"]
        df_tele["PairID"] = df_tele["FrameID"] - 1

        df_merged = pd.merge(
            df_gps,
            df_tele[["PairID", "Speed", "SteeringAngle", "Brake"]],
            on="PairID",
            how="inner"
        )

        # אם לא הצליח, ננסה Tele = Gps - 1
        if df_merged.empty:
            df_tele["PairID"] = df_tele["FrameID"] + 1
            df_merged = pd.merge(
                df_gps,
                df_tele[["PairID", "Speed", "SteeringAngle", "Brake"]],
                on="PairID",
                how="inner"
            )

        if df_merged.empty:
            return pd.DataFrame(columns=final_columns)

        # FrameID נשאיר של ה-GPS (PairID)
        df_merged["FrameID"] = df_merged["PairID"]
        df_merged = df_merged.drop(columns=["PairID"])

        # ודא שאין LaneID (גם אם הגיע מהקבצים)
        if "LaneID" in df_merged.columns:
            df_merged = df_merged.drop(columns=["LaneID"])

        for c in final_columns:
            if c not in df_merged.columns:
                df_merged[c] = pd.NA

        return df_merged[final_columns]

    return pd.DataFrame(columns=final_columns)





def process_file(file_path, participant_id=None, condition=None, map_type=None):
    """מבצע את כל שלבי העיבוד על קובץ JSON יחיד"""

    df = load_json(file_path)

    # 🔥 יישור קו אחד לכל הפייפליין
    df = normalize_type_column(df)
    # print(df["Type"].value_counts())


    df = ensure_columns(df)
    df = extract_simulation_position(df)

    # הוספת מזהים
    df["Id"] = participant_id
    df["Condition"] = condition
    df["Map"] = map_type

    df_gps, df_tele, df_term = split_dataframes(df)


    
    common = set(df_gps["FrameID"]).intersection(set(df_tele["FrameID"]))



    df_merged = merge_dataframes(df_gps, df_tele)

    
    # ✅ הוספת Termination לפלט הסופי (כדי שהזמן ירוץ עד הסוף)
    if df_term is not None and not df_term.empty:
        # ודא שיש עמודה Type בפלט כדי לזהות Termination
        if "Type" not in df_merged.columns:
            df_merged["Type"] = "Merged"  # ערך ברירת מחדל לשורות הרגילות
    
        df_term = df_term.copy()
        df_term["Type"] = "Termination"  # סימון ברור
    
        # יישור עמודות (כל מה שחסר -> NA)
        for c in df_merged.columns:
            if c not in df_term.columns:
                df_term[c] = pd.NA
        for c in df_term.columns:
            if c not in df_merged.columns:
                df_merged[c] = pd.NA
    
        # סדר עמודות כמו df_merged
        df_term = df_term[df_merged.columns]
    
        # concat + מיון בזמן כדי שה-Termination יהיה בסוף
        df_merged = pd.concat([df_merged, df_term], ignore_index=True, sort=False)
        df_merged = df_merged.sort_values("SimulationTime", kind="mergesort").reset_index(drop=True)
        df_merged = df_merged.drop(columns=["Type", "Reason"], errors="ignore")


    return df_merged



############################################################################

## הוספת אירועים קינמטיים לנתוני הרכב


    
def load_kinematic_events(file_path):
    """טוען את קובץ האירועים הקינמטיים ומחזיר DataFrame מסונן (EgoCar בלבד)"""
    df_kinematic = pd.read_csv(file_path)

    # --- סינון לרכב EgoCar בלבד ---
    if "triggered_by" not in df_kinematic.columns:
        raise ValueError("Missing 'triggered_by' column in kinematic events file")

    df_kinematic = df_kinematic[df_kinematic["triggered_by"] == "Egocar"].copy()

    # --- סינון לפי סוגי אירועים קינמטיים ---
    kinematic_event_types = ["Braking", "Accelerating", "TurnRight", "TurnLeft"]
    df_kinematic = df_kinematic[df_kinematic["Event_Name"].isin(kinematic_event_types)]

    df_kinematic = df_kinematic[
        ["Id", "Condition", "Map", "Onset.SimulationTime", "End.SimulationTime", "Event_Name"]
    ]

    return df_kinematic, kinematic_event_types

def add_kinematic_events(df_merged, df_kinematic, kinematic_event_types):
    """מוסיף עמודות בינאריות לנתוני הרכב עבור אירועים קינמטיים"""
    for event in kinematic_event_types:
        df_merged[event] = 0  # ברירת מחדל - אין אירוע
    
    for index, row in df_kinematic.iterrows():
        mask = (
            (df_merged["Id"] == row["Id"]) &
            (df_merged["Condition"] == row["Condition"]) &
            (df_merged["Map"] == row["Map"]) &
            (df_merged["SimulationTime"] >= row["Onset.SimulationTime"]) &
            (df_merged["SimulationTime"] <= row["End.SimulationTime"])
        )
        df_merged.loc[mask, row["Event_Name"]] = 1  # שינוי הערך ל-1 בזמן שהאירוע מתרחש
    
    return df_merged

def process_kinematic_data(file_path, df_merged):
    """טוען ומוסיף אירועים קינמטיים ל-DataFrame ממוזג"""
    df_kinematic, kinematic_event_types = load_kinematic_events(file_path)
    df_merged = add_kinematic_events(df_merged, df_kinematic, kinematic_event_types)
    return df_merged

######################################################################################################
## שילוב אירועים מרחביים בנתוני הרכב


def load_spacial_events(file_path):
    """טוען את קובץ האירועים הספציאליים ומחזיר DataFrame מסונן (EgoCar בלבד)"""
    df_spacial = pd.read_csv(file_path)

    df_spacial = df_spacial[df_spacial["triggered_by"] == "Egocar"].copy()

    # --- המשך לוגיקה קיימת ---
    df_spacial["Type"] = df_spacial["Type"].str.strip()  # ניקוי רווחים למניעת בעיות סינון
    df_spacial = df_spacial[
    df_spacial["Type"].str.contains("Spacial", na=False) |
    df_spacial["Type"].str.contains("Start/End Simulation", na=False)].copy()
   # אם אין עמודת Reason – ניצור אותה
    if "Reason" not in df_spacial.columns:
        df_spacial["Reason"] = pd.NA

    df_spacial = df_spacial[
        ["Id", "Condition", "SimulationTime", "Event_Name", "Reason"]
    ].copy()

    df_spacial = df_spacial.rename(columns={"Event_Name": "SpacialEvent"})

    # נרמול שמות לא עקביים לאירוע הולך רגל: השורה האמצעית (בין Start section
    # walkerN ל-End section walkerN) נקראת לפעמים "Walker1"/"Walker2"/"Walker4"
    # ולפעמים "Pedestrian3" (מפה B) במקום "section walkerN" -- מאחדים לשם אחיד
    # כדי שכל שלבי אותו אירוע (התחלה/אמצע/סוף) ייקראו באותו אופן.
    def _normalize_bare_walker_name(name):
        if pd.isna(name):
            return name
        s = str(name).strip()
        m = re.match(r"(?i)^(walker|pedestrian)\s*(\d+)$", s)
        if m:
            return f"section walker{m.group(2)}"
        return name

    df_spacial["SpacialEvent"] = df_spacial["SpacialEvent"].apply(_normalize_bare_walker_name)

    # רק Termination שומר Reason, השאר נשארים NaN
    df_spacial.loc[df_spacial["SpacialEvent"] != "Termination", "Reason"] = pd.NA


    return df_spacial

  

def merge_spacial_events(df_merged, df_spacial):
    """ממזג את אירועי ה-Spacial לתוך ה-DataFrame הראשי לפי SimulationTime, Id, ו-Condition"""
    df_spacial["SimulationTime"] = df_spacial["SimulationTime"].round(1)
    df_merged["SimulationTime"] = df_merged["SimulationTime"].round(1)
    df_merged = pd.merge(df_merged, df_spacial, on=["Id", "Condition", "SimulationTime"], how="left")
    df_merged["SpacialEvent"] = df_merged["SpacialEvent"].fillna("None")
    
    return df_merged

def synthesize_missing_tl_start_events(df_spacial):
    """
    For any (Id, Condition) that has 'Traffic light N' or 'End traffic light N'
    but no 'Start traffic light N', insert a synthetic start event estimated as:
        StartPoint_time + median_delay_from_other_participants_same_condition

    Outlier delays (> mean + 3*std) are excluded before computing the median.
    """
    synth_rows = []

    for (pid, cond), group in df_spacial.groupby(["Id", "Condition"]):
        tl_group = group[
            group["SpacialEvent"].str.contains("traffic light", case=False, na=False)
        ]

        # Collect all TL numbers present for this participant/condition
        tl_nums = set()
        for ev in tl_group["SpacialEvent"].dropna():
            m = re.search(r"traffic light (\d+)", str(ev), re.IGNORECASE)
            if m:
                tl_nums.add(int(m.group(1)))

        for tl_num in sorted(tl_nums):
            has_start = tl_group["SpacialEvent"].str.contains(
                f"start traffic light {tl_num}", case=False, na=False
            ).any()
            if has_start:
                continue  # already present, nothing to do

            # Find this participant's StartPoint time
            sp_rows = group[group["SpacialEvent"] == "StartPoint"]
            if sp_rows.empty:
                continue
            sp_time = float(sp_rows["SimulationTime"].iloc[0])

            # Compute median delay from other participants (same condition) who have the start event
            others_tl = df_spacial[
                (df_spacial["Condition"] == cond) &
                (df_spacial["SpacialEvent"].str.contains(
                    f"start traffic light {tl_num}", case=False, na=False)) &
                (df_spacial["Id"] != pid)
            ][["Id", "SimulationTime"]].rename(columns={"SimulationTime": "tl_t"})

            others_sp = df_spacial[
                (df_spacial["Condition"] == cond) &
                (df_spacial["SpacialEvent"] == "StartPoint") &
                (df_spacial["Id"].isin(others_tl["Id"]))
            ][["Id", "SimulationTime"]].rename(columns={"SimulationTime": "sp_t"})

            delays_df = pd.merge(others_sp, others_tl, on="Id")
            if delays_df.empty:
                continue

            delays = delays_df["tl_t"] - delays_df["sp_t"]
            # Exclude outliers > mean + 3*std
            threshold = delays.mean() + 3 * delays.std()
            clean_delays = delays[delays <= threshold]
            median_delay = float(clean_delays.median())

            synth_time = sp_time + median_delay
            print(f"SYNTH: inserting 'Start traffic light {tl_num}' for {pid}/{cond} "
                  f"at t={synth_time:.3f} (StartPoint={sp_time:.3f} + delay={median_delay:.3f}s)")

            synth_rows.append({
                "Id": pid,
                "Condition": cond,
                "SimulationTime": synth_time,
                "SpacialEvent": f"Start traffic light {tl_num}",
                "Reason": pd.NA,
            })

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_spacial = pd.concat([df_spacial, df_synth], ignore_index=True)
        df_spacial = df_spacial.sort_values(
            ["Id", "Condition", "SimulationTime"]
        ).reset_index(drop=True)

    return df_spacial


def process_spacial_data(file_path, df_merged):
    """מבצע את כל שלבי העיבוד: טוען את האירועים הספציאליים, מסנן, וממזג אותם ל-DataFrame הראשי"""
    df_spacial = load_spacial_events(file_path)
    df_spacial = synthesize_missing_tl_start_events(df_spacial)
    df_merged = merge_spacial_events(df_merged, df_spacial)
    return df_merged

###########################################################################################################
## עיבוד אירועים מרחביים בנתוני הסימולציה

def clean_event_name(event):
    """
    מנקה את שמות האירועים על ידי הסרת מילים כמו 'start' ו-'end',
    הסרת תווים מיוחדים והפחתת רווחים כפולים.
    """
    if pd.isna(event) or event in ['StartPoint','Start', 'None', 'Termination', 'EndPoint']:
        return event  # שומר על הערכים המיוחדים כפי שהם
    event = event.lower().strip()
    event = re.sub(r"\b(start|end)\b", "", event)  # מסיר 'start' או 'end'
    event = re.sub(r"[^a-zA-Z0-9\s]", "", event)  # מסיר תווים מיוחדים
    event = re.sub(r"\s+", " ", event)  # מחליף רווחים כפולים ברווח בודד
    return event.strip()


def prepare_spacial_events(df_final):
    """
    מוסיף עמודת בסיס לאירועים אחרי ניקוי שמותיהם.
    """
    df_final["SimulationTime"] = df_final["SimulationTime"].astype(float)
    df_final["BaseEvent"] = df_final["SpacialEvent"].apply(clean_event_name)
    return df_final

def separate_start_end_events(df_final):
    """
    מחלק את הנתונים לטבלאות אירועי התחלה וסיום על פי שמות האירועים.
    """
    df_start = df_final[
        (df_final["SpacialEvent"].str.contains("start", case=False, na=False)) &
        (~df_final["SpacialEvent"].isin(['StartPoint','Start', 'None', 'Termination', 'EndPoint']))
    ].copy()

    df_end = df_final[
        (df_final["SpacialEvent"].str.contains("end", case=False, na=False)) &
        (~df_final["SpacialEvent"].isin(['StartPoint','Start', 'None', 'Termination', 'EndPoint']))
    ].copy()

    # הסרת כפילויות
    df_start = df_start.drop_duplicates(subset=["Id", "Condition", "BaseEvent"], keep="first")
    df_end = df_end.drop_duplicates(subset=["Id", "Condition", "BaseEvent"], keep="first")

    # Fallback: if an end event exists but no start event, treat the plain entry
    # event (no start/end prefix) as the start so the window still gets filled.
    end_base_events = set(df_end["BaseEvent"].dropna())
    start_base_events = set(df_start["BaseEvent"].dropna())
    missing_starts = end_base_events - start_base_events

    if missing_starts:
        invalid = {'StartPoint', 'Start', 'None', 'Termination', 'EndPoint'}
        df_entry_as_start = df_final[
            df_final["BaseEvent"].isin(missing_starts) &
            (~df_final["SpacialEvent"].str.contains("end", case=False, na=False)) &
            (~df_final["SpacialEvent"].isin(invalid))
        ].drop_duplicates(subset=["Id", "Condition", "BaseEvent"], keep="first")
        df_start = pd.concat([df_start, df_entry_as_start], ignore_index=True)

    return df_start, df_end

def merge_event_pairs(df_start, df_end):
    """
    ממזג את אירועי ההתחלה והסיום כדי ליצור זוגות אירועים.
    """
    df_pairs = pd.merge(df_start, df_end, on=["Id", "Condition", "BaseEvent"], 
                        suffixes=("_start", "_end"))
    return df_pairs

def fill_intermediate_events(df_final, df_pairs):
    """
    מעדכן את ערכי `SpacialEvent` בשורות שבין התחלה לסיום עם שם האירוע המתאים.
    """
    for index, row in df_pairs.iterrows():
        start_time = row["SimulationTime_start"]
        end_time = row["SimulationTime_end"]
        event_name = row["BaseEvent"]

        mask = (
            (df_final["Id"] == row["Id"]) &
            (df_final["Condition"] == row["Condition"]) &
            (df_final["SimulationTime"] > start_time) &  # שינוי: > במקום >=
            (df_final["SimulationTime"] < end_time)      # שינוי: < במקום <=
        )

        df_final.loc[mask, "SpacialEvent"] = event_name

    return df_final

import pandas as pd

def fill_collision_tail_before_termination(
    df_final: pd.DataFrame,
    collided_pattern: str = "collided-",
    invalid_events=None,
) -> pd.DataFrame:
    """
    ממלא את הערכים הריקים (None/NaN) בעמודת SpacialEvent
    מהשורה שאחרי האירוע האחרון התקף ועד לשורה שלפני ה-Termination הראשון (לא כולל),
    רק אם ה-Termination הראשון הוא Termination עם Reason שמכיל collided-.

    לא משנה את שורת ה-Termination עצמה.
    """

    if invalid_events is None:
        invalid_events = {"StartPoint", "Start", "EndPoint", "None", "Termination"}

    df = df_final.copy()

    # ודא ש-Reason קיים
    if "Reason" not in df.columns:
        df["Reason"] = pd.NA

    # ודא שיש BaseEvent (נשתמש ב-clean_event_name שלך)
    if "BaseEvent" not in df.columns:
        if "clean_event_name" not in globals():
            raise NameError("clean_event_name לא מוגדרת בסביבה. ודא שהפונקציה קיימת לפני הקריאה.")
        df["BaseEvent"] = df["SpacialEvent"].apply(clean_event_name)

    # עבודה לפי נסיעה
    for (trip_id, cond), g_idx in df.groupby(["Id", "Condition"], sort=False).groups.items():
        g = df.loc[g_idx].sort_values("SimulationTime", kind="mergesort")
        g_indices = g.index.tolist()

        # מציאת Termination ראשון עם collided-
        term_mask = (
            (g["SpacialEvent"] == "Termination")
            & (g["Reason"].astype(str).str.contains(collided_pattern, case=False, na=False))
        )
        if not term_mask.any():
            continue

        termination_idx = g[term_mask].index[0]  # הראשון בזמן
        termination_pos = g_indices.index(termination_idx)

        # אין מה למלא אם ה-Termination הוא ממש בתחילת הקבוצה
        if termination_pos == 0:
            continue

        # מציאת האירוע האחרון התקף לפני ה-Termination
        before = g.iloc[:termination_pos].copy()
        valid_mask = (
            before["SpacialEvent"].notna()
            & (~before["SpacialEvent"].isin(invalid_events))
            & (before["SpacialEvent"].astype(str).str.strip() != "")
        )
        if not valid_mask.any():
            continue

        last_event_idx = before[valid_mask].index[-1]
        last_event_pos = g_indices.index(last_event_idx)

        # BaseEvent של האירוע האחרון
        last_spacial_event = df.at[last_event_idx, "SpacialEvent"]
        base_event = clean_event_name(last_spacial_event)

        # הטווח למילוי: משורה אחרי האירוע האחרון ועד שורה לפני Termination
        start_pos = last_event_pos + 1
        end_pos = termination_pos - 1
        if start_pos > end_pos:
            continue

        fill_indices = g_indices[start_pos : end_pos + 1]

        # ממלאים רק ערכים ריקים (None/NaN/"None"/רווחים)
        is_empty = (
            df.loc[fill_indices, "SpacialEvent"].isna()
            | (df.loc[fill_indices, "SpacialEvent"] == "None")
            | (df.loc[fill_indices, "SpacialEvent"].astype(str).str.strip() == "")
        )

        df.loc[[i for i in fill_indices if is_empty.loc[i]], "SpacialEvent"] = base_event
        # אם תרצה שגם BaseEvent יתיישר בהתאם:
        df.loc[[i for i in fill_indices if is_empty.loc[i]], "BaseEvent"] = base_event

    return df


import numpy as np
import pandas as pd

def _worldtime_to_seconds(series: pd.Series) -> pd.Series:
    """
    ממיר WorldTime לסקאלה מספרית של שניות.
    תומך ב:
    - מספרים (נשאר כמו שהוא)
    - Timedelta strings (00:00:01.234, '0 days 00:00:01.234')
    - Datetime strings ('2025-03-09 19:09:46.123')
    """
    # 1) נסה numeric
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        # אם רוב הערכים הפכו למספרים – כנראה זה כבר numeric
        if s_num.notna().mean() > 0.5:
            return s_num

    # 2) נסה to_timedelta
    s_td = pd.to_timedelta(series.astype(str), errors="coerce")
    if s_td.notna().any():
        return s_td.dt.total_seconds()

    # 3) נסה to_datetime
    s_dt = pd.to_datetime(series, errors="coerce")
    if s_dt.notna().any():
        # ממירים לשניות יחסית לראשון התקף (כדי לקבל ציר זמן מתחיל מ-0)
        first = s_dt.dropna().iloc[0]
        return (s_dt - first).dt.total_seconds()

    # אם כלום לא עבד
    return pd.Series(np.nan, index=series.index)


def add_time_since_event_start_world_special(
    df_final: pd.DataFrame,
    world_time_col: str = "WorldTime",
    event_col: str = "SpacialEvent",   # שים לב: אצלך זה SpacialEvent
    invalid_events=None,
) -> pd.DataFrame:
    """
    מוסיף רק עמודה אחת:
    time_since_event_start_world

    מבוסס על event_col (SpacialEvent) ולא BaseEvent.
    """

    if invalid_events is None:
        invalid_events = {"StartPoint", "Start", "EndPoint", "None", "Termination"}

    df = df_final.copy()

    if world_time_col not in df.columns:
        raise KeyError(f"Missing column '{world_time_col}'")
    if event_col not in df.columns:
        raise KeyError(f"Missing column '{event_col}'")

    # המרה חכמה לשניות
    world_sec = _worldtime_to_seconds(df[world_time_col])

    # אם הכל NaN – זו בדיוק הבעיה שדיברנו עליה
    if world_sec.notna().mean() == 0:
        # נשאיר הכל NaN אבל עם הדפסה אחת ברורה
        print(f"❌ WorldTime conversion failed: column '{world_time_col}' could not be parsed to numeric/timedelta/datetime.")
        df["time_since_event_start_world"] = np.nan
        return df

    df["time_since_event_start_world"] = np.nan

    # ניקוי אירוע (אם יש clean_event_name, נשתמש בה, בלי ליצור עמודות)
    if "clean_event_name" in globals():
        norm = lambda x: clean_event_name(x)
    else:
        norm = lambda x: x

    # עבודה לפי נסיעה
    for (trip_id, cond), g_idx in df.groupby(["Id", "Condition"], sort=False).groups.items():
        g = df.loc[g_idx].copy()
        # סדר לפי זמן עולם (שניות)
        g["_world_sec"] = world_sec.loc[g_idx].values
        g = g.sort_values("_world_sec", kind="mergesort")

        ev = g[event_col].apply(norm)
        ev_str = ev.astype(str).str.strip()

        invalid_lower = {s.lower() for s in invalid_events}
        valid = (
            ev.notna()
            & (ev_str != "")
            & (~ev_str.isin(invalid_events))
            & (~ev_str.str.lower().isin(invalid_lower))
        )

        if not valid.any():
            continue

        prev_valid = valid.shift(1, fill_value=False)
        prev_ev = ev.shift(1)

        start_instance = valid & (~prev_valid | (ev != prev_ev))

        start_world = g["_world_sec"].where(start_instance).ffill()
        time_since = (g["_world_sec"] - start_world).where(valid, np.nan)

        df.loc[g.index, "time_since_event_start_world"] = time_since.values

    return df



# --- חיבור לפייפליין שלך: בסוף process_spacial_events או מיד אחרי ---
def process_spacial_events(df_final):
    """
    מבצע את כל שלבי העיבוד על אירועים ספציאליים:
    - ניקוי שמות אירועים
    - חלוקה לאירועי התחלה וסיום
    - יצירת זוגות של אירועים
    - מילוי שורות ביניים באירועים המתאימים
    - טיפול ב-Termination ראשון עם collided-: מילוי ערכים ריקים עד לפני Termination
    - תוספת: זמן שעבר מאז תחילת האירוע לפי WorldTime
    """
    df_final = prepare_spacial_events(df_final)
    df_start, df_end = separate_start_end_events(df_final)
    df_pairs = merge_event_pairs(df_start, df_end)
    df_final = fill_intermediate_events(df_final, df_pairs)

    # תוספת: מילוי "זנב" לפני Termination ראשון עם collided-
    df_final = fill_collision_tail_before_termination(df_final)

    # תיקון: BaseEvent חושב לפני מילוי השורות הביניים ב-fill_intermediate_events,
    # לכן הוא נשאר "None" עבור שורות שה-SpacialEvent שלהן מולא רק אחר כך.
    # מחשבים אותו שוב עכשיו, אחרי שה-SpacialEvent הסופי כבר קיים.
    df_final = prepare_spacial_events(df_final)

    # תוספת חדשה: זמן מאז תחילת האירוע (WorldTime)
    df_final = add_time_since_event_start_world_special(df_final)

    return df_final





##############################################################################################################

## סיווג אירועים מרחביים לקטגוריות
def categorize_event(event):
    """
    מסווג אירועים מיוחדים לפי שמות מעודכנים של הסימולציה.
    מתעלם מ־Start / End / Egocar / מספרים.
    """
    if pd.isna(event):
        return None

    event = event.lower().strip()

    # 🔹 נקודות התחלה / סיום סימולציה
    if re.search(r"\b(startpoint|endpoint)\b", event):
        return "SimulationPoints"

    # 🚶 הולכי רגל (walker1, section walker2, וכו')
    elif re.search(r"\bwalker\s*\d*\b", event):
        return "Pedestrians"

    # 🚗 עקיפה
    elif re.search(r"\bovertake\b", event):
        return "Overtake"

    # 🚦 רמזורים
    elif re.search(r"\btraffic light\s*\d*\b", event):
        return "TrafficLights"

    # ↔️ Gap Acceptance
    elif re.search(r"\bgap acceptance\b", event):
        return "GapAcceptance"

    return None


def add_event_categories(df_final):
    """
    מוסיף עמודת קטגוריה 'event_category' לכל שורה לפי סיווג האירוע,
    ומוסיף **עמודות בינאריות רק עבור Pedestrians ו-OverTake**.
    """

    # 🔹 קביעת קטגוריית האירוע
    df_final["event_category"] = df_final["SpacialEvent"].apply(categorize_event)

    # 🔹 יצירת עמודות בינאריות  עבור car_infront_SuddenStop Pedestrians ו- Barriers
    #df_final["Pedestrians"] = (df_final["event_category"] == "Pedestrians").astype(int)
    # df_final["overtake"] = (df_final["event_category"] == "overtake").astype(int)

    return df_final

def process_event_categorization(df_final):
    """
    מבצע את תהליך סיווג האירועים, כולל יצירת עמודות בינאריות **רק עבור Pedestrians ו- Barriers**.
    מחזיר דאטהפריים חדש בשם `df_categorized_events`.
    """
    df_categorized_events = add_event_categories(df_final.copy())

    return df_categorized_events
 
###############################################################################################################

## טעינת קובץ objects

def load_json_file(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # אם הנתונים הם מילון עם מפתח "Logs", נשתמש בו
    if isinstance(data, dict) and "Logs" in data:
        return pd.json_normalize(data["Logs"])
    else:
        return pd.DataFrame(data)

def convert_columns_to_numeric(df, columns):

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # המרה עם טיפול בערכים שגויים
    return df

def sort_by_simulation_time(df):

    return df.sort_values("SimulationTime")

def process_objects_data(file_path):

    df_objects = load_json_file(file_path)
    
    # המרת עמודות מספריות (נניח שאלה השמות, ניתן לשנות בהתאם)
    numeric_columns = ["SimulationTime", "SimulationPosition.x", "SimulationPosition.y"]
    df_objects = convert_columns_to_numeric(df_objects, numeric_columns)

    # מיון לפי זמן הסימולציה
    df_sorted = sort_by_simulation_time(df_objects)

    return df_sorted
######################################################################################################################

## חיפוש אובייקטים קרובים לאירועים מרחביים

def load_object_points(object_points_path):
    """
    טוען את נתוני האירועים המרחביים מקובץ CSV.

    פרמטרים:
    - object_points_path (str): נתיב לקובץ ה-CSV.

    מחזיר:
    - DataFrame עם נתוני האירועים.
    """
    return pd.read_csv(object_points_path)

def filter_object_points_by_map(df_object_points, df_categorized_events):
    """
    מסנן את נתוני האירועים כך שיכללו רק את המפות הקיימות ב- df_categorized_events.

    פרמטרים:
    - df_object_points (pd.DataFrame): DataFrame עם האירועים המרחביים.
    - df_categorized_events (pd.DataFrame): DataFrame עם נתוני הסימולציה והמפות.

    מחזיר:
    - DataFrame מסונן לפי מפות רלוונטיות בלבד.
    """
    unique_maps = df_categorized_events["Map"].unique()
    
    if len(unique_maps) > 1:
        print(f"⚠️ זוהו מספר מפות: {unique_maps}. יתבצע סינון לפי כולן.")

    return df_object_points[df_object_points["Map"].isin(unique_maps)]

def find_matching_objects(df_object_points, df_objects, tolerance=0.5):
    """
    מחפש אובייקטים קרובים לאירועים מרחביים ומחזיר טבלה מקושרת.

    פרמטרים:
    - df_object_points (pd.DataFrame): נתוני האירועים המרחביים.
    - df_objects (pd.DataFrame): נתוני האובייקטים.
    - tolerance (float): טולרנס מרחק לחיפוש לפי X ו-Y (ברירת מחדל: 0.5).

    מחזיר:
    - DataFrame עם עמודות:
        - "SpacialEvent" (שם האירוע)
        - "relevant_object_name" (רשימה משורשרת של אובייקטים רלוונטיים לכל אירוע, ללא כפילויות)
    """
    matched_objects = []

    for _, row in df_object_points.iterrows():
        event = row["SpacialEvent"]

        # חיפוש לפי מזהה רמזור (אם קיים)
        if pd.notna(row.get("traffic_light_id")):
            traffic_light_id = int(row["traffic_light_id"])
            matched_rows = df_objects[
                df_objects["Name"].str.startswith("traffic.traffic_light", na=False) & 
                df_objects["Name"].str.endswith(str(traffic_light_id))
            ]
        
        else:  # חיפוש לפי קואורדינטות X ו-Y
            x_target, y_target = row["SimulationPosition.x"], row["SimulationPosition.y"]

            matched_rows = df_objects[
                (np.abs(df_objects["SimulationPosition.x"] - x_target) <= tolerance) &
                (np.abs(df_objects["SimulationPosition.y"] - y_target) <= tolerance)
            ]

        # אחסון תוצאות ההתאמה
        object_names = set(matched_rows["Name"].tolist())  # שימוש ב- set() כדי למנוע כפילויות

        if object_names:
            matched_objects.append({
                "SpacialEvent": event,
                "relevant_object_name": ", ".join(object_names)
            })

    return pd.DataFrame(matched_objects)

def aggregate_matched_objects(df_matched_objects):
    """
    מאחד את רשימת האובייקטים המשויכים לכל אירוע למחרוזת אחת.

    פרמטרים:
    - df_matched_objects (pd.DataFrame): DataFrame עם התאמות אירועים-אובייקטים.

    מחזיר:
    - DataFrame מקובץ עם עמודות:
        - "SpacialEvent" (שם האירוע)
        - "relevant_object_name" (רשימת אובייקטים משויכים לאירוע)
    """
    return df_matched_objects.groupby("SpacialEvent")["relevant_object_name"].apply(lambda x: ", ".join(set(x))).reset_index()

def find_relevant_objects(object_points_path, df_objects, df_categorized_events, tolerance=0.5):
    """
    מחפש אובייקטים קרובים לאירועים מרחביים בקובץ objectPoints.csv
    ומסנן לפי המפה המתאימה (Map) המופיעה ב- df_categorized_events.

    פרמטרים:
    - object_points_path (str): נתיב לקובץ ה-CSV.
    - df_objects (pd.DataFrame): נתוני האובייקטים בעולם הסימולציה.
    - df_categorized_events (pd.DataFrame): נתוני הסימולציה הכוללים אירועים ומפות.
    - tolerance (float): טולרנס לחיפוש לפי X ו-Y (ברירת מחדל: 0.5).

    מחזיר:
    - DataFrame עם עמודות:
        - "SpacialEvent" (שם האירוע)
        - "relevant_object_name" (רשימה משורשרת של אובייקטים רלוונטיים לכל אירוע)
    """
    df_object_points = load_object_points(object_points_path)
    df_object_points = filter_object_points_by_map(df_object_points, df_categorized_events)
    df_matched_objects = find_matching_objects(df_object_points, df_objects, tolerance)
    df_unique_events_objects = aggregate_matched_objects(df_matched_objects)

    return df_unique_events_objects

############################################################################################################################

## הוספת timestamp לבלימה ועצירת הרכב בעמודת car_infront_SuddenStop

def add_sudden_stop_phases(df_categorized_events, df_unique_events_objects, df_objects):
    df_updated = df_categorized_events.copy()

    # חיפוש שורת אירוע המכילה את המילים "overtake moving"    # ✅ לוודא שעמודת Overtake קיימת
    if 'Overtake' not in df_updated.columns:
        df_updated['Overtake'] = None

    # חיפוש שורת אירוע המכילה את המילים "overtake moving"
    relevant_rows = df_unique_events_objects[
        df_unique_events_objects['SpacialEvent'].str.contains("overtake", case=False, na=False)
    ]
    if relevant_rows.empty:
        print("❌ No matching event found with 'overtake'")
        return df_updated

    # ניקח את השורה הראשונה שתואמת
    relevant_row = relevant_rows.iloc[0]
    object_name = relevant_row['relevant_object_name']

    # סינון נתוני האובייקט
    object_data = df_objects[df_objects['Name'] == object_name].copy()
    object_data.sort_values(by='SimulationTime', inplace=True)
    object_data.reset_index(drop=True, inplace=True)

    # בדיקת התחלה בעצירה
    if object_data.iloc[0]['Speed'] < 0.5:
        moving_points = object_data[object_data['Speed'] > 1]
        if moving_points.empty:
            return df_updated
        movement_start_time = moving_points.iloc[0]['SimulationTime']
        object_data = object_data[object_data['SimulationTime'] >= movement_start_time].copy()
    else:
        movement_start_time = object_data.iloc[0]['SimulationTime']

    # חישוב ירידת מהירות על פני 5 פריימים
    object_data['speed_drop_5_frames'] = object_data['Speed'] - object_data['Speed'].shift(-5)
    deceleration_candidates = object_data[object_data['speed_drop_5_frames'] >= 5]
    deceleration_time = deceleration_candidates.iloc[0]['SimulationTime'] if not deceleration_candidates.empty else None

    stop_candidates = object_data[object_data['Speed'] < 0.5]
    stop_time = stop_candidates.iloc[0]['SimulationTime'] if not stop_candidates.empty else None

    restart_time = None
    if stop_time:
        after_stop = object_data[object_data['SimulationTime'] > stop_time]
        restart_candidates = after_stop[after_stop['Speed'] > 1]
        if not restart_candidates.empty:
            restart_time = restart_candidates.iloc[0]['SimulationTime']

    # עדכון עמודת car_infront_SuddenStop לפי הזמנים
    for label, sim_time in [('start drive', movement_start_time),
                            ('start brake', deceleration_time),
                            ('stop', stop_time),
                            ('restart drive', restart_time)]:
        if sim_time is not None:
            closest_idx = (df_updated['SimulationTime'] - sim_time).abs().idxmin()
            df_updated.at[closest_idx, 'Overtake'] = label

    return df_updated

###################################################################################################################

##  חישוב מרחק מהרכב לאובייקטים רלוונטיים בכל אירוע

def calculate_all_event_distances(df_categorized_events, df_objects, df_unique_events_objects):
    """
    מחשב מרחקים בין הרכב לכל האובייקטים הרלוונטיים עבור כל האירועים המיוחדים,
    עם התאמות לפי מפה, כיוון תנועה וסינון רכבים מסוג שוטר כשנדרש.
    """
    df_with_distances = df_categorized_events.copy()
    df_with_distances["distance_from_relevant_object"] = np.nan
    df_with_distances["relevant_object_name"] = None

    for _, event_row in df_unique_events_objects.iterrows():
        event_raw = event_row["SpacialEvent"]

        # ניקוי שם האירוע
        cleaned_event = re.sub(r"\b(start|end|egocar)\b", "", event_raw, flags=re.IGNORECASE).strip()
        cleaned_event = re.sub(r"\s{2,}", " ", cleaned_event)

        event_mask = df_with_distances["SpacialEvent"].str.contains(cleaned_event, case=False, na=False)
        df_event = df_with_distances[event_mask].copy()

        if df_event.empty:
            continue

        map_type = df_event["Map"].iloc[0] if "Map" in df_event.columns else "Unknown"

        relevant_objects = event_row["relevant_object_name"].split(", ")
        is_single_object = len(relevant_objects) == 1

        if not is_single_object and map_type in ["A", "B"]:
            # סינון רכבי שוטר אם יש יותר מאובייקט
            relevant_objects = [
                obj for obj in relevant_objects
                if not re.match(r"vehicle\.dodge\.charger_police\s*\d*$", obj)
            ]

        for idx, row in df_event.iterrows():
            t = row["SimulationTime"]
            lat_vehicle = float(row["Latitude"])
            lon_vehicle = float(row["Longitude"])

            closest_distance = np.inf
            closest_object = None

            for obj_name in relevant_objects:
                df_obj = df_objects[df_objects["Name"] == obj_name]
                if df_obj.empty:
                    continue

                df_obj = df_obj.copy()
                df_obj["Longitude"] = pd.to_numeric(df_obj["Longitude"], errors="coerce")
                df_obj["Latitude"] = pd.to_numeric(df_obj["Latitude"], errors="coerce")

                closest_idx = df_obj["SimulationTime"].sub(t).abs().idxmin()
                obj_row = df_obj.loc[closest_idx]

                lat_obj = obj_row["Latitude"]
                lon_obj = obj_row["Longitude"]

                # 🧭 כיוון תנועה
                if is_single_object:
                    is_ahead = True  # אם רק אובייקט אחד – לא בודקים כיוון
                elif map_type in ["A", "B"]:
                    is_ahead = lon_obj <= lon_vehicle  # מזרח למערב
                else:  # מפה C או אחרת
                    is_ahead = lon_obj >= lon_vehicle  # מערב למזרח

                if not is_ahead:
                    continue

                distance = meda.distanceHaversineVectors(
                    [lat_vehicle], [lon_vehicle],
                    [lat_obj], [lon_obj]
                )[0]

                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj_name

            if closest_object:
                df_with_distances.at[idx, "distance_from_relevant_object"] = closest_distance
                df_with_distances.at[idx, "relevant_object_name"] = closest_object

    return df_with_distances

###################################################################################################################

#  מוסיף עמודה חדשה (ברירת מחדל: 'Padastrians') ומסמן את תחילת הופעת כל הולך רגל.


def add_pedestrian_stage_markers(
    df_with_distances: pd.DataFrame,
    df_unique_events_objects: pd.DataFrame,
    df_objects: pd.DataFrame,
    object_points_path: str,
    max_reasonable_m: float = 15.0,   # אפשר לשנות ל-10 או 20 לפי מה שמתאים לך
    label_col: str = "Pedestrian",
) -> pd.DataFrame:
    """
    מחליף את add_pedestrians_start_walking_simple + add_pedestrian_crossing_min_distance
    בלוגיקה אחידה אחת: 3 שלבים לכל הולך רגל, כולם נמצאים באותה שיטה --
    זמן הסימולציה שבו מיקום הולך הרגל (במסלול הרציף שלו ב-df_objects) הכי
    קרוב לנקודת ייחוס מ-objectPoints.csv:

    1) "start walking: <name>"  -- הכי קרוב ל-(SimulationPosition.x, SimulationPosition.y)  (מרחק מישורי)
    2) "start crossing: <name>" -- הכי קרוב ל-(ped_start_lon, ped_start_lat)  (Haversine), מחפשים משלב 1 והלאה
    3) "end crossing: <name>"   -- הכי קרוב ל-(ped_end_lon, ped_end_lat)  (Haversine), מחפשים משלב 2 והלאה,
       עם סף איכות max_reasonable_m כמו קודם.

    בנוסף: אם ברגע היצירה (שלב 1) הקטגוריה (SpacialEvent/event_category) עדיין
    ריקה -- כלומר הולך הרגל נוצר לפני שהאירוע המופעל על ידי הרכב התחיל --
    ממלאים את הקטגוריה אחורה מרגע היצירה, אבל רק בשורות שהיו ריקות (לא דורסים
    קטגוריה קיימת).
    """

    df_points = load_object_points(object_points_path)

    if "SimulationTime" not in df_with_distances.columns:
        raise ValueError("df_with_distances must contain 'SimulationTime'")
    if not {"SpacialEvent", "relevant_object_name"}.issubset(df_unique_events_objects.columns):
        raise ValueError("df_unique_events_objects must contain 'SpacialEvent' and 'relevant_object_name'")
    required_point_cols = {
        "SpacialEvent", "SimulationPosition.x", "SimulationPosition.y",
        "ped_start_lat", "ped_start_lon", "ped_end_lat", "ped_end_lon",
    }
    if not required_point_cols.issubset(df_points.columns):
        raise ValueError(f"objectPoints missing required columns: {sorted(required_point_cols)}")
    if not {"Name", "SimulationTime"}.issubset(df_objects.columns):
        raise ValueError("df_objects must contain 'Name' and 'SimulationTime'")

    df_unique = df_unique_events_objects.copy()
    df_unique["SpacialEvent"] = df_unique["SpacialEvent"].astype(str).str.strip()
    df_unique["relevant_object_name"] = df_unique["relevant_object_name"].astype(str).str.strip()

    df_points = df_points.copy()
    df_points["SpacialEvent"] = df_points["SpacialEvent"].astype(str).str.strip()

    df_out = df_with_distances.copy()
    df_out["SimulationTime"] = pd.to_numeric(df_out["SimulationTime"], errors="coerce")
    main_times = df_out["SimulationTime"]

    if label_col not in df_out.columns:
        df_out[label_col] = np.nan

    if main_times.isna().all():
        print("❌ df_with_distances['SimulationTime'] is all NaN — cannot place markers")
        return df_out

    cols_lower = {c.lower(): c for c in df_objects.columns}
    lat_col = cols_lower.get("latitude") or next((cols_lower[c] for c in cols_lower if "lat" in c), None)
    lon_col = cols_lower.get("longitude") or next((cols_lower[c] for c in cols_lower if "lon" in c), None)
    if lat_col is None or lon_col is None:
        raise ValueError("df_objects must contain Latitude/Longitude columns (or columns containing 'lat'/'lon')")

    df_obj = df_objects.copy()
    df_obj["SimulationTime"] = pd.to_numeric(df_obj["SimulationTime"], errors="coerce")
    df_obj[lat_col] = pd.to_numeric(df_obj[lat_col], errors="coerce")
    df_obj[lon_col] = pd.to_numeric(df_obj[lon_col], errors="coerce")
    if "SimulationPosition.x" in df_obj.columns:
        df_obj["SimulationPosition.x"] = pd.to_numeric(df_obj["SimulationPosition.x"], errors="coerce")
    if "SimulationPosition.y" in df_obj.columns:
        df_obj["SimulationPosition.y"] = pd.to_numeric(df_obj["SimulationPosition.y"], errors="coerce")

    used_rows_for_current_walker = set()

    def mark_time(sim_time: float, text: str):
        """
        מסמן טקסט בשורה שזמן הסימולציה שלה הכי קרוב ל-sim_time. מדלג על שורות
        שכבר שובצו לשלב קודם של אותו הולך רגל (used_rows_for_current_walker) --
        גם אחרי שכפינו זמנים שונים בפריימים הגולמיים של הולך הרגל (stage1_idx+1
        וכו'), שני זמנים קרובים מדי עדיין יכולים "להתעגל" לאותה שורה בדיוק
        בדאטה הראשי (שדוגם בקצב/היסט אחר) -- זה מבטיח ששלבים שונים לעולם לא
        יתמזגו לאותו תא, גם במקרה כזה.
        """
        diffs = (main_times - sim_time).abs()
        if used_rows_for_current_walker:
            diffs = diffs.copy()
            diffs.loc[list(used_rows_for_current_walker)] = np.inf
        idx = diffs.idxmin()
        used_rows_for_current_walker.add(idx)
        current = df_out.at[idx, label_col]
        if pd.isna(current) or str(current).strip() == "":
            df_out.at[idx, label_col] = text
        else:
            cur = str(current)
            if text not in cur:
                df_out.at[idx, label_col] = f"{cur} | {text}"

    def find_time_of_min_distance(traj, target_x, target_y, x_col, y_col, use_haversine):
        """
        מחזיר (matched_time, positional_index_in_traj, min_distance) עבור השורה
        ב-traj שהמיקום שלה (x_col, y_col) הכי קרוב ל-(target_x, target_y).
        use_haversine=True מתייחס ל-(x_col, y_col) כ-(lon, lat) במעלות ומחשב
        מרחק Haversine; אחרת מרחק מישורי (אוקלידי) ביחידות x_col/y_col.
        """
        xs = traj[x_col].astype(float).to_numpy()
        ys = traj[y_col].astype(float).to_numpy()
        n = len(traj)
        if n == 0:
            return None, None, None

        if use_haversine:
            d = np.array(
                meda.distanceHaversineVectors(ys, xs, [target_y] * n, [target_x] * n),
                dtype=float,
            )
        else:
            d = np.hypot(xs - target_x, ys - target_y)

        if np.isnan(d).all():
            return None, None, None

        i = int(np.nanargmin(d))
        return float(traj.iloc[i]["SimulationTime"]), i, float(d[i])

    ped_events = df_unique[
        df_unique["SpacialEvent"].str.contains("walker", case=False, na=False)
    ][["SpacialEvent", "relevant_object_name"]].drop_duplicates()

    if ped_events.empty:
        print("ℹ️ No pedestrian events found (SpacialEvent contains 'walker') in df_unique_events_objects")
        return df_out

    any_backfill_happened = False
    critical_lines = {}  # object_name -> (line_a_x, line_a_y, line_b_x, line_b_y), in meters

    for _, row in ped_events.iterrows():
        event_name = row["SpacialEvent"]
        object_name = row["relevant_object_name"]
        used_rows_for_current_walker.clear()

        pts = df_points[df_points["SpacialEvent"] == event_name]
        if pts.empty:
            print(f"⚠️ No reference points for event: {event_name}")
            continue
        pts = pts.iloc[0]

        spawn_x, spawn_y = pts["SimulationPosition.x"], pts["SimulationPosition.y"]
        s_lat, s_lon = pts["ped_start_lat"], pts["ped_start_lon"]
        e_lat, e_lon = pts["ped_end_lat"], pts["ped_end_lon"]

        if pd.isna(spawn_x) or pd.isna(spawn_y):
            print(f"⚠️ Missing SimulationPosition.x/y for event: {event_name} — skipped")
            continue
        if pd.isna(s_lat) or pd.isna(s_lon) or pd.isna(e_lat) or pd.isna(e_lon):
            print(f"⚠️ Missing ped_start/ped_end lat/lon for event: {event_name} — skipped")
            continue

        spawn_x, spawn_y = float(spawn_x), float(spawn_y)
        s_lat, s_lon, e_lat, e_lon = float(s_lat), float(s_lon), float(e_lat), float(e_lon)

        traj = df_obj[df_obj["Name"] == object_name].dropna(
            subset=["SimulationTime", lat_col, lon_col, "SimulationPosition.x", "SimulationPosition.y"]
        ).sort_values("SimulationTime").reset_index(drop=True)

        if traj.empty:
            print(f"⚠️ Object not found or no valid lat/lon/position: {object_name}")
            continue

        # שלב 1: נוצר / מתחיל ללכת -- הכי קרוב לנקודת ההיווצרות (מרחק מישורי)
        stage1_time, stage1_idx, stage1_dist = find_time_of_min_distance(
            traj, spawn_x, spawn_y, "SimulationPosition.x", "SimulationPosition.y", use_haversine=False
        )
        if stage1_time is None:
            print(f"⚠️ Could not compute stage-1 (created) time for {object_name} (event={event_name})")
            continue
        mark_time(stage1_time, "start walking")

        # מילוי קטגוריה אחורה מרגע ההיווצרות, אם היא ריקה שם (רק מילוי, לא דריסה).
        # העוגן התחתון הוא הזמן בפועל של השורה שבה הוצב הסימון (לא stage1_time הגולמי
        # מהמסלול של הולך הרגל) -- כי mark_time משבץ לפי השורה הכי קרובה, שיכולה
        # להיות מעט *לפני* stage1_time, ואז השוואת ">=" מול stage1_time הייתה מפספסת
        # בדיוק את השורה שבה מופיע הסימון עצמו.
        base_event_label = clean_event_name(event_name)
        stage1_marked_idx = (main_times - stage1_time).abs().idxmin()
        stage1_backfill_start_time = min(stage1_time, df_out.loc[stage1_marked_idx, "SimulationTime"])
        window_start_candidates = df_out.loc[
            df_out["SpacialEvent"].astype(str).str.strip() == event_name, "SimulationTime"
        ]
        if not window_start_candidates.empty:
            window_start_time = window_start_candidates.min()
            if stage1_backfill_start_time < window_start_time:
                is_empty_category = df_out["SpacialEvent"].isna() | df_out["SpacialEvent"].astype(str).str.strip().isin(["None", "nan", ""])
                gap_mask = (
                    (df_out["SimulationTime"] >= stage1_backfill_start_time)
                    & (df_out["SimulationTime"] < window_start_time)
                    & is_empty_category
                )
                df_out.loc[gap_mask, "SpacialEvent"] = base_event_label
                if "BaseEvent" in df_out.columns:
                    df_out.loc[gap_mask, "BaseEvent"] = base_event_label
                if "event_category" in df_out.columns:
                    df_out.loc[gap_mask, "event_category"] = "Pedestrians"
                if gap_mask.any():
                    any_backfill_happened = True

        # שלב 2: יורד לכביש / מתחיל לחצות -- הכי קרוב לנקודת תחילת החצייה (Haversine),
        # מחפשים מהפריים שאחרי שלב 1 (לא כולל אותו) -- כך שלב 1 ושלב 2 לעולם לא
        # יצביעו על אותה שורה בדיוק, גם אם נקודת ההיווצרות זהה לנקודת תחילת החצייה
        # (כמו שקורה בפועל להולכי רגל שנוצרים ממש על קצה הכביש).
        if stage1_idx + 1 >= len(traj):
            print(f"⚠️ No trajectory left after stage-1 for {object_name} (event={event_name})")
            continue
        traj_from_stage1 = traj.iloc[stage1_idx + 1:].reset_index(drop=True)
        stage2_time, stage2_idx, stage2_dist = find_time_of_min_distance(
            traj_from_stage1, s_lon, s_lat, lon_col, lat_col, use_haversine=True
        )
        if stage2_time is None:
            print(f"⚠️ Could not compute stage-2 (start crossing) time for {object_name} (event={event_name})")
            continue
        if stage2_dist > max_reasonable_m:
            print(
                f"⚠️ start-crossing min_dist too large ({stage2_dist:.2f}m > {max_reasonable_m}m) "
                f"| event={event_name} | object={object_name} — NOT marking"
            )
            continue
        mark_time(stage2_time, "start crossing")

        # שלב 3: עולה למדרכה / מסיים לחצות -- הכי קרוב לנקודת סיום החצייה (Haversine),
        # מחפשים מהפריים שאחרי שלב 2 (לא כולל אותו), מאותה סיבה.
        if stage2_idx + 1 >= len(traj_from_stage1):
            print(f"⚠️ No trajectory left after stage-2 for {object_name} (event={event_name})")
            continue
        traj_from_stage2 = traj_from_stage1.iloc[stage2_idx + 1:].reset_index(drop=True)
        stage3_time, stage3_idx, stage3_dist = find_time_of_min_distance(
            traj_from_stage2, e_lon, e_lat, lon_col, lat_col, use_haversine=True
        )
        if stage3_time is None:
            print(f"⚠️ Could not compute stage-3 (end crossing) time for {object_name} (event={event_name})")
            continue
        if stage3_dist > max_reasonable_m:
            print(
                f"⚠️ end-crossing min_dist too large ({stage3_dist:.2f}m > {max_reasonable_m}m) "
                f"| event={event_name} | object={object_name} — NOT marking end"
            )
            continue
        mark_time(stage3_time, "end crossing")

        # שומרים את קו החצייה (בין נקודת תחילת החצייה לנקודת סיום החצייה,
        # ביחידות SimulationPosition.x/y -- מטרים, אותה מערכת צירים כמו
        # PositionX/PositionY של הרכב) -- כדי לחשב אחר כך time_to_critical_point
        # לכל שורה של הולך רגל זה: מרחק ניצב לקו הזה, חלקי Speed.
        line_a_x = float(traj_from_stage1.iloc[stage2_idx]["SimulationPosition.x"])
        line_a_y = float(traj_from_stage1.iloc[stage2_idx]["SimulationPosition.y"])
        line_b_x = float(traj_from_stage2.iloc[stage3_idx]["SimulationPosition.x"])
        line_b_y = float(traj_from_stage2.iloc[stage3_idx]["SimulationPosition.y"])
        critical_lines[object_name] = (line_a_x, line_a_y, line_b_x, line_b_y)

    if any_backfill_happened:
        # אחרי שהקטגוריה/האירוע מולאו אחורה, שאר המשתנים הרלוונטיים לאירוע
        # (relevant_object_name, distance_from_relevant_object,
        # time_since_event_start_world) צריכים להתעדכן גם הם עבור אותן שורות.
        # במקום לשכפל את הלוגיקה, פשוט מריצים שוב את שתי הפונקציות הקיימות
        # שכבר מחשבות את הערכים האלה -- שתיהן מבוססות אך ורק על SpacialEvent/
        # BaseEvent הנוכחיים, שעכשיו כבר משתרעים אחורה עד רגע ההיווצרות, אז
        # הן יתפסו את השורות שמולאו בדיוק כמו כל שורה אחרת של אותו אירוע.
        # (time_to_collision יתעדכן בהמשך הפייפליין באופן טבעי, כי הוא כבר רץ
        # אחרי שלב הזה ויראה event_category=="Pedestrians" עם relevant_object_name
        # מלא.)
        df_out = calculate_all_event_distances(df_out, df_objects, df_unique_events_objects)
        df_out = add_time_since_event_start_world_special(df_out)

    # time_to_critical_point: מרחק ניצב מהרכב לקו החצייה (בין נקודת תחילת
    # החצייה לנקודת סיום החצייה) של הולך הרגל הרלוונטי לאותה שורה, חלקי
    # Speed הנוכחי -- בלי כיוון/וקטור, בדיוק כמו נוסחת ה-TTC הישנה
    # (מרחק/מהירות), רק מול הקו במקום מול נקודה/אובייקט נע.
    if critical_lines and "PositionX" in df_out.columns and "PositionY" in df_out.columns:
        df_out["time_to_critical_point"] = np.nan
        pos_x = pd.to_numeric(df_out["PositionX"], errors="coerce")
        pos_y = pd.to_numeric(df_out["PositionY"], errors="coerce")
        speed = pd.to_numeric(df_out["Speed"], errors="coerce")

        for obj_name, (ax, ay, bx, by) in critical_lines.items():
            obj_mask = df_out["relevant_object_name"].astype(str).str.strip() == obj_name
            if not obj_mask.any():
                continue
            # baseline_sign: הצד של הקו שבו הרכב נמצא כשהוא מתחיל להיות רלוונטי
            # לאירוע הזה. ברגע שהסימן מתהפך (הרכב עבר לצד השני של קו החצייה),
            # מפסיקים לחשב time_to_critical_point עבור שאר השורות של האובייקט
            # הזה -- אין טעם לדווח "זמן עד ההגעה לנקודה" אחרי שכבר הגיעו אליה.
            baseline_sign = None
            passed_critical_point = False
            for idx in df_out.index[obj_mask]:
                px, py, sp_kmh = pos_x.at[idx], pos_y.at[idx], speed.at[idx]
                # Speed מגיע בקמ"ש (ראו הערה ב-compute_ttc_static_object) --
                # ממירים למ'/ש כאן, בתוך הנוסחה, לפני שמשתמשים בו.
                sp = sp_kmh / 3.6 if pd.notna(sp_kmh) else sp_kmh
                # MIN_MEANINGFUL_SPEED: מתחת לזה, distance/speed מתפוצץ לערך
                # עצום וחסר משמעות (בדיוק כמו שקרה ב-TTC הישן לפני שתיקנו) --
                # לא מדווחים ערך במקום מספר אבסורדי.
                MIN_MEANINGFUL_SPEED = 0.5
                if pd.isna(px) or pd.isna(py) or pd.isna(sp) or sp <= MIN_MEANINGFUL_SPEED:
                    continue
                if passed_critical_point:
                    continue
                signed_dist = signed_perpendicular_distance_to_line(px, py, ax, ay, bx, by)
                current_sign = 1 if signed_dist > 0 else (-1 if signed_dist < 0 else 0)
                if baseline_sign is None:
                    if current_sign != 0:
                        baseline_sign = current_sign
                elif current_sign != 0 and current_sign != baseline_sign:
                    passed_critical_point = True
                    continue
                df_out.at[idx, "time_to_critical_point"] = abs(signed_dist) / sp

    return df_out


def process_pedestrian_labels(
    df_with_distances: pd.DataFrame,
    df_unique_events_objects: pd.DataFrame,
    df_objects: pd.DataFrame,
    object_points_path: str,
) -> pd.DataFrame:
    """
    פונקציית עטיפה לפייפליין שמריצה את כל הלוגיקה של הולכי הרגל, דרך
    add_pedestrian_stage_markers: מסמנת 3 שלבים אחידים בעמודה 'Pedestrian'
    ("start walking: <name>" / "start crossing: <name>" / "end crossing: <name>"),
    וממלאת אחורה את הקטגוריה מרגע ההיווצרות אם היא ריקה שם.
    """

    # ניקוי שמות כדי למנוע תקלות/אזהרות
    df_unique_events_objects = df_unique_events_objects.copy()
    df_unique_events_objects["relevant_object_name"] = (
        df_unique_events_objects["relevant_object_name"].astype(str).str.strip()
    )
    df_unique_events_objects["SpacialEvent"] = (
        df_unique_events_objects["SpacialEvent"].astype(str).str.strip()
    )

    # בסיס: אם אין עמודה - ניצור כדי ששתי הפונקציות ישרשרו לאותו מקום
    df_out = df_with_distances.copy()
    if "Pedestrian" not in df_out.columns:
        df_out["Pedestrian"] = np.nan

    # 3 שלבים אחידים: נוצר/מתחיל ללכת -> יורד לכביש -> עולה למדרכה
    df_out = add_pedestrian_stage_markers(
        df_with_distances=df_out,
        df_unique_events_objects=df_unique_events_objects,
        df_objects=df_objects,
        object_points_path=object_points_path,
        max_reasonable_m=15.0,   # אפשר לשנות ל-10 או 20
        label_col="Pedestrian",
    )

    return df_out


###################################################################################################################
## הוספת מצב הרמזור (Traffic Light State) לנתוני הסימולציה


import pandas as pd

def add_traffic_light_state(df_with_distances, df_objects):
    """
    מוסיף עמודה TrafficLight לאירועי TrafficLights לפי relevant_object_name ו-CurrentState/PreviousState.
    תיקון: אם ה-state הראשון באותו רמזור הוא Yellow -> משתמשים ב-PreviousState שלו כדי למלא אחורה את כל ה-NA לפניו.
    לא מפיל פייפליין אם חסר מידע.
    """

    # אם אין אירועים בכלל
    if "event_category" not in df_with_distances.columns:
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    df_traffic_events = df_with_distances[df_with_distances["event_category"] == "TrafficLights"].copy()

    # אין אירועי רמזור
    if df_traffic_events.empty:
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    # אין שיוך אובייקט רלוונטי לרמזור
    if (
        "relevant_object_name" not in df_traffic_events.columns
        or df_traffic_events["relevant_object_name"].dropna().empty
    ):
        print("⚠️ TrafficLights events exist אבל relevant_object_name ריק -> TrafficLight נשאר NA")
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    # חייבים לפחות CurrentState כדי להוציא צבע
    if "CurrentState" not in df_objects.columns:
        print("⚠️ CurrentState לא קיים ב-df_objects -> TrafficLight נשאר NA")
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    # חייבים Name כדי לקשר לרמזור
    if "Name" not in df_objects.columns:
        print("⚠️ Name לא קיים ב-df_objects -> TrafficLight נשאר NA")
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    # --- הסתמכות על שם בלבד (בלי Type) ---
    relevant_names = df_traffic_events["relevant_object_name"].dropna().astype(str).unique()
    df_traffic_lights = df_objects[df_objects["Name"].astype(str).isin(relevant_names)].copy()

    if df_traffic_lights.empty:
        print("⚠️ אין אובייקטים תואמים לרמזורים לפי Name ב-df_objects -> TrafficLight נשאר NA")
        df_with_distances["TrafficLight"] = pd.NA
        return df_with_distances

    # --- התאמת זמן (כמו אצלך) ---
    df_traffic_events["SimulationTime_rounded"] = df_traffic_events["SimulationTime"].round(1)
    df_traffic_lights["SimulationTime_rounded"] = df_traffic_lights["SimulationTime"].round(1)

    # נשתדל להביא גם PreviousState אם קיים (לצורך התיקון)
    cols_take = ["SimulationTime_rounded", "Name", "CurrentState"]
    if "PreviousState" in df_traffic_lights.columns:
        cols_take.append("PreviousState")

    df_traffic_events = df_traffic_events.merge(
        df_traffic_lights[cols_take],
        left_on=["SimulationTime_rounded", "relevant_object_name"],
        right_on=["SimulationTime_rounded", "Name"],
        how="left"
    )

    df_traffic_events = df_traffic_events.sort_values(["relevant_object_name", "SimulationTime"])

    # TrafficLight = CurrentState (לפני ffill/backfill)
    df_traffic_events["TrafficLight"] = df_traffic_events["CurrentState"]

    # --- מילוי קדימה בתוך אותו רמזור (כמו אצלך, אבל לפי group) ---
    df_traffic_events["TrafficLight"] = (
        df_traffic_events.groupby("relevant_object_name")["TrafficLight"].ffill()
    )

    # --- תיקון: אם ה-state הראשון באותו רמזור הוא Yellow -> מלא אחורה לפי PreviousState ---
    # זה עובד רק אם PreviousState קיים
    if "PreviousState" in df_traffic_events.columns:
        def fill_prefix_if_starts_with_yellow(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("SimulationTime").copy()

            # למצוא את השורה הראשונה שבה TrafficLight לא NA
            first_valid_idx = g["TrafficLight"].first_valid_index()
            if first_valid_idx is None:
                return g

            first_state = g.loc[first_valid_idx, "TrafficLight"]
            if pd.isna(first_state) or str(first_state).strip().lower() != "yellow":
                return g

            # לקחת PreviousState מאותה שורה (בדרך כלל Green)
            prev_state = g.loc[first_valid_idx, "PreviousState"]
            if pd.isna(prev_state):
                return g

            # למלא רק את מה שלפני first_valid_idx ורק איפה ש-NA
            pos = g.index.get_loc(first_valid_idx)
            prefix_idx = g.index[:pos]

            mask = g.loc[prefix_idx, "TrafficLight"].isna()
            g.loc[prefix_idx[mask], "TrafficLight"] = prev_state

            return g

        df_traffic_events = (
            df_traffic_events.groupby("relevant_object_name", group_keys=False)
            .apply(fill_prefix_if_starts_with_yellow)
        )

    # --- להחזיר חזרה ל-df_with_distances ---
    # מומלץ למזג על SimulationTime_rounded כדי למנוע בעיות float,
    # אבל אם אצלך כבר זה מיושר ואתה רוצה להשאיר כמו מקודם - אפשר.
    df_with_distances = df_with_distances.copy()
    df_with_distances["SimulationTime_rounded"] = df_with_distances["SimulationTime"].round(1)

    df_with_distances = df_with_distances.merge(
        df_traffic_events[["SimulationTime_rounded", "relevant_object_name", "TrafficLight"]],
        on=["SimulationTime_rounded", "relevant_object_name"],
        how="left"
    )

    # אם לא רוצים להשאיר עמודת עזר:
    df_with_distances.drop(columns=["SimulationTime_rounded"], inplace=True, errors="ignore")

    return df_with_distances






def add_traffic_light_junction_phase(
    df_main,
    spacial_file_path,
    participant_id=None,
    condition=None,
    round_to=1,   # נשאר כדי לא לשבור חתימה בפייפליין (לא בשימוש)
):
    df = df_main.copy()
    out_col = "TrafficLight_JunctionPhase"

    if out_col not in df.columns:
        df[out_col] = pd.NA

    if "SimulationTime" not in df.columns:
        return df

    # --- Resolve Id / Condition ---
    if "Id" in df.columns and df["Id"].dropna().any():
        df_id = str(df["Id"].dropna().iloc[0]).strip()
    else:
        df_id = str(participant_id).strip() if participant_id is not None else None

    if "Condition" in df.columns and df["Condition"].dropna().any():
        df_cond = str(df["Condition"].dropna().iloc[0]).strip()
    else:
        df_cond = str(condition).strip() if condition is not None else None

    if df_id is None or df_cond is None:
        return df

    # --- Load spacial ---
    try:
        sp = pd.read_csv(spacial_file_path)
    except Exception:
        return df

    needed = {"Id","Condition","Scenario","triggered_by","Event_Name","SimulationTime"}
    if any(c not in sp.columns for c in needed):
        return df

    # --- Normalize ---
    sp = sp.copy()
    sp["Id"] = sp["Id"].astype(str).str.strip()
    sp["Condition"] = sp["Condition"].astype(str).str.strip()
    sp["Scenario"] = sp["Scenario"].astype(str).str.strip()
    sp["triggered_by"] = sp["triggered_by"].astype(str).str.strip()
    sp["Event_Name"] = sp["Event_Name"].astype(str).str.strip()
    sp["SimulationTime"] = pd.to_numeric(sp["SimulationTime"], errors="coerce")

    sp = sp[
        (sp["Scenario"] == "Accompanied") &
        (sp["triggered_by"] == "Egocar") &
        (sp["Id"] == df_id) &
        (sp["Condition"] == df_cond)
    ].copy()

    if sp.empty:
        return df

    # --- Parse traffic light events ---
    def parse_event(s):
        s2 = s.lower()
        m = re.search(r"(start|end)?\s*traffic\s*light\s*(\d+)", s2)
        if not m:
            return None, None
        kind = m.group(1) if m.group(1) else "entry"
        tl   = int(m.group(2))
        return kind, tl

    parsed = sp["Event_Name"].apply(parse_event)
    sp["kind"] = parsed.apply(lambda x: x[0])
    sp["tl"]   = parsed.apply(lambda x: x[1])
    sp = sp.dropna(subset=["kind","tl","SimulationTime"])

    if sp.empty:
        return df

    # --- Build windows per TL (first chronological triple per TL) ---
    windows = []
    for tl, g in sp.sort_values("SimulationTime").groupby("tl"):
        g = g.sort_values("SimulationTime")

        t_start = g.loc[g["kind"] == "start", "SimulationTime"]
        t_entry = g.loc[g["kind"] == "entry", "SimulationTime"]
        t_end   = g.loc[g["kind"] == "end",   "SimulationTime"]

        # DEBUG: show event counts per TL to diagnose missing events
        print(f"DEBUG TL {tl}: start={len(t_start)} entry={len(t_entry)} end={len(t_end)}")

        # Need at least one anchor before the junction (start OR entry) plus the end
        if (t_start.empty and t_entry.empty) or t_end.empty:
            continue

        ts = float(t_start.iloc[0]) if not t_start.empty else float(t_entry.iloc[0])
        te = float(t_entry.iloc[0]) if not t_entry.empty else ts
        tx = float(t_end.iloc[0])

        if ts <= te <= tx:
            windows.append((tl, ts, te, tx))

    if not windows:
        return df

    # --- Prepare df_main times ---
    df["SimulationTime"] = pd.to_numeric(df["SimulationTime"], errors="coerce")
    df = df.sort_values("SimulationTime").reset_index(drop=True)

    valid = df["SimulationTime"].dropna()
    if valid.empty:
        return df

    idx_valid = valid.index.to_numpy()
    t_valid = valid.to_numpy()

    def nearest_index(target):
        j = int(np.argmin(np.abs(t_valid - target)))
        return idx_valid[j]

    df[out_col] = pd.NA

    # --- Apply each window ---
    windows = sorted(windows, key=lambda x: x[1])  # chronological

    for tl, ts, te, tx in windows:
        i_s = nearest_index(ts)
        i_e = nearest_index(te)
        i_x = nearest_index(tx)

        if not (i_s <= i_e <= i_x):
            continue

        # Approaching
        df.loc[i_s:i_e-1, out_col] = "Approaching"

        # InsideJunction
        df.loc[i_e+1:i_x-1, out_col] = "InsideJunction"

        # AtStopLine
        df.loc[i_e, out_col] = "AtStopLine"

        # LeavingJunction: fill from i_x to the last row of this TL event
        tl_mask_after = df.loc[i_x:, "SpacialEvent"].astype(str).str.lower().str.contains(
            f"traffic light {int(tl)}", na=False
        )
        i_end = int(tl_mask_after[tl_mask_after].index[-1]) if tl_mask_after.any() else i_x
        df.loc[i_x:i_end, out_col] = "LeavingJunction"

    return df


################################################################################################################################

### 🧩 חיבור פער בין רכבים עוקבים (`gap_acceptance`) לדאטה פריים הראשי

def calculate_gap_acceptance_table(df_objects, df_with_distances_light):
    """
    מחשב פערי מרחק (Gap Acceptance) בין רכבים עוקבים, לפי האירוע הרלוונטי בהתאם למפה.

    מחזיר:
    - DataFrame עם עמודות:
        SimulationTime, object1, object2, 
        CumulativeDistance_object1, CumulativeDistance_object2, GapAcceptance
    """

    # 🗺️ זיהוי המפה
    map_name = df_with_distances_light["Map"].dropna().unique()
    if len(map_name) != 1:
        print("❌ לא ניתן לקבוע מפה בודדת.")
        return None
    map_name = map_name[0]

        # 🎯 סינון לפי אירוע רלוונטי# 🎯 סינון לפי אירוע רלוונטי לפי המפה
    if map_name in ["A", "B"]:
        # מתאים גם ל- start וגם ל- end gap acceptance
        event_pattern = r"(?i)Egocar (?:start |end )?gap acceptance"
    elif map_name == "C":
        event_pattern = r"(?i)\bgap\s*acceptance\b"

    else:
        print(f"❌ מפה {map_name} לא נתמכת.")
        return None


    # סינון שורות האירוע
    event_mask = df_with_distances_light["SpacialEvent"].str.contains(event_pattern, na=False)
    df_event = df_with_distances_light[event_mask].copy()

    if df_event.empty:
        print(f"⚠️ לא נמצאו שורות תואמות לאירוע במפה {map_name}. מחזיר DataFrame ריק.")
        return pd.DataFrame(columns=[
            "SimulationTime", "object1", "object2",
            "CumulativeDistance_object1", "CumulativeDistance_object2", "GapAcceptance"
        ])


    # הפקת שמות האובייקטים הרלוונטיים מתוך העמודה שלנו
    relevant_objects_raw = df_event["relevant_object_name"].dropna().unique()
    relevant_objects = set()
    for name_list in relevant_objects_raw:
        relevant_objects.update(name_list.split(", "))
    relevant_objects = list(relevant_objects)

    # קביעת סדר לפי זמן הופעה ראשון
    first_times = df_objects[df_objects["Name"].isin(relevant_objects)].groupby("Name")["WorldTime"].min().sort_values()
    ordered_vehicles = list(first_times.index)

    # חישוב מרחק מצטבר לכל רכב
    distance_data = []

    for vehicle in ordered_vehicles:
        df_vehicle = df_objects[df_objects["Name"] == vehicle].copy()
        df_vehicle = df_vehicle.sort_values("WorldTime")
        df_vehicle["WorldTime"] = pd.to_datetime(df_vehicle["WorldTime"])
        df_vehicle["Speed"] = pd.to_numeric(df_vehicle["Speed"], errors='coerce')
        df_vehicle["TimeDiff"] = df_vehicle["WorldTime"].diff().dt.total_seconds().fillna(0)
        df_vehicle["CumulativeDistance"] = (df_vehicle["Speed"] * df_vehicle["TimeDiff"]).cumsum()
        df_vehicle["Name"] = vehicle
        distance_data.append(df_vehicle[["SimulationTime", "CumulativeDistance", "Name"]])

    # איחוד כל הרכבים
    df_distances_all = pd.concat(distance_data)

    # יצירת טבלת פערים
    simulation_times = sorted(df_event["SimulationTime"].unique())
    rows = []

    for t in simulation_times:
        for i in range(len(ordered_vehicles) - 1):
            obj1 = ordered_vehicles[i]
            obj2 = ordered_vehicles[i + 1]

            dist1 = df_distances_all[(df_distances_all["Name"] == obj1) &
                                     (df_distances_all["SimulationTime"] <= t)]["CumulativeDistance"].max()

            dist2 = df_distances_all[(df_distances_all["Name"] == obj2) &
                                     (df_distances_all["SimulationTime"] <= t)]["CumulativeDistance"].max()

            gap = np.nan
            if pd.notna(dist1) and pd.notna(dist2):
                gap = abs(dist2 - dist1)

            rows.append({
                "SimulationTime": t,
                "object1": obj1,
                "object2": obj2,
                "CumulativeDistance_object1": dist1,
                "CumulativeDistance_object2": dist2,
                "GapAcceptance": gap
            })

    df_gap_table = pd.DataFrame(rows)
    return df_gap_table

def merge_gap_acceptance_to_main(df_with_distances_light, df_gap_acceptance):
    """
    ממזג את נתוני gap_acceptance (פער בין רכבים עוקבים) אל תוך הדאטהפריים הראשי df_with_distances_light.

    פרמטרים:
    - df_with_distances_light: DataFrame עם נתוני האירועים והמרחקים.
    - df_gap_acceptance: DataFrame עם חישוב פער המרחקים בין רכבים עוקבים.

    מחזיר:
    - DataFrame עם עמודה חדשה gap_acceptance שנוספה לפי התאמה.
    """

    # שינוי שם עמודות כדי להתאים למבנה df_with_distances_light
    df_temp = df_gap_acceptance.rename(columns={
        "object1": "relevant_object_name",
        "GapAcceptance": "gap_acceptance"  # תיקון שם העמודה
    })

    # מיזוג לפי SimulationTime ו-relevant_object_name
    df_with_distances_gap = df_with_distances_light.merge(
        df_temp[["SimulationTime", "relevant_object_name", "gap_acceptance"]],
        on=["SimulationTime", "relevant_object_name"],
        how="left"
    )

    return df_with_distances_gap

####################################################################################################################

## 📘 תיאור פונקציות לחישוב זמן להתנגשות (TTC) ואיתור עקיפה

def create_distance_to_police_df(df_with_distances_gap, df_objects):
    # זיהוי סוג מפה
    map_type = str(df_with_distances_gap["Map"].iloc[0])

    df_police = df_objects[
        df_objects["Name"].str.contains(r"vehicle\.dodge\.charger_police", regex=True, na=False)
    ]

    # if df_police.empty:
        # במפה C אין משטרה – זה תקין
    #     if map_type != "C":
    #         print("❌ לא נמצא רכב משטרה.")
    #     return None
    
    
    if df_police.empty:
    # במפה C אין משטרה – זה תקין
        if map_type != "C":
            print("❌ לא נמצא רכב משטרה.")
            
        return pd.DataFrame(
            columns=["SimulationTime", "distance_to_police", "police_object_name"]
        )


    police_lat = float(df_police.iloc[0]["Latitude"])
    police_lon = float(df_police.iloc[0]["Longitude"])
    police_name = df_police.iloc[0]["Name"]

    from Andromeda.tidy import distanceHaversineVectors

    # 🛠️ המרה ל-float ליתר ביטחון
    latitudes = pd.to_numeric(df_with_distances_gap["Latitude"], errors="coerce")
    longitudes = pd.to_numeric(df_with_distances_gap["Longitude"], errors="coerce")

    distances = distanceHaversineVectors(
        latitudes.values,
        longitudes.values,
        [police_lat] * len(df_with_distances_gap),
        [police_lon] * len(df_with_distances_gap)
    )

    df_result = pd.DataFrame({
        "SimulationTime": df_with_distances_gap["SimulationTime"],
        "distance_to_police": distances,
        "police_object_name": police_name
    })

    return df_result


def detect_overtake_start_simulation_time(df_with_distances_gap, window=15, threshold=-0.000001):
    df_tmp = df_with_distances_gap.copy()

    # ניקוי + lower
    df_tmp["SpacialEvent"] = df_tmp["SpacialEvent"].astype(str).str.lower()

    # להבטיח מספרים
    df_tmp["Latitude"] = pd.to_numeric(df_tmp["Latitude"], errors="coerce")

    # סינון רק gap acceptance
    df_gap = df_tmp[df_tmp["SpacialEvent"].str.contains("egocar gap acceptance", na=False)].copy()
    if df_gap.empty:
        return None

    df_gap["lat_diff"] = df_gap["Latitude"].diff()
    df_gap["lat_diff_rolling"] = df_gap["lat_diff"].rolling(window=window).sum()

    start_row = df_gap[df_gap["lat_diff_rolling"] <= threshold].head(1)
    if not start_row.empty:
        return start_row["SimulationTime"].values[0]

    return None


def detect_ego_stop_time(df_with_distances_gap, speed_threshold=0.3, min_duration_s=1.0):
    """
    מוצא מתי הרכב שלנו בפועל נעצר (ממתין מאחורי רכב המשטרה), בתוך אירוע
    ה-gap acceptance -- באותה שיטת חלון-נגלל כמו detect_overtake_start_simulation_time,
    אך מבוסס על Speed במקום Latitude. דורש כמה פריימים רצופים מתחת לסף (לא רק
    פריים בודד) כדי להימנע מרעש צף קרוב-לאפס שכבר מצאנו ב-Speed (למשל 2.07e-07).
    זהו רגע המעבר החדש -- מוקדם יותר מ-overtake_sim_time -- ל-time_to_collision:
    לפניו TTC מול רכב המשטרה, ממנו והלאה TTC מול הרכבים ממול.
    """
    df_tmp = df_with_distances_gap.copy()
    df_tmp["SpacialEvent"] = df_tmp["SpacialEvent"].astype(str).str.lower()
    # Speed מגיע בקמ"ש -- ראו הערה ב-compute_ttc_static_object. speed_threshold
    # (0.3) מוגדר במ'/ש, אז ממירים כאן לפני ההשוואה.
    df_tmp["Speed"] = pd.to_numeric(df_tmp["Speed"], errors="coerce") / 3.6
    df_tmp["SimulationTime"] = pd.to_numeric(df_tmp["SimulationTime"], errors="coerce")
    df_tmp = df_tmp.sort_values("SimulationTime").reset_index(drop=True)

    df_gap = df_tmp[df_tmp["SpacialEvent"].str.contains("egocar gap acceptance", na=False)].copy()
    if df_gap.empty:
        return None
    df_gap = df_gap.reset_index(drop=True)

    time_diffs = df_gap["SimulationTime"].diff().dropna()
    avg_dt = time_diffs.median() if not time_diffs.empty and time_diffs.median() > 0 else 0.1
    window = max(1, int(round(min_duration_s / avg_dt)))

    is_slow = df_gap["Speed"] <= speed_threshold
    sustained = is_slow.rolling(window=window).sum() >= window

    stop_positions = np.flatnonzero(sustained.to_numpy())
    if len(stop_positions) == 0:
        return None

    # הזמן המדווח הוא תחילת החלון הרצוף הראשון (לא סופו)
    first_end_pos = int(stop_positions[0])
    first_start_pos = max(0, first_end_pos - window + 1)
    return float(df_gap.iloc[first_start_pos]["SimulationTime"])


def compute_ttc_static_object(ego_x, ego_y, ego_speed, ego_yaw_deg, obj_x, obj_y):
    """
    TTC מול אובייקט קבוע (מיקום יחיד, כמו רכב משטרה חונה שנרשם פעם אחת בקובץ
    האובייקטים) -- פורט של compute_ttc_static מהסקריפט החיצוני: מהירות הסגירה
    היא הטלת וקטור המהירות של הרכב שלנו על כיוון הקו לאובייקט.
    מחזיר (ttc, distance); ttc=inf אם לא מתקרבים (closing_speed<=0).
    """
    rx = obj_x - ego_x
    ry = obj_y - ego_y
    distance = float(np.hypot(rx, ry))

    if distance < 1e-6:
        return 0.0, distance

    # ego_speed מגיע בקמ"ש (כפי שנרשם בקובץ הגולמי -- אומת אמפירית מול מהירות
    # מחושבת ממיקום/זמן, יחס ~3.6 עקבי), בעוד PositionX/Y במטרים -- ממירים
    # למ'/ש כאן, בתוך הנוסחה עצמה, כדי ש-closing_speed יהיה במ'/ש אמיתיים.
    ego_speed_mps = ego_speed / 3.6
    yaw = np.radians(ego_yaw_deg)
    ego_vx = ego_speed_mps * np.cos(yaw)
    ego_vy = ego_speed_mps * np.sin(yaw)

    closing_speed = (rx * ego_vx + ry * ego_vy) / distance
    # מתחת לסף הזה, distance/closing_speed מתפוצץ לערך עצום וחסר משמעות
    # (בדיוק כמו שקרה בענפים אחרים לפני שהוספנו הגנה דומה) -- לא רק
    # closing_speed<=0, אלא כל קצב סגירה זניח.
    MIN_MEANINGFUL_CLOSING_SPEED = 0.5
    if closing_speed <= MIN_MEANINGFUL_CLOSING_SPEED:
        return float("inf"), distance

    return distance / closing_speed, distance


def compute_ttc_dynamic_object(ego_x, ego_y, ego_speed, ego_yaw_deg, obj_x, obj_y, obj_vx, obj_vy, eps=1e-6):
    """
    TTC מול אובייקט נע (רכב ממול וכו'), לפי מודל קצב-סגירת-מרחק (range-rate) --
    אותו עיקרון כמו compute_ttc_static_object, מוכלל לאובייקט עם מהירות משלו:
    מהירות הסגירה = הטלת המהירות היחסית (אובייקט פחות הרכב שלנו) על קו הראייה
    ביניהם. TTC = מרחק / מהירות סגירה (אינסוף אם לא מתקרבים).

    למה לא פתרון ריבועי של נקודת-מגע (כמו compute_ttc_dynamic בסקריפט המקורי
    ששיתפת)? בדקנו בפועל מול רכבים ממול באירועי gap acceptance: יש הפרש רוחבי
    קבוע (נתיב נגדי, כ-3-4 מ') בין הרכב שלנו לרכב הממול, כך שהם לעולם לא
    מגיעים לאותה נקודה (x,y) ברדיוס קטן (min_dist) -- מודל ריבועי עם min_dist
    קטן מחזיר במקרה הזה "inf" (דיסקרימיננטה שלילית) גם כשבבירור יש התקרבות
    מהירה על ציר הנסיעה. מודל קצב-הסגירה עונה נכון על "כמה זמן עד שהמרחק
    ביננו יורד ל-0 בקירוב לינארי", בלי תלות ברוחב הנתיב -- מתאים לשאלת
    gap acceptance (מתי הרכבים יהיו זה לצד זה), לא לשאלת התנגשות-נקודתית.
    """
    rx = obj_x - ego_x
    ry = obj_y - ego_y
    distance = float(np.hypot(rx, ry))

    if distance < eps:
        return 0.0

    # ego_speed מגיע בקמ"ש -- ראו הערה זהה ב-compute_ttc_static_object.
    ego_speed_mps = ego_speed / 3.6
    yaw = np.radians(ego_yaw_deg)
    ego_vx = ego_speed_mps * np.cos(yaw)
    ego_vy = ego_speed_mps * np.sin(yaw)

    vrel_x = obj_vx - ego_vx
    vrel_y = obj_vy - ego_vy

    # מהירות הסגירה = מינוס הנגזרת של המרחק (d|r|/dt = (r . v_rel)/|r|)
    closing_speed = -(rx * vrel_x + ry * vrel_y) / distance
    # מתחת לסף הזה, distance/closing_speed מתפוצץ לערך עצום וחסר משמעות --
    # אומת בנתונים האמיתיים (עד 49,459 שניות עבור רכבים ממול כשקצב הסגירה
    # קרוב לאפס אך לא בדיוק אפס). לא רק closing_speed<=0, אלא כל קצב סגירה זניח.
    MIN_MEANINGFUL_CLOSING_SPEED = 0.5
    if closing_speed <= MIN_MEANINGFUL_CLOSING_SPEED:
        return float("inf")

    return distance / closing_speed


def perpendicular_distance_to_line(px, py, ax, ay, bx, by):
    """
    מרחק ניצב (perpendicular) מנקודה P לקו האינסופי שעובר דרך A ו-B.
    לא תלוי בכיוון/מהירות -- רק גיאומטריה של מיקומים. משמש עבור
    time_to_critical_point = מרחק זה חלקי Speed, באותו רעיון של TTC הישן
    (מרחק חלקי מהירות), רק מול קו החצייה (בין ped_start ל-ped_end) במקום
    מול נקודה/אובייקט נע.
    """
    dx = bx - ax
    dy = by - ay
    line_len = np.hypot(dx, dy)
    if line_len < 1e-9:
        return float(np.hypot(px - ax, py - ay))
    # |cross(B-A, P-A)| / |B-A|
    cross = dx * (py - ay) - dy * (px - ax)
    return float(abs(cross) / line_len)


def signed_perpendicular_distance_to_line(px, py, ax, ay, bx, by):
    """
    כמו perpendicular_distance_to_line, אבל עם סימן: הסימן מציין באיזה צד
    של הקו הנקודה נמצאת. משמש כדי לזהות מתי הרכב עבר מצד אחד של קו החצייה
    לצד השני (שינוי סימן), ואז להפסיק לחשב time_to_critical_point.
    """
    dx = bx - ax
    dy = by - ay
    line_len = np.hypot(dx, dy)
    if line_len < 1e-9:
        return float(np.hypot(px - ax, py - ay))
    cross = dx * (py - ay) - dy * (px - ax)
    return float(cross / line_len)


def compute_ttc_quadratic_object(ego_x, ego_y, ego_speed, ego_yaw_deg, obj_x, obj_y, obj_vx, obj_vy, min_dist=2.0, eps=1e-8):
    """
    TTC לפי פתרון ריבועי של מפגש נקודתי -- פורט מדויק של compute_ttc_dynamic
    מהסקריפט החיצוני ששיתפת: פותר |r + v_rel*t|^2 = min_dist^2 ומחזיר את הזמן
    העתידי המוקדם ביותר שבו המרחק בין הרכב שלנו לאובייקט יורד ל-min_dist,
    בהנחת מהירויות קבועות לשני הצדדים. מוגן מפני מהירות יחסית קרובה לאפס
    ומפני דיסקרימיננטה שלילית -- מחזיר inf במקום להתפוצץ למספר עצום.

    מתאים לאובייקטים שהמסלול שלהם חוצה בפועל את הנתיב שלנו (כמו הולך רגל
    שחוצה כביש) -- בודק אם המסלולים המלאים (שני הצירים) אכן ייפגשו בנקודה.
    בניגוד לזה, compute_ttc_dynamic_object (מודל קצב-סגירה) מתאים לאובייקטים
    בנתיב מקביל/נגדי עם הפרש רוחבי קבוע (כמו רכב ממול בנתיב הנגדי), שבו דרישת
    מפגש נקודתי לעולם לא תתקיים בגלל רוחב הנתיב, גם כשההתקרבות אמיתית.
    """
    rx = obj_x - ego_x
    ry = obj_y - ego_y

    # ego_speed מגיע בקמ"ש -- ראו הערה זהה ב-compute_ttc_static_object.
    ego_speed_mps = ego_speed / 3.6
    yaw = np.radians(ego_yaw_deg)
    ego_vx = ego_speed_mps * np.cos(yaw)
    ego_vy = ego_speed_mps * np.sin(yaw)

    vrel_x = obj_vx - ego_vx
    vrel_y = obj_vy - ego_vy

    a = vrel_x * vrel_x + vrel_y * vrel_y
    b = 2 * (rx * vrel_x + ry * vrel_y)
    c = rx * rx + ry * ry - min_dist * min_dist

    if c <= 0:
        return 0.0
    if abs(a) < eps:
        return float("inf")

    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t_candidates = [t for t in (t1, t2) if t > eps]
    return min(t_candidates) if t_candidates else float("inf")


# def calculate_time_to_collision_with_police(
#     df_with_distances_gap,
#     df_unique_events_objects,
#     df_objects,
#     df_distance_to_police,
#     overtake_sim_time
# ):
#     """
# #   " מחשבת זמן להתנגשות (TTC) בהתאם למפה, לאירוע, ולעיתוי העקיפה.#" מוסיפה שתי עמודות:
#     - time_to_collision
#     - ttc_object_name (האובייקט שנמדד מולו ה-TTC)
# #"    פרמטרים:"
#     - df_with_distances_gap: דאטה עם מרחק מהאובייקט הרלוונטי.
#     - df_unique_events_objects: טבלת ההתאמה בין אירועים לאובייקטים + תנועות יחסיות.
#     - df_objects: נתוני האובייקטים (מיקום ומהירות).
#     - df_distance_to_police: טבלה עם SimulationTime, distance_to_police, police_object_name.
#     - overtake_sim_time: זמן סימולציה שבו מתחילה העקיפה (עבור מפות A/B).
 #   "מחזיר:"
#     - DataFrame עם העמודות החדשות.
#     """

#     df_ttc = df_with_distances_gap.copy()
#     df_ttc["time_to_collision"] = np.nan
#     df_ttc["ttc_object_name"] = None

#     df_ttc["Speed"] = pd.to_numeric(df_ttc["Speed"], errors="coerce").fillna(0.0)
#     df_objects["Speed"] = pd.to_numeric(df_objects["Speed"], errors="coerce").fillna(0.0)

#     map_type = df_ttc["Map"].iloc[0]
#     is_map_A_or_B = map_type in ["A", "B"]

#     for idx, row in df_ttc.iterrows():
#         t = row["SimulationTime"]
#         speed = row["Speed"]

#         if pd.isna(speed) or speed <= 0:
#             continue

#         spacial_event = str(row.get("SpacialEvent", "")).lower()
#         is_gap_acceptance_event = "egocar gap acceptance" in spacial_event

def calculate_time_to_collision_with_police(
    df_with_distances_gap,
    df_unique_events_objects,
    df_objects,
    df_distance_to_police,
    overtake_sim_time
):
    """
    שכתוב עם פיזיקה נכונה של תנועה יחסית (בהשראת סקריפט ה-CARLA TTC שהובא):
    - מול רכב המשטרה (מיקום קבוע, רשומה יחידה בקובץ האובייקטים): מהירות סגירה
      = הטלת וקטור המהירות של הרכב שלנו על כיוון הקו למשטרה (compute_ttc_static_object).
    - מול הרכבים ממול (אחרי שהרכב שלנו נעצר וממתין לעקוף): מודל קצב-סגירה
      (compute_ttc_dynamic_object) עם וקטורי מהירות אמיתיים לשני הצדדים, על כל
      מועמד מתוך df_unique_events_objects (לא scan עצמאי), עם מינימום TTC בין
      המועמדים -- רכב שכבר עבר את הרכב שלנו מקבל TTC=inf באופן טבעי (המרחק רק
      גדל) ופשוט נופל מהתחרות על המינימום. לא המודל הריבועי המקורי (מפגש
      נקודתי) כי יש הפרש רוחבי קבוע (רוחב נתיב) שגורם לו תמיד להחזיר inf גם
      כשההתקרבות אמיתית.
    - מול הולך רגל (event_category=="Pedestrians"): המודל הריבועי המקורי של
      מפגש נקודתי (compute_ttc_quadratic_object) -- כאן כן מתאים, כי המסלול
      של הולך הרגל אמור לחצות בפועל את הנתיב שלנו.
    - נקודת המעבר בין שלב המשטרה לשלב הרכבים ממול: detect_ego_stop_time (מתי
      הרכב שלנו בפועל נעצר) אם זוהתה עצירה; אם לא (הרכב רק מאט, למשל בתנאי
      Remote) -- overtake_sim_time (רגע תחילת הפנייה בפועל), כדי שעדיין יהיה
      שלב "מול המשטרה" עד לרגע שבו העקיפה בפועל מתחילה.
    - במפה C / באירועים שאינם gap acceptance ואינם הולך רגל: נשארת הלוגיקה
      הקיימת (מחוץ לתחום התיקון הנוכחי), עם הגנה נקודתית מפני חלוקה במהירות
      קרובה לאפס.
    """
    # מרחק מקסימלי סביר לחישוב TTC בכלל -- מועמד "רלוונטי" יכול בפועל להיות
    # רחוק מאוד ברגע נתון (אומת בנתונים אמיתיים: 659 מ'), מה שמייצר TTC ענק
    # וחסר משמעות גם כשקצב הסגירה חיובי. אותו MAX_DISTANCE_TO_CALC_TTC כמו
    # בסקריפט המקורי ששיתפת.
    MAX_DISTANCE_TO_CALC_TTC = 200.0

    df_ttc = df_with_distances_gap.copy()
    df_ttc["time_to_collision"] = np.nan
    df_ttc["ttc_object_name"] = None

    df_ttc["Speed"] = pd.to_numeric(df_ttc["Speed"], errors="coerce").fillna(0.0)
    df_ttc["Yaw"] = pd.to_numeric(df_ttc["Yaw"], errors="coerce") if "Yaw" in df_ttc.columns else np.nan
    df_ttc["PositionX"] = pd.to_numeric(df_ttc["PositionX"], errors="coerce")
    df_ttc["PositionY"] = pd.to_numeric(df_ttc["PositionY"], errors="coerce")
    df_objects["Speed"] = pd.to_numeric(df_objects["Speed"], errors="coerce").fillna(0.0)

    map_type = df_ttc["Map"].iloc[0]
    is_map_A_or_B = map_type in ["A", "B"]

    # רכב המשטרה: מיקום קבוע (רשומה יחידה בקובץ האובייקטים -- לא ניתן לחשב
    # ממנו מהירות/מסלול, ולכן משתמשים בו כנקודת ייחוס קבועה על פני כל הנסיעה)
    police_x = police_y = police_name = None
    if is_map_A_or_B:
        df_police = df_objects[
            df_objects["Name"].str.contains(r"vehicle\.dodge\.charger_police", regex=True, na=False)
        ]
        if not df_police.empty:
            police_x = float(df_police.iloc[0]["SimulationPosition.x"])
            police_y = float(df_police.iloc[0]["SimulationPosition.y"])
            police_name = df_police.iloc[0]["Name"]

    ego_stop_time = detect_ego_stop_time(df_with_distances_gap) if is_map_A_or_B else None
    # נקודת המעבר בין שלב המשטרה לשלב הרכבים ממול: אם זוהתה עצירה מלאה --
    # ברגע העצירה. אם לא (הרכב רק מאט ולא עוצר, למשל בתנאי Remote) -- ברגע
    # תחילת העקיפה (overtake_sim_time), שכן עד אז עדיין רלוונטי לעקוב אחרי
    # המשטרה, וממנו והלאה הסיכון האמיתי הוא כבר התנועה ממול.
    phase_boundary_time = ego_stop_time if ego_stop_time is not None else overtake_sim_time

    for idx, row in df_ttc.iterrows():
        t = row["SimulationTime"]
        speed = row["Speed"]
        yaw = row["Yaw"]
        ego_x, ego_y = row["PositionX"], row["PositionY"]

        if pd.isna(speed) or speed <= 0:
            continue
        if pd.isna(yaw) or pd.isna(ego_x) or pd.isna(ego_y):
            continue

        # ✅ עצור חישוב TTC אחרי שהרכב עבר את הרמזור
        junction_phase = str(row.get("TrafficLight_JunctionPhase", "")).strip()
        if junction_phase in ["InsideJunction", "LeavingJunction", ""]:
            continue

        spacial_event = str(row.get("SpacialEvent", "")).lower()
        is_gap_acceptance_event = "egocar gap acceptance" in spacial_event
        in_police_phase = (
            is_map_A_or_B and is_gap_acceptance_event
            and phase_boundary_time is not None and t < phase_boundary_time
            and police_x is not None
        )
        # ברירת המחדל לכל שורת gap acceptance היא שלב הרכבים ממול -- גם אם
        # לא זוהתה עצירה מלאה בפועל וגם לא זוהה overtake_sim_time (שני ה"None"
        # ביחד). כך שורות gap acceptance אף פעם לא נופלות לענף הישן/השגוי --
        # רק שורות שאינן gap acceptance כלל (מפה C, רמזורים, הולכי רגל וכו')
        # ממשיכות לענפים הייעודיים להן.
        in_oncoming_phase = (
            is_map_A_or_B and is_gap_acceptance_event and not in_police_phase
        )
        is_pedestrian_event = str(row.get("event_category", "")).strip() == "Pedestrians"

        # שלב 1: לפני שהרכב שלנו נעצר -- TTC מול רכב המשטרה
        if in_police_phase:
            ttc, distance = compute_ttc_static_object(ego_x, ego_y, speed, yaw, police_x, police_y)
            if 0 < distance <= MAX_DISTANCE_TO_CALC_TTC and not np.isinf(ttc):
                df_ttc.at[idx, "time_to_collision"] = ttc
                df_ttc.at[idx, "ttc_object_name"] = police_name

        # שלב 2: מרגע שנעצרנו (המתנה + העקיפה עצמה) -- TTC מול הרכבים ממול,
        # מינימום מבין כל המועמדים שכבר מותאמים לאירוע ב-df_unique_events_objects
        elif in_oncoming_phase:
            obj_name_raw = row.get("relevant_object_name")
            if pd.isna(obj_name_raw) or str(obj_name_raw).strip() == "":
                continue

            candidate_names = [n.strip() for n in str(obj_name_raw).split(",") if n.strip()]
            best_ttc = float("inf")
            best_name = None

            for cand in candidate_names:
                cand_traj = df_objects[df_objects["Name"] == cand].dropna(
                    subset=["SimulationTime", "SimulationPosition.x", "SimulationPosition.y"]
                ).sort_values("SimulationTime").reset_index(drop=True)
                if len(cand_traj) < 2:
                    continue

                nearest_pos = (cand_traj["SimulationTime"] - t).abs().idxmin()
                i0, i1 = (nearest_pos - 1, nearest_pos) if nearest_pos > 0 else (0, 1)
                r0, r1 = cand_traj.iloc[i0], cand_traj.iloc[i1]
                dt = r1["SimulationTime"] - r0["SimulationTime"]
                if dt <= 0:
                    continue

                obj_vx = (r1["SimulationPosition.x"] - r0["SimulationPosition.x"]) / dt
                obj_vy = (r1["SimulationPosition.y"] - r0["SimulationPosition.y"]) / dt
                obj_x = cand_traj.iloc[nearest_pos]["SimulationPosition.x"]
                obj_y = cand_traj.iloc[nearest_pos]["SimulationPosition.y"]

                # מועמד "רלוונטי" (df_unique_events_objects) יכול בפועל להיות
                # רחוק מאוד ברגע הנוכחי הספציפי, גם אם קצב הסגירה שלו חיובי --
                # מגבילים לטווח סביר (MAX_DISTANCE_TO_CALC_TTC, מוגדר למעלה).
                cand_distance = np.hypot(obj_x - ego_x, obj_y - ego_y)
                if cand_distance > MAX_DISTANCE_TO_CALC_TTC:
                    continue

                cand_ttc = compute_ttc_dynamic_object(ego_x, ego_y, speed, yaw, obj_x, obj_y, obj_vx, obj_vy)
                if cand_ttc < best_ttc:
                    best_ttc = cand_ttc
                    best_name = cand

            if best_name is not None and not np.isinf(best_ttc):
                df_ttc.at[idx, "time_to_collision"] = best_ttc
                df_ttc.at[idx, "ttc_object_name"] = best_name

        # הולך רגל (event_category=="Pedestrians") -- מודל ריבועי של מפגש
        # נקודתי (compute_ttc_quadratic_object), כי המסלול של הולך הרגל אמור
        # לחצות בפועל את הנתיב שלנו -- לא נתיב מקביל עם הפרש רוחבי קבוע כמו
        # רכב ממול. מהירות הולך הרגל מחושבת מהפרש מיקומים בין שתי דגימות
        # עוקבות בקובץ הגולמי, בדיוק כמו שנעשה לרכבים ממול.
        elif is_pedestrian_event:
            obj_name = row.get("relevant_object_name")
            if pd.isna(obj_name) or str(obj_name).strip() == "":
                continue

            ped_traj = df_objects[df_objects["Name"] == obj_name].dropna(
                subset=["SimulationTime", "SimulationPosition.x", "SimulationPosition.y"]
            ).sort_values("SimulationTime").reset_index(drop=True)
            if len(ped_traj) < 2:
                continue

            nearest_pos = (ped_traj["SimulationTime"] - t).abs().idxmin()
            i0, i1 = (nearest_pos - 1, nearest_pos) if nearest_pos > 0 else (0, 1)
            r0, r1 = ped_traj.iloc[i0], ped_traj.iloc[i1]
            dt = r1["SimulationTime"] - r0["SimulationTime"]
            if dt <= 0:
                continue

            ped_vx = (r1["SimulationPosition.x"] - r0["SimulationPosition.x"]) / dt
            ped_vy = (r1["SimulationPosition.y"] - r0["SimulationPosition.y"]) / dt
            ped_x = ped_traj.iloc[nearest_pos]["SimulationPosition.x"]
            ped_y = ped_traj.iloc[nearest_pos]["SimulationPosition.y"]

            if np.hypot(ped_x - ego_x, ped_y - ego_y) > MAX_DISTANCE_TO_CALC_TTC:
                continue

            ttc = compute_ttc_quadratic_object(ego_x, ego_y, speed, yaw, ped_x, ped_y, ped_vx, ped_vy, min_dist=2.0)
            if not np.isinf(ttc):
                df_ttc.at[idx, "time_to_collision"] = ttc
                df_ttc.at[idx, "ttc_object_name"] = obj_name

        # מפה C, אירוע שאינו gap acceptance ואינו הולך רגל, או שלא הצלחנו
        # לזהות שלב עצירה -- נשארת הלוגיקה הקיימת (מחוץ לתחום התיקון הנוכחי)
        else:
            obj_name = row["relevant_object_name"]
            distance = row["distance_from_relevant_object"]

            if pd.isna(distance) or distance <= 0:
                continue

            obj_type_row = df_unique_events_objects[df_unique_events_objects["relevant_object_name"] == obj_name]
            if obj_type_row.empty:
                relative_movement = "static"
            else:
                relative_movement = obj_type_row["relative_movement"].values[0] if "relative_movement" in obj_type_row else "static"

            obj_speed_row = df_objects[df_objects["Name"] == obj_name]
            # שני ה-Speed מגיעים בקמ"ש (גם ego וגם אובייקטים אחרים -- ראו הערה
            # ב-compute_ttc_static_object) -- ממירים כאן למ'/ש לפני שמשלבים
            # אותם עם distance (במטרים).
            object_speed = (obj_speed_row["Speed"].values[0] / 3.6) if not obj_speed_row.empty else 0.0
            speed = speed / 3.6

            if relative_movement == "static":
                relative_speed = speed
            elif relative_movement == "same_direction":
                relative_speed = max(speed - object_speed, 0)
            elif relative_movement == "opposite_direction":
                relative_speed = speed + object_speed
            elif relative_movement == "crossing":
                relative_speed = speed
            else:
                continue

            # תיקון נקודתי: מחלקים רק כשיש מהירות יחסית משמעותית. בלי הגנה הזו,
            # שורות עם Speed כמעט-אפס (למשל במהלך עצירה לפני הולך רגל, לא קשור
            # למשטרה) מייצרות ערכי TTC עצומים וחסרי משמעות (עד ~3.7×10¹² שניות
            # שנמצאו בדאטה המקורי) -- זה לא מתקן את הבעיה השורשית (relative_movement
            # תמיד "static" בענף הזה, מחוץ לתחום התיקון הנוכחי), רק מונע את
            # ההתפוצצות המספרית הספציפית.
            MIN_MEANINGFUL_RELATIVE_SPEED = 0.5
            if relative_speed > MIN_MEANINGFUL_RELATIVE_SPEED:
                ttc = distance / relative_speed
                df_ttc.at[idx, "time_to_collision"] = ttc
                df_ttc.at[idx, "ttc_object_name"] = obj_name

    return df_ttc

############################################################################################################################################

## הוספת נתוני תמלול (Transcript) לאירועים בסימולציה



import pandas as pd

def load_transcription_data(file_path, participant_id=None, condition=None, map_type=None):
    df = pd.read_csv(file_path)

    required_cols = {
        "triggered_by", "Type", "Event_Name",
        "Id", "Condition", "SimulationTime", "SimulationTimeEnd", "text",
        "speaker"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in transcription file: {sorted(list(missing))}")

    # סינון בסיסי
    df = df[df["triggered_by"].astype(str).str.strip() == "Egocar"].copy()
    df = df[df["Type"].astype(str).str.strip() == "Transcript"].copy()

    # ✅ סינון לנסיעה הנוכחית (אם נשלח)
    if participant_id is not None:
        df = df[df["Id"].astype(str).str.strip() == str(participant_id).strip()].copy()
    if condition is not None:
        df = df[df["Condition"].astype(str).str.strip() == str(condition).strip()].copy()
    if map_type is not None and "Map" in df.columns:
        df = df[df["Map"].astype(str).str.strip() == str(map_type).strip()].copy()

    if df.empty:
        # מחזיר DF ריק "תקני"
        return pd.DataFrame(columns=["Id", "Condition", "SimulationTime", "SimulationTimeEnd", "text", "speaker", "transcript_type"])

    # בחירת מקור בתוך הנסיעה בלבד
    df["Event_Name"] = df["Event_Name"].astype(str).str.strip()
    if (df["Event_Name"] == "TranscriptManual").any():
        df_best = df[df["Event_Name"] == "TranscriptManual"].copy()
        df_best["transcript_type"] = "manual"
    else:
        df_best = df[df["Event_Name"] == "TranscriptWhisper"].copy()
        df_best["transcript_type"] = "whisper"

    # תיקון End חסר
    df_best["SimulationTimeEnd"] = df_best["SimulationTimeEnd"].fillna(df_best["SimulationTime"])

    return df_best[["Id", "Condition", "SimulationTime", "SimulationTimeEnd", "text", "speaker", "transcript_type"]].copy()





import pandas as pd
import numpy as np

def add_transcription_to_events(df_with_ttc: pd.DataFrame,
                               df_transcripts: pd.DataFrame,
                               time_col: str = "SimulationTime",
                               round_to: int = 1,
                               concat_if_overlap: bool = True) -> pd.DataFrame:
    """
    מדביק ל-df_with_ttc את עמודות:
      - text
      - speaker
    מתוך df_transcripts.

    לוגיקה:
    1) אם יש מקטע דיבור [start, end] -> מדביק לכל השורות בתוך הטווח.
    2) אם לא נמצאה אף שורה בתוך הטווח -> מדביק לשורה עם הזמן הכי קרוב ל-start.

    הערות:
    - df_transcripts אמור להיות מסונן לנסיעה הנוכחית (Id+Condition(+Map)).
    - round_to משמש רק להגדרת טולרנס קטן (כדי להתמודד עם float), לא להשוואת שוויון.
    """

    df_updated = df_with_ttc.copy()

    # יצירת עמודות אם לא קיימות
    if "text" not in df_updated.columns:
        df_updated["text"] = pd.NA
    if "speaker" not in df_updated.columns:
        df_updated["speaker"] = pd.NA
    if "transcript_type" not in df_updated.columns:
        df_updated["transcript_type"] = pd.NA

    # ודא שיש עמודות חובה
    if time_col not in df_updated.columns:
        raise KeyError(f"df_with_ttc missing '{time_col}'")

    needed_t = {"SimulationTime", "SimulationTimeEnd", "text", "speaker", "transcript_type"}
    missing_t = needed_t - set(df_transcripts.columns)
    if missing_t:
        raise KeyError(f"df_transcripts missing columns: {sorted(list(missing_t))}")

    # ניקוי speaker (בגלל שיש לך 'מלווה ' עם רווח)
    df_t = df_transcripts.copy()
    df_t["speaker"] = df_t["speaker"].astype(str).str.strip()

    # אם end חסר -> נקודתי
    df_t["SimulationTimeEnd"] = df_t["SimulationTimeEnd"].fillna(df_t["SimulationTime"])

    # מסדרים אירועים לפי זמן ושומרים index מקורי כדי לכתוב חזרה נכון
    df_e = df_updated.reset_index().rename(columns={"index": "_orig_index"})
    df_e = df_e.sort_values(by=[time_col]).reset_index(drop=True)

    times = df_e[time_col].to_numpy(dtype=float)
    orig_idx = df_e["_orig_index"].to_numpy()

    # טולרנס קטן סביב הטווח (במקום round==)
    tol = 0.5 * (10 ** (-round_to))

    def assign_row(target_idx, txt, spk, trans_type):
        """מכניס טקסט/דובר/סוג תמלול לשורה אחת; אם יש כבר תוכן ונבחר concat -> משרשר."""
        cur_txt = df_updated.at[target_idx, "text"]
        cur_spk = df_updated.at[target_idx, "speaker"]
        cur_type = df_updated.at[target_idx, "transcript_type"]

        if pd.isna(cur_txt) or cur_txt is None or str(cur_txt).strip() == "":
            df_updated.at[target_idx, "text"] = txt
            df_updated.at[target_idx, "speaker"] = spk
            df_updated.at[target_idx, "transcript_type"] = trans_type
        else:
            if concat_if_overlap:
                df_updated.at[target_idx, "text"] = f"{cur_txt} | {txt}"
                # speaker: אם שונה, נשרשר כדי לא לאבד מידע
                if pd.isna(cur_spk) or cur_spk is None or str(cur_spk).strip() == "":
                    df_updated.at[target_idx, "speaker"] = spk
                elif str(cur_spk) != str(spk):
                    df_updated.at[target_idx, "speaker"] = f"{cur_spk} | {spk}"
                # transcript_type: אם שונה, נשרשר
                if pd.isna(cur_type) or cur_type is None or str(cur_type).strip() == "":
                    df_updated.at[target_idx, "transcript_type"] = trans_type
                elif str(cur_type) != str(trans_type):
                    df_updated.at[target_idx, "transcript_type"] = f"{cur_type} | {trans_type}"
            # אם לא רוצים שרשור – פשוט לא נוגעים (שומרים את הראשון)

    # עובר על כל מקטעי התמלול
    for _, r in df_t.iterrows():
        start = float(r["SimulationTime"])
        end = float(r["SimulationTimeEnd"]) if pd.notna(r["SimulationTimeEnd"]) else start
        if end < start:
            start, end = end, start

        txt = r["text"]
        spk = r["speaker"]
        trans_type = r["transcript_type"]

        left = start - tol
        right = end + tol

        # מחפש אינדקסים בתוך הטווח בעזרת searchsorted (מהיר)
        i0 = np.searchsorted(times, left, side="left")
        i1 = np.searchsorted(times, right, side="right")

        if i0 < i1:
            # מדביק לכל השורות בתוך הטווח
            for j in range(i0, i1):
                assign_row(orig_idx[j], txt, spk, trans_type)
        else:
            # אין אף sample בתוך הטווח -> בוחר את הזמן הכי קרוב ל-start
            pos = np.searchsorted(times, start, side="left")
            if pos == 0:
                nearest = 0
            elif pos >= len(times):
                nearest = len(times) - 1
            else:
                nearest = pos if abs(times[pos] - start) < abs(times[pos - 1] - start) else pos - 1

            assign_row(orig_idx[nearest], txt, spk, trans_type)

    return df_updated







def add_comment_flag(df_with_text):
    df = df_with_text.copy()
    df["comment_flag"] = (
        df["text"].astype(str).str.strip().ne("") &
        df["text"].notna()
    ).astype(int)
    return df




def add_start_comment_column(df_with_final_comments: pd.DataFrame,
                              df_transcripts: pd.DataFrame,
                              time_col: str = "SimulationTime") -> pd.DataFrame:
    """
#     שיטה B:
#     מסמן start_comment=1 בשורה ב-df_with_final_comments שהזמן שלה הכי קרו#     לכל תחילת מקטע תמלול (SimulationTime) מתוך df_transcripts.

#     הנחה:
    df_transcripts כבר מסונן לנסיעה הנוכחית (Id+Condition(+Map)), ולכן אין צורך ב-merge לפי Id/Condition כאן.
    """

    df_out = df_with_final_comments.copy()

    if time_col not in df_out.columns:
        raise KeyError(f"df_with_final_comments missing '{time_col}'")
    if "SimulationTime" not in df_transcripts.columns:
        raise KeyError("df_transcripts missing 'SimulationTime'")

    # אתחול עמודה
    df_out["start_comment"] = 0

    # אם אין תמלולים - מחזירים כמו שהוא עם start_comment=0
    if df_transcripts is None or df_transcripts.empty:
        return df_out

    # שומרים אינדקס מקורי וממיינים לפי זמן כדי לעבוד מהר
    df_sorted = df_out.reset_index().rename(columns={"index": "_orig_index"})
    df_sorted = df_sorted.sort_values(by=[time_col]).reset_index(drop=True)

    times = df_sorted[time_col].to_numpy(dtype=float)
    orig_idx = df_sorted["_orig_index"].to_numpy()

    # זמנים של תחילת התמלולים (מומלץ להוריד כפילויות כדי לא לעבוד סתם)
    starts = (
        df_transcripts["SimulationTime"]
        .dropna()
        .astype(float)
        .drop_duplicates()
        .to_numpy()
    )

    # לכל start -> מצמידים את השורה הקרובה ביותר
    for s in starts:
        pos = np.searchsorted(times, s, side="left")

        if pos == 0:
            nearest = 0
        elif pos >= len(times):
            nearest = len(times) - 1
        else:
            # בוחרים בין pos לבין pos-1
            nearest = pos if abs(times[pos] - s) < abs(times[pos - 1] - s) else pos - 1

        df_out.at[orig_idx[nearest], "start_comment"] = 1

    return df_out



import re

def add_first_feedback_in_event(df):
    df = df.copy()

    # Step 1 – normalize SpacialEvent to core name
    def clean_event(s):
        s = str(s).strip()
        s = re.sub(r"(?i)^(egocar\s+)?(start|end)\s+", "", s)
        return s.strip().lower()

    df["SpacialEvent_core"] = df["SpacialEvent"].apply(clean_event)

    # Step 2 – initialize column as NA
    df["first_feedback_in_event"] = pd.NA

    # Step 3 – within each event group, mark first start_comment=1
    group_cols = ["Id", "Condition", "SpacialEvent_core"]

    def _is_accompanier_or_na(spk):
        if pd.isna(spk):
            return True
        s = str(spk).strip().lower()
        if s in ("", "nan"):
            return True
        # כולל גם שורות משולבות (כמו "מלווה | נהג", כשתמלול של שני הדוברים
        # נחת על אותה שורה) -- לא רק "מלווה" בלבד, אלא כל שורה שהמלווה מדבר
        # בה, גם אם הנהג מדבר בה גם כן.
        return "מלווה" in s or "accompanier" in s

    for _, group in df.groupby(group_cols):
        accompanier_mask = group["speaker"].apply(_is_accompanier_or_na)
        feedback_rows = group[(group["start_comment"] == 1) & accompanier_mask]
        if feedback_rows.empty:
            continue  # leave as NA

        first_idx = feedback_rows.sort_values("SimulationTime").index[0]

        # set all rows in group to 0, then first feedback to 1
        df.loc[group.index, "first_feedback_in_event"] = 0
        df.loc[first_idx, "first_feedback_in_event"] = 1

    # SpacialEvent_core was only ever an internal grouping key. Leaving it in the
    # returned DataFrame made this branch produce one more column than the
    # empty-transcript branch of process_transcription_pipeline (which never adds
    # it), causing append_output() to silently column-shift every other trip.
    df = df.drop(columns=["SpacialEvent_core"])

    return df
# def process_transcription_pipeline(
#     df_with_ttc,
#     transcription_file_path,
#     participant_id,
#     condition,
#     map_type,
#     verbose=False
# ):
#     """
#     מריץ את כל שלבי התמלול בבת אחת עבור נסיעה נוכחית:
#     1) load_transcription_data (מסונן לנסיעה)
#     2) add_transcription_to_events (מדביק text+speaker)
#     3) add_comment_flag
#     4) add_start_comment_column (שיטה B: nearest start)

#     מחזיר:
#     - df_with_final_comments: הדאטה הראשי אחרי הדבקה+דגלים
#     - df_transcripts: טבלת התמלולים של הנסיעה (לדיבוג/שימוש עתידי)
#     """

#     df_transcripts = load_transcription_data(
#         transcription_file_path,
#         participant_id=participant_id,
#         condition=condition,
#         map_type=map_type
#     )
#     if df_transcripts is None or df_transcripts.empty:
        # נחזיר DF עם עמודות קיימות, אבל בלי טקסט/דגלים
#         df_out = df_with_ttc.copy()
#         if "text" not in df_out.columns:
#             df_out["text"] = pd.NA
#         if "speaker" not in df_out.columns:
#             df_out["speaker"] = pd.NA
#         df_out = add_comment_flag(df_out)
#         df_out["start_comment"] = 0
#         if verbose:
#             print("process_transcription_pipeline: df_transcripts empty -> returned without transcripts.")
#         return df_out, df_transcripts

#     if verbose:
#         print("After load_transcription_data:", df_transcripts.shape)

#     df_with_text = add_transcription_to_events(df_with_ttc, df_transcripts)
#     if verbose:
#         print("After add_transcription_to_events:", df_with_text.shape)

#     df_with_final_comments = add_comment_flag(df_with_text)
#     if verbose:
#         print("After add_comment_flag:", df_with_final_comments.shape)

#     df_with_final_comments = add_start_comment_column(df_with_final_comments, df_transcripts)
#     if verbose:
#         print("After add_start_comment_column:", df_with_final_comments.shape)

#     return df_with_final_comments, df_transcripts

def process_transcription_pipeline(
    df_with_ttc,
    transcription_file_path,
    participant_id,
    condition,
    map_type,
    verbose=False
):
    """
    מריץ את כל שלבי התמלול בבת אחת עבור נסיעה נוכחית:
    1) load_transcription_data (מסונן לנסיעה)
    2) add_transcription_to_events (מדביק text+speaker)
    3) add_comment_flag
    4) add_start_comment_column (שיטה B: nearest start)
    5) add_first_feedback_in_event
    מחזיר:
    - df_with_final_comments: הדאטה הראשי אחרי הדבקה+דגלים
    - df_transcripts: טבלת התמלולים של הנסיעה (לדיבוג/שימוש עתידי)
    """
    df_transcripts = load_transcription_data(
        transcription_file_path,
        participant_id=participant_id,
        condition=condition,
        map_type=map_type
    )
    if df_transcripts is None or df_transcripts.empty:
        df_out = df_with_ttc.copy()
        if "text" not in df_out.columns:
            df_out["text"] = pd.NA
        if "speaker" not in df_out.columns:
            df_out["speaker"] = pd.NA
        if "transcript_type" not in df_out.columns:
            df_out["transcript_type"] = pd.NA
        df_out = add_comment_flag(df_out)
        df_out["start_comment"] = 0
        df_out["first_feedback_in_event"] = pd.NA  # אין תמלולים -> הכל NA
        if verbose:
            print("process_transcription_pipeline: df_transcripts empty -> returned without transcripts.")
        return df_out, df_transcripts

    if verbose:
        print("After load_transcription_data:", df_transcripts.shape)

    df_with_text = add_transcription_to_events(df_with_ttc, df_transcripts)
    if verbose:
        print("After add_transcription_to_events:", df_with_text.shape)

    df_with_final_comments = add_comment_flag(df_with_text)
    if verbose:
        print("After add_comment_flag:", df_with_final_comments.shape)

    df_with_final_comments = add_start_comment_column(df_with_final_comments, df_transcripts)
    if verbose:
        print("After add_start_comment_column:", df_with_final_comments.shape)

    df_with_final_comments = add_first_feedback_in_event(df_with_final_comments)  # ✅ שלב חדש
    if verbose:
        print("After add_first_feedback_in_event:", df_with_final_comments.shape)

    return df_with_final_comments, df_transcripts
