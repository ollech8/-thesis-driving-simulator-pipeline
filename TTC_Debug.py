# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:10:02 2026

@author: ASUS VIVOBOOK
"""

import json
import pandas as pd


def json_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads your simulation JSON file into a pandas DataFrame.
    Handles both common formats:
    1) {"Logs": [ {..}, {..}, ... ]}
    2) [ {..}, {..}, ... ]  (list at root)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: dict with "Logs"
    if isinstance(data, dict) and "Logs" in data:
        logs = data["Logs"]
    # Case 2: list at root
    elif isinstance(data, list):
        logs = data
    else:
        raise ValueError(f"Unexpected JSON structure in: {file_path}")

    df = pd.json_normalize(logs)
    return df


def show_df_overview(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print("=" * 90)
    print(f"{name} | shape: {df.shape}")
    print("- Columns:")
    print(df.columns.tolist())
    print("- Dtypes:")
    print(df.dtypes)
    print("- Head:")
    print(df.head(n))
    print("- Tail:")
    print(df.tail(n))
    print("- Missing values (top 30):")
    print(df.isna().sum().sort_values(ascending=False).head(30))
    print("=" * 90)


# --- Paths (your files) ---
# egocar_file_path = r"H:\האחסון שלי\Ariel Uni\C1_120624\Simulation\Accompanied\Avatar\EgoCar_Simulation_Avatar_MapC_2025-04-21_14-34-46.json"
# object_file_path = r"H:\האחסון שלי\Ariel Uni\C1_120624\Simulation\Accompanied\Avatar\Objects_Simulation_Avatar_MapC_2025-04-21_14-34-45.json"
egocar_file_path = r"H:\האחסון שלי\Ariel Uni\C2_031421\Simulation\Accompanied\Avatar\EgoCar_Simulation_Avatar_MapB_2025-01-30_16-50-13.json"
object_file_path = r"H:\האחסון שלי\Ariel Uni\C2_031421\Simulation\Accompanied\Avatar\Objects_Simulation_Avatar_MapB_2025-01-30_16-50-12.json"
# --- Load ---
df_egocar = json_to_dataframe(egocar_file_path)
df_objects = json_to_dataframe(object_file_path)

# --- Quick overview ---
show_df_overview(df_egocar, "EgoCar")
show_df_overview(df_objects, "Objects")

# --- Optional: save to CSV for easier inspection ---
# df_egocar.to_csv(r"H:\egocar_debug.csv", index=False, encoding="utf-8-sig")
# df_objects.to_csv(r"H:\objects_debug.csv", index=False, encoding="utf-8-sig")
