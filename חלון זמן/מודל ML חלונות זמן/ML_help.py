"""
ML_help.py
----------
Histogram of time-until-first-feedback for each traffic light event.

For every event_id that received feedback:
  - Find the start_time of the first window where feedback_next_1s == 1
  - That is the "time until first feedback" (seconds since entering the TL approach zone)

Plots:
  - One histogram per traffic light number (TL1, TL2, TL3, ...)
  - One combined histogram across all traffic lights
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "sliding_windows_1s_clean.csv"
)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ── Extract traffic light number from event_id ────────────────────────────────
# event_id format: "C1_036248__Avatar__traffic light 2"
df["tl_number"] = (
    df["event_id"]
    .str.extract(r"traffic light (\d+)", flags=re.IGNORECASE)[0]
    .astype(float)
    .astype("Int64")
)

# ── For each event, find the start_time of the FIRST feedback window ──────────
feedback_rows = df[df["feedback_next_1s"] == 1].copy()

first_feedback = (
    feedback_rows
    .sort_values("start_time")
    .groupby("event_id", as_index=False)
    .agg(
        time_to_first_feedback=("start_time", "first"),
        tl_number=("tl_number", "first"),
        Id=("Id", "first"),
        Condition=("Condition", "first"),
        Map=("Map", "first"),
    )
)

print(f"Events with feedback: {len(first_feedback)}")
print(f"Traffic lights found: {sorted(first_feedback['tl_number'].dropna().unique())}")
print()
print(first_feedback[["Id", "Condition", "Map", "tl_number", "time_to_first_feedback"]]
      .sort_values(["tl_number", "time_to_first_feedback"])
      .to_string(index=False))

# ── Output folder ────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML_help output")
os.makedirs(OUT_DIR, exist_ok=True)

# Save the per-event table as CSV
csv_path = os.path.join(OUT_DIR, "time_to_first_feedback.csv")
first_feedback.to_csv(csv_path, index=False)
print(f"Table saved to: {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
tl_numbers = sorted(first_feedback["tl_number"].dropna().unique())
n_tl = len(tl_numbers)

# One row of subplots: one per TL + one combined
fig, axes = plt.subplots(1, n_tl + 1, figsize=(5 * (n_tl + 1), 5), sharey=False)

bin_width = 2  # seconds per bin

for i, tl in enumerate(tl_numbers):
    ax = axes[i]
    data = first_feedback[first_feedback["tl_number"] == tl]["time_to_first_feedback"]
    max_t = data.max()
    bins = range(0, int(max_t) + bin_width + 1, bin_width)

    ax.hist(data, bins=bins, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_title(f"Traffic Light {tl}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Seconds into approach", fontsize=11)
    ax.set_ylabel("Number of drives", fontsize=11)
    ax.set_xlim(left=0)
    ax.yaxis.get_major_locator().set_params(integer=True)

    # Annotate mean and median
    ax.axvline(data.mean(),   color="red",    linestyle="--", linewidth=1.4,
               label=f"Mean {data.mean():.1f}s")
    ax.axvline(data.median(), color="orange", linestyle=":",  linewidth=1.4,
               label=f"Median {data.median():.1f}s")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, f"n={len(data)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="gray")

# Combined histogram (last subplot)
ax_all = axes[-1]
all_data = first_feedback["time_to_first_feedback"]
max_t_all = all_data.max()
bins_all = range(0, int(max_t_all) + bin_width + 1, bin_width)

ax_all.hist(all_data, bins=bins_all, color="darkorchid", edgecolor="white", linewidth=0.6)
ax_all.set_title("All Traffic Lights Combined", fontsize=13, fontweight="bold")
ax_all.set_xlabel("Seconds into approach", fontsize=11)
ax_all.set_ylabel("Number of drives", fontsize=11)
ax_all.set_xlim(left=0)
ax_all.yaxis.get_major_locator().set_params(integer=True)

ax_all.axvline(all_data.mean(),   color="red",    linestyle="--", linewidth=1.4,
               label=f"Mean {all_data.mean():.1f}s")
ax_all.axvline(all_data.median(), color="orange", linestyle=":",  linewidth=1.4,
               label=f"Median {all_data.median():.1f}s")
ax_all.legend(fontsize=9)
ax_all.text(0.97, 0.95, f"n={len(all_data)}", transform=ax_all.transAxes,
            ha="right", va="top", fontsize=10, color="gray")

plt.suptitle("Time Until First Instructor Feedback — by Traffic Light",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

# Save + show
out_path = os.path.join(OUT_DIR, "time_to_first_feedback.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()
