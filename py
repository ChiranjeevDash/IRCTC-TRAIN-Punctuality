# ============================================
# Per-Train Delay Stats (Clean Output Version)
# Dataset: https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets
# ============================================

import os
import pandas as pd
import numpy as np

# ---------- 1. Clone dataset ----------
!rm -rf DA323_IndianRailwayTrainDelayDatasets
!git clone -q https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets.git

base_dir = "/content/DA323_IndianRailwayTrainDelayDatasets/Dataset"
train_list_path = os.path.join(base_dir, "Train_List.csv")

# ---------- 2. Load train list ----------
train_list = pd.read_csv(train_list_path)

# Detect number & name columns
num_col_candidates = [c for c in train_list.columns if "number" in c.lower()]
name_col_candidates = [c for c in train_list.columns if "name" in c.lower()]

if not num_col_candidates or not name_col_candidates:
    raise ValueError("Could not detect Train_Number / Train_Name columns in Train_List.csv")

TRAIN_NO_COL = num_col_candidates[0]
TRAIN_NAME_COL = name_col_candidates[0]

train_list["train_name_clean"] = (
    train_list[TRAIN_NAME_COL].astype(str).str.lower().str.strip()
)
train_list["train_no_str"] = train_list[TRAIN_NO_COL].astype(str).str.strip()

print("=" * 60)
print(f"Loaded {len(train_list)} trains from dataset.")
print("=" * 60)

# ---------- 3. Function for clean stats ----------
def get_train_delay_stats(train_query, top_n=5):
    """
    train_query: part of train name, e.g. 'rajdhani', 'guwahati', 'new delhi'
    """
    q = train_query.lower().strip()
    matches = train_list[train_list["train_name_clean"].str.contains(q)]

    print("\n" + "=" * 60)
    print(f" Search query: '{train_query}'")
    print("=" * 60)

    if matches.empty:
        print("No trains found.\nHere are a few example trains from the dataset:\n")
        print(
            train_list[[TRAIN_NO_COL, TRAIN_NAME_COL]]
            .head(10)
            .to_string(index=False)
        )
        print("=" * 60)
        return None

    # Show top matches
    subset = matches[[TRAIN_NO_COL, TRAIN_NAME_COL]].head(top_n)
    print("Matching trains:")
    print(subset.to_string(index=False))

    # Pick the first match
    chosen = matches.iloc[0]
    train_no = chosen["train_no_str"]
    train_name = chosen[TRAIN_NAME_COL]

    print("\nUsing first match:")
    print(f"  Train No : {train_no}")
    print(f"  Train    : {train_name}")

    # ---------- 4. Load route file ----------
    route_path = os.path.join(base_dir, "Train_Route", f"{train_no}.csv")
    if not os.path.exists(route_path):
        print("\n[!] Route CSV not found for this train:")
        print(f"    {route_path}")
        print("=" * 60)
        return None

    route_df = pd.read_csv(route_path)

    # Detect average delay column
    avg_delay_col = [
        c for c in route_df.columns
        if "average" in c.lower() and "delay" in c.lower()
    ]
    if not avg_delay_col:
        raise ValueError("Could not detect an 'Average Delay' column.")
    AVG_DELAY_COL = avg_delay_col[0]

    # Detect probability columns
    prob_cols = {}
    for c in route_df.columns:
        cl = c.lower()
        if "right time" in cl or "0-15" in cl:
            prob_cols["right_time"] = c
        elif "slight" in cl or "15-60" in cl:
            prob_cols["slight_delay"] = c
        elif "significant" in cl or ">1 hour" in cl or "> 1 hour" in cl:
            prob_cols["significant_delay"] = c
        elif "cancelled" in cl or "unknown" in cl:
            prob_cols["cancelled_unknown"] = c

    # Clean numeric values
    route_df[AVG_DELAY_COL] = pd.to_numeric(route_df[AVG_DELAY_COL], errors="coerce")

    for key, col in prob_cols.items():
        route_df[col] = (
            route_df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        route_df[col] = pd.to_numeric(route_df[col], errors="coerce")

    # ---------- 5. Aggregate stats ----------
    avg_delay_minutes = route_df[AVG_DELAY_COL].mean()

    probs = {}
    for key, col in prob_cols.items():
        if col in route_df:
            probs[key] = (route_df[col].mean() or 0.0) / 100.0

    # ---------- 6. Pretty output ----------
    print("\n" + "-" * 60)
    print("   HISTORICAL DELAY SUMMARY")
    print("-" * 60)
    print(f"Train No : {train_no}")
    print(f"Train    : {train_name}")
    print(f"Average delay across route : {avg_delay_minutes:6.2f} minutes")

    print("\nDelay probability (approx, based on history):")
    print(f"  • On time / ≤15 min late : {probs.get('right_time', 0)*100:6.2f}%")
    print(f"  • 15–60 min late         : {probs.get('slight_delay', 0)*100:6.2f}%")
    print(f"  • >60 min late           : {probs.get('significant_delay', 0)*100:6.2f}%")
    print(f"  • Cancelled / unknown    : {probs.get('cancelled_unknown', 0)*100:6.2f}%")
    print("-" * 60 + "\n")

    return {
        "train_no": train_no,
        "train_name": train_name,
        "avg_delay_min": avg_delay_minutes,
        "probs": probs,
    }

_ = get_train_delay_stats("rajdhani")
