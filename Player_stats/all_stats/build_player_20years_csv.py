import os
from pathlib import Path

import pandas as pd

# Base directory where per-season CSV files are stored
DATA_DIR = Path(".")

# Output file path for the merged 20-year player dataset
OUTPUT_CSV = DATA_DIR / "player_20years.csv"

# List to collect DataFrames from each season
all_dfs = []

# Loop over start years for seasons from 2004-2005 up to 2025-2026
for start_year in range(2004, 2026):
    end_year = start_year + 1
    # Season label like "2004-2005"
    season_name = f"{start_year}-{end_year}"
    # Expected CSV path for this season
    csv_path = DATA_DIR / f"{season_name}.csv"

    # Skip if the per-season CSV does not exist
    if not csv_path.exists():
        print(f"[WARN] File not found, skip: {csv_path}")
        continue

    print(f"[INFO] Loading: {csv_path}")
    # Load the season-level player stats CSV
    df = pd.read_csv(csv_path)

    # Add a Season column so we know which season each row belongs to
    df["Season"] = season_name

    # Store this season's DataFrame for later concatenation
    all_dfs.append(df)

# If no files were successfully loaded, raise an error
if not all_dfs:
    raise RuntimeError("No CSV files were loaded. Check file names and DATA_DIR.")

# Concatenate all season DataFrames into a single DataFrame
merged = pd.concat(all_dfs, ignore_index=True)

print(f"[INFO] Merged shape: {merged.shape}")

# Save the merged multi-season player dataset to a single CSV
merged.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved merged CSV to: {OUTPUT_CSV}")
