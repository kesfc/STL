import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HIST_YEARS = range(2005, 2026)  # Range of ending years to iterate through (2005–2025 seasons)

summary_root = os.path.join("Team_stats", "summary")  # Root folder containing per-season summary CSVs

rows = []
for end_year in HIST_YEARS:
    season = f"{end_year - 1}-{end_year}"  # Construct season string like "2004-2005"
    path = os.path.join(summary_root, f"{season}.csv")  # Path to the season summary file
    if not os.path.exists(path):
        print(f"[warning] file not found: {path}, skip")  # Warn if file missing and skip
        continue

    df_season = pd.read_csv(path)  # Load season summary CSV
    rows.append(df_season)  # Store DataFrame for later concatenation

data = pd.concat(rows, ignore_index=True)  # Combine all seasons into one DataFrame

print("All seasons shape:", data.shape)  # Show final merged dataset size
print("Columns:", list(data.columns))  # Show available columns
out_all_path = os.path.join("Team_stats", "all_seasons_team_summary.csv")  # Output file path
os.makedirs(os.path.dirname(out_all_path), exist_ok=True)  # Ensure directory exists
data.to_csv(out_all_path, index=False)  # Save combined dataset
print(f"All seasons combined saved to {out_all_path}")  # Notify user of output location
