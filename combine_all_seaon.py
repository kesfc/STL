import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HIST_YEARS = range(2005, 2026)        

summary_root = os.path.join("Team_stats", "summary")

rows = []
for end_year in HIST_YEARS:
    season = f"{end_year-1}-{end_year}"  
    path = os.path.join(summary_root, f"{season}.csv")
    if not os.path.exists(path):
        print(f"[warning] file not found: {path}, skip")
        continue

    df_season = pd.read_csv(path)
    rows.append(df_season)

data = pd.concat(rows, ignore_index=True)

print("All seasons shape:", data.shape)
print("Columns:", list(data.columns))
out_all_path = os.path.join("Team_stats", "all_seasons_team_summary.csv")
os.makedirs(os.path.dirname(out_all_path), exist_ok=True)
data.to_csv(out_all_path, index=False)
print(f"All seasons combined saved to {out_all_path}")