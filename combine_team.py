import pandas as pd

df = pd.read_csv("Player_stats/all_stats/2024-2025.csv")

df = df[~df["Team"].isin(["2TM", "3TM"])]

stat_cols = [
    "FG", "FGA",
    "3P", "3PA",
    "2P", "2PA",
    "FT", "FTA",
    "ORB", "DRB", "TRB",
    "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
    "Trp-Dbl",
]

team_stats = df.groupby("Team", as_index=False)[stat_cols].sum()

team_stats["FG%"]  = team_stats["FG"]  / team_stats["FGA"]
team_stats["3P%"]  = team_stats["3P"]  / team_stats["3PA"]
team_stats["2P%"]  = team_stats["2P"]  / team_stats["2PA"]
team_stats["FT%"]  = team_stats["FT"]  / team_stats["FTA"]
team_stats["eFG%"] = (team_stats["FG"] + 0.5 * team_stats["3P"]) / team_stats["FGA"]

for col in ["FG%", "3P%", "2P%", "FT%", "eFG%"]:
    team_stats[col] = team_stats[col].fillna(0)

team_stats.to_csv("2024-2025_team_summary.csv", index=False)

print(team_stats.head())
