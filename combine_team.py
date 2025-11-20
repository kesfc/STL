import pandas as pd
import os

S = range(2025, 2027) 
for i in S:
    SEASON = f"{i-1}-{i}"
    print(f"Processing season {SEASON} ...")

    stats_path = os.path.join("Player_stats", "all_stats", f"{SEASON}.csv")
    df = pd.read_csv(stats_path)

    df = df[~df["Team"].isin(["2TM", "3TM", "4TM", "5TM"])]

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

    wl_path = os.path.join("Team_stats", "WL", f"{SEASON}.csv")
    wl_df = pd.read_csv(wl_path)

    team_stats = team_stats.merge(
        wl_df,
        left_on="Team",
        right_on="TEAM_ABBR",
        how="left"
    )

    team_stats = team_stats.drop(columns=["TEAM_ABBR"])

    team_stats["Season"] = SEASON              
    team_stats["SeasonEndYear"] = i          
    cols = ["Season", "SeasonEndYear", "Team"] + [
        c for c in team_stats.columns
        if c not in ["Season", "SeasonEndYear", "Team"]
    ]
    team_stats = team_stats[cols]

    out_dir = os.path.join("Team_stats", "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SEASON}.csv")

    team_stats.to_csv(out_path, index=False)

    print(team_stats.head())
    print(f"Saved to {out_path}")
