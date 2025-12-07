import pandas as pd
import os

S = range(2005, 2026)  # Range of ending years for seasons (e.g., 2004-2005 up to 2024-2025)
for i in S:
    SEASON = f"{i - 1}-{i}"  # Season label in the format "YYYY-YYYY", e.g., "2004-2005"
    print(f"Processing season {SEASON} ...")

    # Path to player-level stats for this season
    stats_path = os.path.join("Player_stats", "all_stats", f"{SEASON}.csv")
    df = pd.read_csv(stats_path)

    # Remove aggregated multi-team rows like "2TM", "3TM", etc. to avoid double-counting
    df = df[~df["Team"].isin(["2TM", "3TM", "4TM", "5TM"])]

    # Columns of counting stats to aggregate at the team level
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

    # Sum player stats by team to obtain team-level totals
    team_stats = df.groupby("Team", as_index=False)[stat_cols].sum()

    # Compute shooting percentage metrics based on aggregated totals
    team_stats["FG%"] = team_stats["FG"] / team_stats["FGA"]
    team_stats["3P%"] = team_stats["3P"] / team_stats["3PA"]
    team_stats["2P%"] = team_stats["2P"] / team_stats["2PA"]
    team_stats["FT%"] = team_stats["FT"] / team_stats["FTA"]
    team_stats["eFG%"] = (team_stats["FG"] + 0.5 * team_stats["3P"]) / team_stats["FGA"]

    # Replace NaN percentages (e.g., when denominator is zero) with 0
    for col in ["FG%", "3P%", "2P%", "FT%", "eFG%"]:
        team_stats[col] = team_stats[col].fillna(0)

    # Path to team win–loss information for this season
    wl_path = os.path.join("Team_stats", "WL", f"{SEASON}.csv")
    wl_df = pd.read_csv(wl_path)

    # Merge team stats with win–loss data based on team abbreviation
    team_stats = team_stats.merge(
        wl_df,
        left_on="Team",
        right_on="TEAM_ABBR",
        how="left"
    )

    # Remove duplicate team abbreviation column from the merged dataframe
    team_stats = team_stats.drop(columns=["TEAM_ABBR"])

    # Add season metadata columns
    team_stats["Season"] = SEASON  # Season label string
    team_stats["SeasonEndYear"] = i  # Season ending year as integer
    # Reorder columns: put season info and team name at the front
    cols = ["Season", "SeasonEndYear", "Team"] + [
        c for c in team_stats.columns
        if c not in ["Season", "SeasonEndYear", "Team"]
    ]
    team_stats = team_stats[cols]

    # Ensure output directory exists for per-season team summaries
    out_dir = os.path.join("Team_stats", "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SEASON}.csv")

    # Save team-level summary stats for this season
    team_stats.to_csv(out_path, index=False)

    # Show a preview and where the file was saved
    print(team_stats.head())
    print(f"Saved to {out_path}")
