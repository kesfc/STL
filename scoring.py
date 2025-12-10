import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt

all_players = []   # Store all players' data across all seasons

# === Iterate over all seasons ===
S = range(2005, 2026)
for i in S:
    SEASON = f"{i - 1}-{i}"
    path = os.path.join("Player_stats", "all_stats", f"{SEASON}.csv")

    if not os.path.exists(path):
        print(f"Missing season file: {path}")
        continue
    
    df = pd.read_csv(path)

    # Remove combined rows like "2TM" to avoid double counting multi-team stats
    df = df[~df["Team"].isin(["2TM", "3TM", "4TM", "5TM"])]

    df["Season"] = SEASON
    all_players.append(df)

# === Concatenate data from all seasons ===
players = pd.concat(all_players, ignore_index=True)

print("Total rows loaded:", len(players))


# ============================================================
# 1. Total points by player age
# ============================================================
# Filter out rows where Age or PTS is NaN
age_pts = (
    players[["Age", "PTS"]]
    .dropna()
    .groupby("Age")["PTS"]
    .sum()                      # ★ Use total points (sum)
    .reset_index()
    .sort_values("Age")
)

print("\n=== Total points (by age) ===")
print(age_pts.head(15))


# ============================================================
# 2. Total points by number of seasons played (experience)
# ============================================================

# Count number of seasons each player appears in as "Seasons_Played"
# Note: here we assume the same name corresponds to the same player
player_seasons = players.groupby("Player")["Season"].nunique().reset_index()
player_seasons.columns = ["Player", "Seasons_Played"]

# Merge back so each player record has its experience in seasons
players_with_exp = players.merge(player_seasons, on="Player", how="left")

exp_pts = (
    players_with_exp[["Seasons_Played", "PTS"]]
    .dropna()
    .groupby("Seasons_Played")["PTS"]
    .sum()                     # ★ Also use total points (sum)
    .reset_index()
    .sort_values("Seasons_Played")
)

print("\n=== Total points (by seasons played) ===")
print(exp_pts.head(15))


# ============================================================
# Optional: export as CSV
# ============================================================
age_pts.to_csv("age_total_points.csv", index=False)
exp_pts.to_csv("seasons_experience_total_points.csv", index=False)

print("\nSaved CSVs:")
print(" - age_total_points.csv")
print(" - seasons_experience_total_points.csv")


# ============================================================
# Plotting: two figures
# ============================================================

# Figure 1: Total points by age
plt.figure(figsize=(10, 6))
plt.plot(age_pts["Age"], age_pts["PTS"], marker="o")
plt.xlabel("Age")
plt.ylabel("Total Points (All Seasons Combined)")
plt.title("Total Points by Age (All Seasons)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_total_points_by_age.png", dpi=200)
plt.show()

# Figure 2: Total points by seasons of experience
plt.figure(figsize=(10, 6))
plt.plot(exp_pts["Seasons_Played"], exp_pts["PTS"], marker="o")
plt.xlabel("Seasons Played")
plt.ylabel("Total Points (All Seasons Combined)")
plt.title("Total Points by Seasons of Experience")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_total_points_by_experience.png", dpi=200)
plt.show()