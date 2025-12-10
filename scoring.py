import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt

all_players = []   # 存放所有赛季的所有球员数据

# === 遍历所有赛季 ===
S = range(2005, 2026)
for i in S:
    SEASON = f"{i - 1}-{i}"
    path = os.path.join("Player_stats", "all_stats", f"{SEASON}.csv")

    if not os.path.exists(path):
        print(f"Missing season file: {path}")
        continue
    
    df = pd.read_csv(path)

    # 去掉 "2TM" 等多队合并行，避免重复计算
    df = df[~df["Team"].isin(["2TM", "3TM", "4TM", "5TM"])]

    df["Season"] = SEASON
    all_players.append(df)

# === 合并所有赛季的数据 ===
players = pd.concat(all_players, ignore_index=True)

print("Total rows loaded:", len(players))


# ============================================================
# 1. 不同年龄球员的【总得分】
# ============================================================
# 过滤掉 Age 或 PTS 为 NaN 的行
age_pts = (
    players[["Age", "PTS"]]
    .dropna()
    .groupby("Age")["PTS"]
    .sum()                      # ★ 改成总得分 sum
    .reset_index()
    .sort_values("Age")
)

print("\n=== 总得分（按年龄） ===")
print(age_pts.head(15))


# ============================================================
# 2. 根据球员打了几个赛季（经验年数）的【总得分】
# ============================================================

# 统计每名球员出现的赛季数量作为 "Seasons_Played"
# 注意：这里假设相同名字就是同一球员
player_seasons = players.groupby("Player")["Season"].nunique().reset_index()
player_seasons.columns = ["Player", "Seasons_Played"]

# 合并回去，让每条球员记录带上他的经验年数
players_with_exp = players.merge(player_seasons, on="Player", how="left")

exp_pts = (
    players_with_exp[["Seasons_Played", "PTS"]]
    .dropna()
    .groupby("Seasons_Played")["PTS"]
    .sum()                     # ★ 同样用总得分 sum
    .reset_index()
    .sort_values("Seasons_Played")
)

print("\n=== 总得分（按打了几个赛季） ===")
print(exp_pts.head(15))


# ============================================================
# 可选择导出 CSV
# ============================================================
age_pts.to_csv("age_total_points.csv", index=False)
exp_pts.to_csv("seasons_experience_total_points.csv", index=False)

print("\nSaved CSVs:")
print(" - age_total_points.csv")
print(" - seasons_experience_total_points.csv")


# ============================================================
# 画图部分：两张图
# ============================================================

# 图 1：不同年龄的总得分
plt.figure(figsize=(10, 6))
plt.plot(age_pts["Age"], age_pts["PTS"], marker="o")
plt.xlabel("Age")
plt.ylabel("Total Points (All Seasons Combined)")
plt.title("Total Points by Age (All Seasons)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_total_points_by_age.png", dpi=200)
plt.show()

# 图 2：按打了几个赛季的总得分
plt.figure(figsize=(10, 6))
plt.plot(exp_pts["Seasons_Played"], exp_pts["PTS"], marker="o")
plt.xlabel("Seasons Played")
plt.ylabel("Total Points (All Seasons Combined)")
plt.title("Total Points by Seasons of Experience")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_total_points_by_experience.png", dpi=200)
plt.show()
