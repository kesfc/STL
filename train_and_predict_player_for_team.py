"""
Plan 5: Use aggregated player stats to predict team win rate (high-win vs low-win).

- Input:
    Player_stats/all_stats/player_20years.csv
    Team_stats/all_seasons_team_summary.csv

- Target:
    Binary label HIGH_WIN:
        1 if team WIN_PCT >= 0.55
        0 otherwise

- Split:
    Time-based split by season (train on early seasons, test on last 3 seasons).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PLAYER_CSV = Path("Player_stats/all_stats/player_20years.csv")
TEAM_CSV = Path("Team_stats/all_seasons_team_summary.csv")

MIN_MINUTES = 200          # minimum MP for a player-season to be included in aggregation
WIN_PCT_THRESHOLD = 0.55   # threshold for "high win" team
TARGET_ACC = 0.80          # target test accuracy


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-player features (per-36 stats, rates) and return updated DataFrame."""

    eps = 1e-6
    mp = df["MP"].replace(0, np.nan)

    # Per-36 stats
    df["PTS_per36"] = df["PTS"] / (mp + eps) * 36.0
    df["AST_per36"] = df["AST"] / (mp + eps) * 36.0
    df["TRB_per36"] = df["TRB"] / (mp + eps) * 36.0
    df["STL_per36"] = df["STL"] / (mp + eps) * 36.0
    df["BLK_per36"] = df["BLK"] / (mp + eps) * 36.0
    df["TOV_per36"] = df["TOV"] / (mp + eps) * 36.0

    df["FGA_per36"] = df["FGA"] / (mp + eps) * 36.0
    df["FTA_per36"] = df["FTA"] / (mp + eps) * 36.0
    df["3PA_per36"] = df["3PA"] / (mp + eps) * 36.0

    # Rate features
    df["ThreePointRate"] = df["3PA"] / (df["FGA"] + eps)          # 3PA / FGA
    df["FreeThrowRate"] = df["FTA"] / (df["FGA"] + eps)          # FTA / FGA
    df["AST_TOV_Ratio"] = df["AST"] / (df["TOV"] + eps)
    df["UsageProxy"] = (df["FGA"] + 0.44 * df["FTA"]) / (mp + eps)  # crude usage proxy

    return df


def aggregate_team_from_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level stats into team-season level features.

    - Sum counting stats (MP, PTS, etc.).
    - Average per-36 and rate features.
    - Count number of players used.
    """

    group_cols = ["Season", "Team"]

    # Columns to sum at team level (counting stats)
    sum_cols = [
        "MP", "G", "GS",
        "FG", "FGA", "3P", "3PA", "2P", "2PA",
        "FT", "FTA",
        "ORB", "DRB", "TRB",
        "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]

    # Columns to average (per-36 and rates)
    mean_cols = [
        "PTS_per36", "AST_per36", "TRB_per36", "STL_per36", "BLK_per36",
        "TOV_per36", "FGA_per36", "FTA_per36", "3PA_per36",
        "ThreePointRate", "FreeThrowRate", "AST_TOV_Ratio", "UsageProxy",
        "Age", "FG%", "3P%", "2P%", "eFG%", "FT%",
    ]
    mean_cols = [c for c in mean_cols if c in df.columns]

    agg_dict = {}

    for c in sum_cols:
        agg_dict[c] = "sum"
    for c in mean_cols:
        agg_dict[c] = "mean"

    # Also keep number of unique players as a feature
    agg_dict["Player"] = "nunique"

    team_feat = (
        df.groupby(group_cols)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"Player": "NumPlayers"})
    )

    return team_feat


# ----------------------------------------------------------------------
# Load player data and build team-level features
# ----------------------------------------------------------------------
print(f"[INFO] Loading player data from: {PLAYER_CSV}")
player_df = pd.read_csv(PLAYER_CSV)

# Extract SeasonStart year (e.g., "2004-2005" -> 2004) for time-based split later
player_df["SeasonStart"] = player_df["Season"].str.split("-").str[0].astype(int)

# Remove "TOT" rows (total across multiple teams) because we want per-team contributions
before_tot = len(player_df)
player_df = player_df[player_df["Team"] != "TOT"].copy()
after_tot = len(player_df)
print(f"[INFO] Dropped {before_tot - after_tot} 'TOT' rows.")

# Filter players with low minutes (to reduce noise)
before_mp = len(player_df)
player_df = player_df[player_df["MP"] >= MIN_MINUTES].copy()
after_mp = len(player_df)
print(f"[INFO] Dropped {before_mp - after_mp} rows with MP < {MIN_MINUTES}.")
print(f"[INFO] Remaining player-rows: {after_mp}")

# Build per-player features
player_df = build_player_features(player_df)

# Aggregate into team-season features
team_from_players = aggregate_team_from_players(player_df)
print(f"[INFO] Team-level rows from players: {team_from_players.shape[0]}")

# Also keep SeasonStart for splitting (use first year of Season string)
team_from_players["SeasonStart"] = team_from_players["Season"].str.split("-").str[0].astype(int)

# ----------------------------------------------------------------------
# Load team labels (win percentage) and join
# ----------------------------------------------------------------------
print(f"[INFO] Loading team data from: {TEAM_CSV}")
team_df = pd.read_csv(TEAM_CSV)

# Compute win percentage from WINS / LOSSES
team_df["WIN_PCT"] = team_df["WINS"] / (team_df["WINS"] + team_df["LOSSES"])
team_df["SeasonStart"] = team_df["Season"].str.split("-").str[0].astype(int)

# We only need Season, Team, WINS, LOSSES, WIN_PCT
team_labels = team_df[["Season", "Team", "SeasonStart", "WINS", "LOSSES", "WIN_PCT"]]

# Inner join: only seasons/teams that exist in both player aggregation and team stats
merged = pd.merge(
    team_from_players,
    team_labels,
    on=["Season", "Team", "SeasonStart"],
    how="inner",
)

print(f"[INFO] Merged team-feature rows: {merged.shape[0]}")

# Build binary label: high-win vs low-win
merged["HIGH_WIN"] = (merged["WIN_PCT"] >= WIN_PCT_THRESHOLD).astype(int)
print(
    "[INFO] Label distribution (HIGH_WIN):\n",
    merged["HIGH_WIN"].value_counts(normalize=True).rename("ratio"),
)

# ----------------------------------------------------------------------
# Train / test split by season (time-based)
# ----------------------------------------------------------------------
unique_seasons = sorted(merged["SeasonStart"].unique())
if len(unique_seasons) <= 4:
    raise RuntimeError("Not enough seasons for proper train/test split.")

# Use last 3 seasons as test, the rest as train
test_seasons = unique_seasons[-3:]
train_seasons = [s for s in unique_seasons if s not in test_seasons]

train_df = merged[merged["SeasonStart"].isin(train_seasons)].copy()
test_df = merged[merged["SeasonStart"].isin(test_seasons)].copy()

print(f"[INFO] Train seasons: {train_seasons}")
print(f"[INFO] Test seasons:  {test_seasons}")
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

# ----------------------------------------------------------------------
# Build feature matrix X and labels y
# ----------------------------------------------------------------------
# Columns to exclude from features
exclude_cols = {
    "Season", "Team", "SeasonStart",
    "WINS", "LOSSES", "WIN_PCT", "HIGH_WIN",
}

feature_cols = [
    c for c in merged.columns
    if c not in exclude_cols and merged[c].dtype != "O"  # exclude non-numeric
]

print(f"[INFO] Number of numeric feature columns: {len(feature_cols)}")
print(f"[INFO] Example feature columns: {feature_cols[:10]}")

X_train = train_df[feature_cols].fillna(0.0).values
y_train = train_df["HIGH_WIN"].values

X_test = test_df[feature_cols].fillna(0.0).values
y_test = test_df["HIGH_WIN"].values

# Compute positive rate to set scale_pos_weight
pos_rate = y_train.mean()
if pos_rate == 0 or pos_rate == 1:
    scale_pos_weight = 1.0
else:
    scale_pos_weight = (1 - pos_rate) / pos_rate

print(f"[INFO] Positive rate in train (HIGH_WIN=1): {pos_rate:.3f}, scale_pos_weight={scale_pos_weight:.2f}")

# ----------------------------------------------------------------------
# Simple inner validation to choose best of a few configs
# ----------------------------------------------------------------------
# Inner split: last 2 train seasons as validation, earlier as inner-train
inner_seasons = sorted(train_df["SeasonStart"].unique())
val_seasons = inner_seasons[-2:]
inner_train_seasons = inner_seasons[:-2]

inner_train_mask = train_df["SeasonStart"].isin(inner_train_seasons)
val_mask = train_df["SeasonStart"].isin(val_seasons)

X_inner_train = train_df[inner_train_mask][feature_cols].fillna(0.0).values
y_inner_train = train_df[inner_train_mask]["HIGH_WIN"].values

X_val = train_df[val_mask][feature_cols].fillna(0.0).values
y_val = train_df[val_mask]["HIGH_WIN"].values

print(f"[INFO] Inner-train size: {X_inner_train.shape[0]}, Val size: {X_val.shape[0]}")

configs = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.1},
]

best_cfg = None
best_val_acc = -1.0

for cfg in configs:
    print(f"[INFO] Trying config: {cfg}")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )),
    ])

    model.fit(X_inner_train, y_inner_train)
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  -> Val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_cfg = cfg

print(f"[INFO] Best config on validation: {best_cfg} (val acc={best_val_acc:.4f})")

# ----------------------------------------------------------------------
# Train final model on full training data with best config
# ----------------------------------------------------------------------
print("[INFO] Training final model on full training data...")

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        learning_rate=best_cfg["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )),
])

final_model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate on test set
# ----------------------------------------------------------------------
print("\n[INFO] Evaluating on test set...")

y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n[RESULT] Test accuracy: {test_acc:.4f}")

print("\n[RESULT] Classification report (test):")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["LowWin", "HighWin"]
))

print("[RESULT] Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Baseline: always predict majority class
majority_class = int(np.round(merged["HIGH_WIN"].value_counts().idxmax()))
baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))
print(f"\n[BASELINE] Always predict '{'HighWin' if majority_class == 1 else 'LowWin'}' "
      f"accuracy: {baseline_acc:.4f}")

# Check target
if test_acc >= TARGET_ACC:
    print(f"\n[INFO] Target achieved: accuracy >= {TARGET_ACC * 100:.0f}%")
else:
    print(f"\n[WARN] Target NOT achieved: accuracy={test_acc:.4f}, target={TARGET_ACC:.2f}")