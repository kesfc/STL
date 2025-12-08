import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Path to the CSV file with 20 years of player stats
DATA_PATH = Path("all_stats") / "player_20years.csv"

# Whether to use 3-class position labels (Guard/Wing/Big) instead of 5-class (PG/SG/SF/PF/C)
USE_THREE_CLASS = True

# Minimum total minutes requirement to keep a season (filter out very small samples)
MIN_MINUTES = 500


def map_pos_five_way(pos: str) -> str:
    """
    Map raw position string into one of 5 main positions: PG, SG, SF, PF, C.
    If the position is something like "PG-SG", use the first one.
    Return None if the position cannot be mapped.
    """
    if not isinstance(pos, str):
        return None
    pos = pos.strip().upper()

    # If already one of the main 5 positions, keep as is
    if pos in {"PG", "SG", "SF", "PF", "C"}:
        return pos

    # Handle hybrid positions like "PG-SG" by taking the first part
    if "-" in pos:
        first = pos.split("-")[0]
        if first in {"PG", "SG", "SF", "PF", "C"}:
            return first

    # Unknown or unmappable position
    return None


def map_pos_three_class(pos: str) -> str:
    """
    Map raw position string into 3 broad groups:
    - Guard: PG or SG
    - Wing: SF
    - Big: PF or C
    Return None if the position cannot be mapped.
    """
    p5 = map_pos_five_way(pos)
    if p5 is None:
        return None

    if p5 in {"PG", "SG"}:
        return "Guard"
    if p5 in {"SF"}:
        return "Wing"
    if p5 in {"PF", "C"}:
        return "Big"
    return None


print(f"[INFO] Loading: {DATA_PATH}")
# Load player stats from CSV
df = pd.read_csv(DATA_PATH)

# Extract season starting year from strings like "2022-23"
df["SeasonStart"] = df["Season"].str.split("-").str[0].astype(int)

# Build position label column based on configuration (3-class or 5-class)
if USE_THREE_CLASS:
    df["PosLabel"] = df["Pos"].apply(map_pos_three_class)
else:
    df["PosLabel"] = df["Pos"].apply(map_pos_five_way)

# Drop rows with unknown or invalid positions
before_pos = len(df)
df = df[~df["PosLabel"].isna()].copy()
after_pos = len(df)
print(f"[INFO] Dropped {before_pos - after_pos} rows with unknown / invalid positions.")
print(f"[INFO] Remaining rows after position filtering: {after_pos}")

# Filter out players with too few minutes (unstable / noisy stats)
before_mp = len(df)
df = df[df["MP"] >= MIN_MINUTES].copy()
after_mp = len(df)
print(f"[INFO] Dropped {before_mp - after_mp} rows with MP < {MIN_MINUTES}.")
print(f"[INFO] Remaining rows after MP filter: {after_mp}")

# Base counting and shooting stats used as raw features
base_cols = [
    "Age", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%",
    "ORB", "DRB", "TRB",
    "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
]

# Keep only columns that actually exist in the dataframe
base_cols = [c for c in base_cols if c in df.columns]

# Small epsilon to avoid division by zero in per-36 and ratio calculations
eps = 1e-6
# Work with minutes; replace 0 with NaN to avoid extreme per-36 values
mp = df["MP"].replace(0, np.nan)

# Per-36 stats for scoring, playmaking, rebounding, defense, and turnovers
df["PTS_per36"] = df["PTS"] / (mp + eps) * 36.0
df["AST_per36"] = df["AST"] / (mp + eps) * 36.0
df["TRB_per36"] = df["TRB"] / (mp + eps) * 36.0
df["STL_per36"] = df["STL"] / (mp + eps) * 36.0
df["BLK_per36"] = df["BLK"] / (mp + eps) * 36.0
df["TOV_per36"] = df["TOV"] / (mp + eps) * 36.0

# Per-36 shooting volume stats
df["FGA_per36"] = df["FGA"] / (mp + eps) * 36.0
df["FTA_per36"] = df["FTA"] / (mp + eps) * 36.0
df["3PA_per36"] = df["3PA"] / (mp + eps) * 36.0

# Rate and ratio-based features (shot mix, free throw rate, AST/TOV, usage proxy)
df["ThreePointRate"] = df["3PA"] / (df["FGA"] + eps)
df["FreeThrowRate"] = df["FTA"] / (df["FGA"] + eps)
df["AST_TOV_Ratio"] = df["AST"] / (df["TOV"] + eps)
df["UsageProxy"] = (df["FGA"] + 0.44 * df["FTA"]) / (mp + eps)

# Newly engineered feature columns
extra_cols = [
    "PTS_per36", "AST_per36", "TRB_per36", "STL_per36", "BLK_per36",
    "TOV_per36", "FGA_per36", "FTA_per36", "3PA_per36",
    "ThreePointRate", "FreeThrowRate", "AST_TOV_Ratio", "UsageProxy",
]

# Final feature set = basic box-score stats + engineered features
feature_cols = base_cols + extra_cols

# Fill any remaining NaNs with 0.0 for model input
df[feature_cols] = df[feature_cols].fillna(0.0)

# Feature matrix and raw string labels
X = df[feature_cols].values
y_raw = df["PosLabel"].values

print(f"[INFO] Final dataset shape: X={X.shape}, y={y_raw.shape}")

# Encode string position labels to integer classes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Print class distribution after all filters
unique, counts = np.unique(y_raw, return_counts=True)
print("[INFO] Position distribution (after filtering):")
for u, c in zip(unique, counts):
    print(f"  {u:>6s}: {c:5d} ({c / len(y_raw):.3f})")

# Define train seasons (<= 2022) and test seasons (>= 2023) based on SeasonStart
train_seasons = sorted(df[df["SeasonStart"] <= 2022]["SeasonStart"].unique())
test_seasons = sorted(df[df["SeasonStart"] >= 2023]["SeasonStart"].unique())

# Boolean masks for train/test split by season
train_mask = df["SeasonStart"].isin(train_seasons)
test_mask = df["SeasonStart"].isin(test_seasons)

# Split data into temporal train and test sets
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"[INFO] Train seasons: {train_seasons}")
print(f"[INFO] Test seasons:  {test_seasons}")
print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

print("[INFO] Training final model with best hyperparameters...")

# Pre-selected best hyperparameters from prior tuning
best_cfg = {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.1}

# Build pipeline: standardize features -> XGBoost multi-class classifier
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", xgb.XGBClassifier(
        objective="multi:softprob",                  # multi-class with softmax probabilities
        num_class=len(label_encoder.classes_),       # number of position classes
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        learning_rate=best_cfg["learning_rate"],
        subsample=0.9,                               # row subsampling for regularization
        colsample_bytree=0.9,                        # feature subsampling per tree
        reg_lambda=1.0,                              # L2 regularization
        reg_alpha=0.0,                               # L1 regularization
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,                                   # use all available CPU cores
    ))
])

# Train the pipeline on training data
final_model.fit(X_train, y_train)

print("\n[INFO] Evaluating on test set...")
# Predict class labels on the test set
y_test_pred = final_model.predict(X_test)
# Compute overall test accuracy
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n[RESULT] Test accuracy: {test_acc:.4f}")

print("\n[RESULT] Classification report (test):")
# Detailed per-class precision/recall/F1 using original class names
print(classification_report(
    y_test,
    y_test_pred,
    target_names=label_encoder.classes_
))

print("[RESULT] Confusion matrix (test):")
# Confusion matrix for multi-class predictions
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Compute simple baseline: always predict the majority class from training data
majority_class = np.bincount(y_train).argmax()
baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))
print(f"\n[BASELINE] Always predict '{label_encoder.inverse_transform([majority_class])[0]}' "
      f"accuracy: {baseline_acc:.4f}")