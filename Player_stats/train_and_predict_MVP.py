import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Path to the CSV file containing 20 years of player statistics
DATA_PATH = Path("all_stats/player_20years.csv")
# Target minimum accuracy threshold (not directly used in code but kept for reference)
TARGET_MIN_ACCURACY = 0.80

print(f"[INFO] Loading: {DATA_PATH}")
# Load the full player stats dataset
df = pd.read_csv(DATA_PATH)

# Extract the ending year of each season (e.g., "2019-20" -> 20 -> 2020 as int)
df["SeasonEndYear"] = df["Season"].str.split("-").str[1].astype(int)

def has_allnba_or_mvp(awards_str: str) -> int:
    """
    Check if a player has All-NBA or MVP related awards in the given award string.
    Returns 1 if yes, 0 otherwise.
    """
    if not isinstance(awards_str, str):
        return 0
    s = awards_str.upper()
    # Mark as positive if the awards string contains "NBA" or "MVP"
    if "NBA" in s or "MVP" in s:
        return 1
    return 0

# Create binary label for whether a player received All-NBA or MVP related awards
df["LABEL_ALLNBA_MVP"] = df["Awards"].apply(has_allnba_or_mvp)

print("[INFO] Label distribution (overall):")
# Show overall positive/negative ratio for the label
print(df["LABEL_ALLNBA_MVP"].value_counts(normalize=True).rename("ratio"))

# Keep only players who actually played games (G > 0)
df = df[df["G"].fillna(0) > 0].copy()

# Collect and sort all unique season ending years
unique_end_years = sorted(df["SeasonEndYear"].unique())
# Need at least 4 distinct years to create a train/test split
if len(unique_end_years) <= 3:
    raise RuntimeError("Not enough seasons to create train/test split.")

# Use the last 3 end-years as test seasons; rest are used for training
test_end_years = unique_end_years[-3:]
train_end_years = [y for y in unique_end_years if y not in test_end_years]

# Split dataset into training and test sets based on season end year
train_df = df[df["SeasonEndYear"].isin(train_end_years)].copy()
test_df = df[df["SeasonEndYear"].isin(test_end_years)].copy()

print(f"[INFO] Train seasons: {train_end_years}")
print(f"[INFO] Test seasons:  {test_end_years}")
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

# Columns to exclude from numeric features (IDs, text columns, label, etc.)
exclude_cols = {
    "Rk",
    "Player",
    "Team",
    "Pos",
    "Awards",
    "Player-additional",
    "Season",
    "LABEL_ALLNBA_MVP",
}

# Automatically collect all numeric feature columns that are not excluded
numeric_cols = [
    c for c in train_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])
]

print(f"[INFO] Number of numeric feature columns: {len(numeric_cols)}")
print("[INFO] Example feature columns:", numeric_cols[:10])

# Build training features and labels
X_train = train_df[numeric_cols].fillna(0).values
y_train = train_df["LABEL_ALLNBA_MVP"].values.astype(int)

# Build test features and labels
X_test = test_df[numeric_cols].fillna(0).values
y_test = test_df["LABEL_ALLNBA_MVP"].values.astype(int)

# Standardize numeric features (mean=0, std=1) for XGBoost input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute positive/negative ratio in training data for class imbalance handling
pos_ratio = y_train.mean()
neg_ratio = 1.0 - pos_ratio
if pos_ratio == 0:
    # Safety check to avoid division by zero if no positive examples exist
    raise RuntimeError("No positive samples (All-NBA/MVP) in training data!")

# scale_pos_weight balances the loss between positive and negative classes
scale_pos_weight = neg_ratio / pos_ratio
print(f"[INFO] Positive rate in train: {pos_ratio:.4f}, scale_pos_weight={scale_pos_weight:.2f}")

print("[INFO] Training model...")

# Configure XGBoost classifier for binary classification
model = XGBClassifier(
    n_estimators=400,          # number of boosting trees
    max_depth=5,              # maximum tree depth
    learning_rate=0.05,       # shrinkage (eta)
    subsample=0.8,            # row subsampling ratio
    colsample_bytree=0.8,     # feature subsampling ratio per tree
    objective="binary:logistic",  # binary classification with logistic output
    eval_metric="logloss",    # evaluation metric for training
    random_state=42,          # reproducibility
    n_jobs=-1,                # use all available CPU cores
    scale_pos_weight=scale_pos_weight,  # handle class imbalance
)

# Train the XGBoost model on scaled training features
model.fit(X_train_scaled, y_train)

# Predict probabilities for the positive class on the test set
y_proba = model.predict_proba(X_test_scaled)[:, 1]
# Convert probabilities to binary predictions with threshold 0.5
y_pred = (y_proba >= 0.5).astype(int)

# Compute test accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test accuracy: {acc:.4f}")

print("\n[RESULT] Class distribution in test set:")
# Show class distribution in the test set
print(test_df["LABEL_ALLNBA_MVP"].value_counts(normalize=True).rename("ratio"))

print("\n[RESULT] Classification report (test):")
# Print precision, recall, F1-score for each class
print(classification_report(y_test, y_pred, digits=4))

print("[RESULT] Confusion matrix (test):")
# Show confusion matrix for test predictions
print(confusion_matrix(y_test, y_pred))