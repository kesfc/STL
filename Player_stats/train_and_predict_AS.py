import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Directory that contains the stats CSV
DATA_DIR = Path("all_stats")
# Full path to the player stats file (20 years of data)
CSV_PATH = DATA_DIR / "player_20years.csv"

print(f"[INFO] Loading: {CSV_PATH}")
# Load the raw player statistics
df = pd.read_csv(CSV_PATH)

# Replace missing award entries with empty string (so string ops work safely)
df["Awards"] = df["Awards"].fillna("")

# Create binary label: 1 if player has All-Star ("AS") in Awards, else 0
df["is_all_star"] = df["Awards"].str.contains("AS", case=False, na=False).astype(int)

# Filter out players who did not play any games in the season
df = df[df["G"] > 0].copy()

# Extract starting year of the season (e.g., "2019-20" -> 2019)
df["SeasonStart"] = df["Season"].str.split("-").str[0].astype(int)

# Get sorted list of all starting seasons
all_seasons = sorted(df["SeasonStart"].unique())
# Require at least 4 seasons to create a train/test split
if len(all_seasons) <= 3:
    raise RuntimeError("Not enough seasons to make a proper train/test split.")

# Use the last 3 seasons as test set, rest as training
test_seasons = all_seasons[-3:]
train_seasons = [s for s in all_seasons if s not in test_seasons]

# Split data into train and test by season
train_df = df[df["SeasonStart"].isin(train_seasons)].copy()
test_df = df[df["SeasonStart"].isin(test_seasons)].copy()

print(f"[INFO] Train seasons: {train_seasons}")
print(f"[INFO] Test seasons:  {test_seasons}")
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

# Target label column
label_col = "is_all_star"

# Candidate numeric feature columns
numeric_cols = [
    "Age", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%",
    "ORB", "DRB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "Trp-Dbl",
]

# Categorical feature columns (position and team)
categorical_cols = ["Pos", "Team"]

# Keep only numeric columns that actually exist in the dataframe
numeric_cols = [c for c in numeric_cols if c in df.columns]

# Drop rows with missing label or missing numeric features in train/test
train_df = train_df.dropna(subset=[label_col] + numeric_cols)
test_df = test_df.dropna(subset=[label_col] + numeric_cols)

# Build feature matrices and label vectors for train and test
X_train = train_df[numeric_cols + categorical_cols].copy()
y_train = train_df[label_col].values

X_test = test_df[numeric_cols + categorical_cols].copy()
y_test = test_df[label_col].values

from sklearn.preprocessing import OneHotEncoder

# Column-wise preprocessing:
# - Standardize numeric columns
# - One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Gradient Boosting classifier for binary classification (All-Star vs non All-Star)
model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

# Full pipeline: preprocessing + model
clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

print("[INFO] Training model...")
# Fit the model on training data
clf.fit(X_train, y_train)

# Predict on the held-out test seasons
y_pred = clf.predict(X_test)
# Compute test accuracy
acc = accuracy_score(y_test, y_pred)

print("\n[RESULT] Test accuracy: {:.4f}".format(acc))
print("[RESULT] Class distribution in test set:")
# Show class distribution (ratio) of labels in the test set
print(pd.Series(y_test).value_counts(normalize=True).rename("ratio"))

print("\n[RESULT] Classification report (test):")
# Print detailed precision/recall/F1 for each class
print(classification_report(y_test, y_pred, digits=4))