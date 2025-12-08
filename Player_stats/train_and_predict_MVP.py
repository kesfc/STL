import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("all_stats/player_20years.csv")
TARGET_MIN_ACCURACY = 0.80

print(f"[INFO] Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

df["SeasonEndYear"] = df["Season"].str.split("-").str[1].astype(int)

def has_allnba_or_mvp(awards_str: str) -> int:
    if not isinstance(awards_str, str):
        return 0
    s = awards_str.upper()
    if "NBA" in s or "MVP" in s:
        return 1
    return 0

df["LABEL_ALLNBA_MVP"] = df["Awards"].apply(has_allnba_or_mvp)

print("[INFO] Label distribution (overall):")
print(df["LABEL_ALLNBA_MVP"].value_counts(normalize=True).rename("ratio"))

df = df[df["G"].fillna(0) > 0].copy()

unique_end_years = sorted(df["SeasonEndYear"].unique())
if len(unique_end_years) <= 3:
    raise RuntimeError("Not enough seasons to create train/test split.")

test_end_years = unique_end_years[-3:]
train_end_years = [y for y in unique_end_years if y not in test_end_years]

train_df = df[df["SeasonEndYear"].isin(train_end_years)].copy()
test_df = df[df["SeasonEndYear"].isin(test_end_years)].copy()

print(f"[INFO] Train seasons: {train_end_years}")
print(f"[INFO] Test seasons:  {test_end_years}")
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

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

numeric_cols = [
    c for c in train_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])
]

print(f"[INFO] Number of numeric feature columns: {len(numeric_cols)}")
print("[INFO] Example feature columns:", numeric_cols[:10])

X_train = train_df[numeric_cols].fillna(0).values
y_train = train_df["LABEL_ALLNBA_MVP"].values.astype(int)

X_test = test_df[numeric_cols].fillna(0).values
y_test = test_df["LABEL_ALLNBA_MVP"].values.astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pos_ratio = y_train.mean()
neg_ratio = 1.0 - pos_ratio
if pos_ratio == 0:
    raise RuntimeError("No positive samples (All-NBA/MVP) in training data!")

scale_pos_weight = neg_ratio / pos_ratio
print(f"[INFO] Positive rate in train: {pos_ratio:.4f}, scale_pos_weight={scale_pos_weight:.2f}")

print("[INFO] Training model...")

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test accuracy: {acc:.4f}")

print("\n[RESULT] Class distribution in test set:")
print(test_df["LABEL_ALLNBA_MVP"].value_counts(normalize=True).rename("ratio"))

print("\n[RESULT] Classification report (test):")
print(classification_report(y_test, y_pred, digits=4))

print("[RESULT] Confusion matrix (test):")
print(confusion_matrix(y_test, y_pred))