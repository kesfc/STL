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

DATA_DIR = Path("all_stats")
CSV_PATH = DATA_DIR / "player_20years.csv"

print(f"[INFO] Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

df["Awards"] = df["Awards"].fillna("")

df["is_all_star"] = df["Awards"].str.contains("AS", case=False, na=False).astype(int)

df = df[df["G"] > 0].copy()

df["SeasonStart"] = df["Season"].str.split("-").str[0].astype(int)

all_seasons = sorted(df["SeasonStart"].unique())
if len(all_seasons) <= 3:
    raise RuntimeError("Not enough seasons to make a proper train/test split.")

test_seasons = all_seasons[-3:]
train_seasons = [s for s in all_seasons if s not in test_seasons]

train_df = df[df["SeasonStart"].isin(train_seasons)].copy()
test_df = df[df["SeasonStart"].isin(test_seasons)].copy()

print(f"[INFO] Train seasons: {train_seasons}")
print(f"[INFO] Test seasons:  {test_seasons}")
print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

label_col = "is_all_star"

numeric_cols = [
    "Age", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%",
    "ORB", "DRB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "Trp-Dbl",
]

categorical_cols = ["Pos", "Team"]

numeric_cols = [c for c in numeric_cols if c in df.columns]

train_df = train_df.dropna(subset=[label_col] + numeric_cols)
test_df = test_df.dropna(subset=[label_col] + numeric_cols)

X_train = train_df[numeric_cols + categorical_cols].copy()
y_train = train_df[label_col].values

X_test = test_df[numeric_cols + categorical_cols].copy()
y_test = test_df[label_col].values

from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

print("[INFO] Training model...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n[RESULT] Test accuracy: {:.4f}".format(acc))
print("[RESULT] Class distribution in test set:")
print(pd.Series(y_test).value_counts(normalize=True).rename("ratio"))

print("\n[RESULT] Classification report (test):")
print(classification_report(y_test, y_pred, digits=4))