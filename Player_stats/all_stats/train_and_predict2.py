import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== 1. Config =====
DATA_PATH = "player_20years.csv"

# ===== 2. Load data =====
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# ===== 3. Basic cleaning =====

df = df.dropna(subset=["Pos"])
print("After dropping rows with missing Pos:", df.shape)


def clean_pos(pos):
    if isinstance(pos, str):
        return pos.split("-")[0].strip()
    return pos


df["Pos_clean"] = df["Pos"].apply(clean_pos)

VALID_POS = ["PG", "SG", "SF", "PF", "C"]
df = df[df["Pos_clean"].isin(VALID_POS)]
print("After keeping only 5 main positions:", df.shape)

# ===== 4. Select features (X) and target (y) =====

drop_cols = [
    "Rk",
    "Player",
    "Team",
    "Pos",
    "Pos_clean",
    "Awards",
    "Player-additional",
    "Season",
]

drop_cols = [c for c in drop_cols if c in df.columns]

candidate_cols = [c for c in df.columns if c not in drop_cols]

for c in candidate_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

y = df["Pos_clean"].copy()
X = df[candidate_cols].copy()

print("Base feature columns:", candidate_cols)

# ===== 5. Feature engineering: per-game / per-minute =====

eps = 1e-6

per_game_cols = [
    "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA",
    "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL",
    "BLK", "TOV", "PF", "PTS", "Trp-Dbl"
]
if "G" in X.columns:
    for col in per_game_cols:
        if col in X.columns:
            X[col + "_perG"] = X[col] / (X["G"] + eps)

if "MP" in X.columns:
    for col in per_game_cols:
        if col in X.columns:
            X[col + "_perMin"] = X[col] / (X["MP"] + eps)

print("After feature engineering, X shape:", X.shape)

print("NaN count per column BEFORE filling:")
print(X.isna().sum())

for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

print("NaN count per column AFTER filling:")
print(X.isna().sum().sum(), "NaNs total (should be 0)")

print("Position distribution:")
print(y.value_counts())

n_samples = X.shape[0]
print("Number of samples:", n_samples)
if n_samples == 0:
    raise RuntimeError("No samples left after cleaning. Check your data and script.")

# ===== 7. Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ===== 8. Logistic Regression (baseline) =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Training Logistic Regression (baseline) ===")
log_clf = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    n_jobs=-1
)
log_clf.fit(X_train_scaled, y_train)

y_pred_log = log_clf.predict(X_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log)
print("\n[LogReg] Test Accuracy:", acc_log)
print("\n[LogReg] Classification Report:")
print(classification_report(y_test, y_pred_log))

print("[LogReg] Confusion Matrix (rows = true, cols = pred):")
print(confusion_matrix(y_test, y_pred_log))

# ===== 9. RandomForest=====
print("\n=== Training RandomForestClassifier ===")
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print("\n[RF] Test Accuracy:", acc_rf)
print("\n[RF] Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("[RF] Confusion Matrix (rows = true, cols = pred):")
print(confusion_matrix(y_test, y_pred_rf))

# ===== 10. RF feature importance=====
importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\n[RF] Top 20 important features:")
print(feat_imp.head(20))
