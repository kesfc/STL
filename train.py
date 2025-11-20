import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


summary_root = os.path.join("Team_stats", "summary")

CURR_SEASON = "2025-2026"          
PREV_SEASON_END = 2025            
REG_SEASON_GAMES = 82             

all_files = sorted(glob.glob(os.path.join(summary_root, "*.csv")))
print("Found summary files:", all_files)

train_rows = []
curr_df = None

for path in all_files:
    season = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)

    if "Season" not in df.columns:
        df["Season"] = season
    if "SeasonEndYear" not in df.columns:
        df["SeasonEndYear"] = int(season.split("-")[1])

    if season == CURR_SEASON:
        print(f"Use {path} as CURRENT season data")
        curr_df = df.copy()
    else:
        train_rows.append(df)

if curr_df is None:
    raise RuntimeError(f"No summary file found for current season {CURR_SEASON}")

train = pd.concat(train_rows, ignore_index=True)
train = train.dropna(subset=["WINS", "LOSSES"])
print("Train shape:", train.shape)

train["WIN_PCT"] = train["WINS"] / (train["WINS"] + train["LOSSES"])

def create_features(df, is_training=True, train_df=None):
    """
    创建特征，避免数据泄露
    """
    df = df.copy()
    
    if "WIN_PCT" not in df.columns and "WINS" in df.columns and "LOSSES" in df.columns:
        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
    
    base_cols = [
        "FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA",
        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
    ]
    
    df["OFF_RTG"] = df["PTS"] / (df["FGA"] - df["ORB"] + df["TOV"] + 0.4 * df["FTA"] + 1) * 100
    df["REB_RATE"] = df["TRB"] / (df["FGA"] - df["FG"] + 1)
    df["AST_TO_RATIO"] = df["AST"] / (df["TOV"] + 1)
    df["EFG_PCT"] = (df["FG"] + 0.5 * df["3P"]) / df["FGA"]
    
    df["TSA"] = df["FGA"] + 0.44 * df["FTA"]
    df["TS_PCT"] = df["PTS"] / (2 * df["TSA"])
    
    if is_training:
        df = df.sort_values(["Team", "SeasonEndYear"])
        
        for col in base_cols:
            df[f"{col}_REL"] = df.groupby("Team")[col].transform(
                lambda x: (x - x.expanding().mean()) / (x.expanding().std() + 1e-8)
            )
            
            df[f"{col}_TREND"] = df.groupby("Team")[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean().pct_change()
            )
    
    else:
        if train_df is not None:
            df = df.sort_values(["Team", "SeasonEndYear"])
            train_df_sorted = train_df.sort_values(["Team", "SeasonEndYear"])
            
            for col in base_cols:
                team_means = train_df_sorted.groupby("Team")[col].mean()
                team_stds = train_df_sorted.groupby("Team")[col].std()
                
                df = df.merge(team_means.rename(f"{col}_MEAN"), on="Team", how="left")
                df = df.merge(team_stds.rename(f"{col}_STD"), on="Team", how="left")
                
                df[f"{col}_REL"] = (df[col] - df[f"{col}_MEAN"]) / (df[f"{col}_STD"] + 1e-8)
                
                df[f"{col}_TREND"] = df.groupby("Team")[col].transform(
                    lambda x: x.pct_change().rolling(2, min_periods=1).mean()
                )
    
    df = df.sort_values(["Team", "SeasonEndYear"])
    df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)
    
    df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean().pct_change()
    )
    
    df["WIN_PCT_STD"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).std()
    )
    
    if is_training:
        global_mean_win_pct = df["WIN_PCT"].mean()
    else:
        global_mean_win_pct = train_df["WIN_PCT"].mean() if train_df is not None else 0.5
        
    df["PREV_WIN_PCT"] = df["PREV_WIN_PCT"].fillna(global_mean_win_pct)
    df["WIN_PCT_TREND"] = df["WIN_PCT_TREND"].fillna(0)
    df["WIN_PCT_STD"] = df["WIN_PCT_STD"].fillna(0.1)
    
    feature_cols = [
        "SeasonEndYear", "PREV_WIN_PCT", "WIN_PCT_TREND", "WIN_PCT_STD",
        "OFF_RTG", "REB_RATE", "AST_TO_RATIO", "EFG_PCT", "TS_PCT"
    ]
    
    rel_cols = [f"{col}_REL" for col in base_cols if f"{col}_REL" in df.columns]
    trend_cols = [f"{col}_TREND" for col in base_cols if f"{col}_TREND" in df.columns]
    
    feature_cols.extend(rel_cols)
    feature_cols.extend(trend_cols)
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].fillna(0)
    
    return df, feature_cols


print("Creating features for training data...")
train, feature_cols = create_features(train, is_training=True)


print(f"Using {len(feature_cols)} features:")
for col in feature_cols[:15]:  
    print(f"  - {col}")

X_train = train[feature_cols].values
y_train = train["WIN_PCT"].values

print("X_train shape:", X_train.shape)


def time_series_validation(X, y, seasons, n_splits=3):
    """
    时间序列交叉验证
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # 确保验证集的时间在训练集之后
        if max(seasons[train_idx]) >= min(seasons[val_idx]):
            print(f"  Skipping fold {fold+1} due to time leakage")
            continue
            
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mae = mean_absolute_error(y_val_fold, y_pred)
        maes.append(mae)
        print(f"  Fold {fold+1} MAE: {mae:.4f} (train: {len(train_idx)}, val: {len(val_idx)})")
    
    return np.mean(maes) if maes else None


print("Performing time series validation...")
seasons = train["SeasonEndYear"].values
ts_mae = time_series_validation(X_train, y_train, seasons)
if ts_mae:
    print(f"Time Series CV MAE: {ts_mae:.4f}")


print("Training final model...")
final_model = HistGradientBoostingRegressor(
    max_depth=5,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

final_model.fit(X_train, y_train)
print("Model trained.")

VALID_FROM_YEAR = 2020
valid_df = train[train["SeasonEndYear"] >= VALID_FROM_YEAR].copy()
if not valid_df.empty:
    X_valid = valid_df[feature_cols].values
    y_valid = valid_df["WIN_PCT"].values
    y_pred = final_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    print(f"Validation MAE on seasons >= {VALID_FROM_YEAR}: {mae:.4f}")


print("Creating features for current season...")
curr, curr_feature_cols = create_features(curr_df, is_training=False, train_df=train)

missing_features = set(feature_cols) - set(curr.columns)
extra_features = set(curr.columns) - set(feature_cols)

if missing_features:
    print(f"Warning: Missing {len(missing_features)} features in current data")
    for feature in missing_features:
        curr[feature] = 0

if extra_features:
    print(f"Info: {len(extra_features)} extra features will be ignored")

X_curr = curr[feature_cols].values

curr["PRED_WIN_PCT"] = final_model.predict(X_curr)

curr["PRED_WIN_PCT"] = curr["PRED_WIN_PCT"].clip(0.1, 0.9)
curr["PRED_WINS"] = curr["PRED_WIN_PCT"] * REG_SEASON_GAMES

curr = curr.sort_values("PRED_WINS", ascending=False).reset_index(drop=True)
curr["PRED_RANK"] = curr.index + 1

print("\nTop 10 predicted teams:")
print(curr[["PRED_RANK", "Team", "PRED_WINS", "PRED_WIN_PCT"]].head(10))

pred_dir = os.path.join("Team_stats", "predictions")
os.makedirs(pred_dir, exist_ok=True)
out_pred_path = os.path.join(pred_dir, f"{CURR_SEASON}_pred_improved.csv")
curr.to_csv(out_pred_path, index=False)
print(f"Prediction saved to {out_pred_path}")

if hasattr(final_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))