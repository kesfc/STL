import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Directory containing per-season team W/L CSV files (e.g., "2015-2016.csv")
WL_DIR = Path("Team_stats") / "WL"


def load_wl_data():
    # Collect rows from all season CSVs
    all_rows = []

    # Get all CSV files in WL_DIR, sorted by filename (chronological if named like "2015-2016.csv")
    csv_files = sorted(WL_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in directory: {WL_DIR}")

    for csv_path in csv_files:
        # Use filename (without extension) as season string, e.g., "2015-2016"
        season_str = csv_path.stem

        # Load a single season of team W/L data
        df_season = pd.read_csv(csv_path)

        # Ensure required columns are present
        required_cols = {"TEAM_ABBR", "WINS", "LOSSES"}
        if not required_cols.issubset(df_season.columns):
            raise ValueError(f"File {csv_path} must contain columns: {required_cols}")

        # Keep only team abbreviation and W/L columns, add season label
        df_season = df_season[["TEAM_ABBR", "WINS", "LOSSES"]].copy()
        df_season["Season"] = season_str

        # Append this season's rows to the list
        all_rows.append(df_season)

    # Concatenate all seasons into one DataFrame
    df = pd.concat(all_rows, ignore_index=True)

    # Use a consistent team column name
    df = df.rename(columns={"TEAM_ABBR": "Team"})

    # Extract season start year (e.g., "2015-2016" -> 2015) for sorting and splitting
    df["SeasonStartYear"] = df["Season"].str.split("-").str[0].astype(int)

    # Compute team win percentage for each season
    df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])

    print(f"[INFO] Loaded {df['Season'].nunique()} seasons, {len(df)} team-season rows.")
    return df


def create_trend_features(df):
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()
    # Sort by team and season so rolling/shift operations are time-ordered
    df = df.sort_values(["Team", "SeasonStartYear"])

    # Previous season's win percentage for each team
    df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

    # Rolling 3-year average win percentage per team
    df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Rolling 5-year average win percentage per team
    df["WIN_PCT_5Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Short-term trend: change in the 3-year rolling average over time
    df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean().diff()
    )

    # List of trend-based features used as model inputs
    feature_cols = ["PREV_WIN_PCT", "WIN_PCT_3Y_AVG", "WIN_PCT_5Y_AVG", "WIN_PCT_TREND"]
    return df, feature_cols


def evaluate_per_season(season_df, true_col="WIN_PCT", pred_col="PRED_WIN_PCT"):
    # Extract true and predicted values as numpy arrays
    y_true = season_df[true_col].values
    y_pred = season_df[pred_col].values

    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Linear (Pearson) and rank-based (Spearman) correlation coefficients
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }


def main():
    # Load multi-season W/L data for all teams
    df = load_wl_data()

    # Create historical trend features from win percentages
    df_feat, feature_cols = create_trend_features(df)

    # Drop rows which have no previous-season info (first year for each team)
    df_feat = df_feat.dropna(subset=["PREV_WIN_PCT"]).copy()

    # Get sorted list of all season start years
    seasons_sorted = sorted(df_feat["SeasonStartYear"].unique())
    if len(seasons_sorted) < 4:
        # Need enough seasons to meaningfully split into train/test (e.g., 18 train + 2 test)
        raise RuntimeError("Need at least 4 seasons to have 18 train + 2 test in general.")

    # Use the last 2 seasons as holdout/test, the rest as training
    holdout_years = seasons_sorted[-2:]
    train_years = [y for y in seasons_sorted if y not in holdout_years]

    print(f"[INFO] All seasons (start years): {seasons_sorted}")
    print(f"[INFO] Train seasons (start years): {train_years}")
    print(f"[INFO] Holdout seasons (start years): {holdout_years}")

    # Split into train and test sets based on SeasonStartYear
    train_df = df_feat[df_feat["SeasonStartYear"].isin(train_years)].copy()
    test_df = df_feat[df_feat["SeasonStartYear"].isin(holdout_years)].copy()

    print(f"[INFO] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Feature matrix and target for training
    X_train = train_df[feature_cols].values
    y_train = train_df["WIN_PCT"].values

    # Feature matrix and target for testing
    X_test = test_df[feature_cols].values
    y_test = test_df["WIN_PCT"].values

    # Standardize features before feeding into linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simple baseline model: linear regression on trend features
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict win percentage for holdout seasons
    y_test_pred = model.predict(X_test_scaled)
    test_df["PRED_WIN_PCT"] = y_test_pred

    print("\n" + "=" * 80)
    print("BASELINE PREDICTIONS (using only W/L history, first 18 years -> last 2 years)")
    print("=" * 80)

    # Evaluate metrics separately for each holdout (test) season
    for year in holdout_years:
        season_mask = test_df["SeasonStartYear"] == year
        season_df = test_df[season_mask].copy()
        if season_df.empty:
            continue

        metrics = evaluate_per_season(season_df, true_col="WIN_PCT", pred_col="PRED_WIN_PCT")

        print("\n" + "-" * 60)
        print(f"Season starting {year}")
        print("-" * 60)
        print("Metrics:")
        print(f"  MAE (WIN_PCT):   {metrics['mae']:.4f}")
        print(f"  RMSE (WIN_PCT):  {metrics['rmse']:.4f}")
        print(f"  R^2:             {metrics['r2']:.4f}")
        print(f"  Pearson r:       {metrics['pearson_r']:.4f}")
        print(f"  Spearman rho:    {metrics['spearman_r']:.4f}")


if __name__ == "__main__":
    main()