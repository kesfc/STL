import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Paths to player-level and team-level data
PLAYER_CSV = "Player_stats/all_stats/player_20years.csv"
TEAM_CSV = "Team_stats/all_seasons_team_summary.csv"


def build_team_level_dataframe():
    """
    Build a team-level dataframe by aggregating player stats
    and merging them with official team-level statistics.
    """
    print(f"[INFO] Loading player data from: {PLAYER_CSV}")
    player_df = pd.read_csv(PLAYER_CSV)

    # Remove "TOT" rows (players who played for multiple teams combined entry)
    if "Team" in player_df.columns:
        tot_count = (player_df["Team"] == "TOT").sum()
        print(f"[INFO] Dropped {tot_count} 'TOT' rows.")
        player_df = player_df[player_df["Team"] != "TOT"]

    # Filter out players with too few minutes (unstable contribution)
    if "MP" not in player_df.columns:
        raise ValueError("Player CSV must contain 'MP' column.")
    before_mp = len(player_df)
    player_df = player_df[player_df["MP"] >= 200]
    after_mp = len(player_df)
    print(f"[INFO] Dropped {before_mp - after_mp} rows with MP < 200.")
    print(f"[INFO] Remaining player-rows: {after_mp}")

    # Derive the season end year if not already present
    if "SeasonEndYear" not in player_df.columns:
        if "Season" not in player_df.columns:
            raise ValueError("Player CSV must contain 'Season' column.")
        player_df["SeasonEndYear"] = (
            player_df["Season"].astype(str).str.split("-").str[-1].astype(int)
        )

    # Keys used to aggregate player rows to team-level per season
    group_cols = ["Season", "SeasonEndYear", "Team"]

    # Player stats to sum when forming team-level totals
    sum_cols = [
        "G", "GS", "MP",
        "FG", "FGA", "3P", "3PA", "2P", "2PA",
        "FT", "FTA",
        "ORB", "DRB", "TRB",
        "AST", "STL", "BLK",
        "TOV", "PF",
        "PTS", "Trp-Dbl",
    ]
    # Keep only columns that actually exist
    sum_cols = [c for c in sum_cols if c in player_df.columns]

    # Define aggregation rules: sum for stats, mean for Age (if available)
    agg_dict = {col: "sum" for col in sum_cols}
    if "Age" in player_df.columns:
        agg_dict["Age"] = "mean"

    # Aggregate player stats into team-season-level rows
    team_from_players = (
        player_df
        .groupby(group_cols)
        .agg(agg_dict)
        .reset_index()
    )

    team_rows = len(team_from_players)
    print(f"[INFO] Team-level rows from players: {team_rows}")

    # Small epsilon to avoid division by zero in percentage calculations
    eps = 1e-8

    # Compute team-level shooting and efficiency metrics from aggregated player stats
    if {"FG", "FGA"}.issubset(team_from_players.columns):
        team_from_players["FG_PCT_P"] = team_from_players["FG"] / (team_from_players["FGA"] + eps)
    if {"3P", "3PA"}.issubset(team_from_players.columns):
        team_from_players["TP_PCT_P"] = team_from_players["3P"] / (team_from_players["3PA"] + eps)
    if {"2P", "2PA"}.issubset(team_from_players.columns):
        team_from_players["TP2_PCT_P"] = team_from_players["2P"] / (team_from_players["2PA"] + eps)
    if {"FT", "FTA"}.issubset(team_from_players.columns):
        team_from_players["FT_PCT_P"] = team_from_players["FT"] / (team_from_players["FTA"] + eps)
    if {"PTS", "FGA", "FTA"}.issubset(team_from_players.columns):
        # True shooting percentage based on player-aggregated totals
        team_from_players["TS_PCT_P"] = team_from_players["PTS"] / (
            2 * (team_from_players["FGA"] + 0.44 * team_from_players["FTA"] + eps)
        )
    if {"FG", "3P", "FGA"}.issubset(team_from_players.columns):
        # Effective field goal percentage from player-aggregated stats
        team_from_players["EFG_PCT_P"] = (
            team_from_players["FG"] + 0.5 * team_from_players["3P"]
        ) / (team_from_players["FGA"] + eps)
    if {"AST", "TOV"}.issubset(team_from_players.columns):
        # Assist-to-turnover ratio from player-aggregated stats
        team_from_players["AST_TO_RATIO_P"] = (
            team_from_players["AST"] / (team_from_players["TOV"] + eps)
        )

    print(f"[INFO] Loading team data from: {TEAM_CSV}")
    team_df = pd.read_csv(TEAM_CSV)

    # Derive SeasonEndYear for team data if needed
    if "SeasonEndYear" not in team_df.columns:
        if "Season" not in team_df.columns:
            raise ValueError("Team CSV must contain 'Season' and 'SeasonEndYear' or derivable info.")
        team_df["SeasonEndYear"] = (
            team_df["Season"].astype(str).str.split("-").str[-1].astype(int)
        )

    # Keep only the key team-level columns and performance stats
    keep_team_cols = [
        "Season", "SeasonEndYear", "Team",
        "WINS", "LOSSES",
        "FG", "FGA", "3P", "3PA", "2P", "2PA",
        "FT", "FTA",
        "ORB", "DRB", "TRB",
        "AST", "STL", "BLK",
        "TOV", "PF", "PTS",
        "FG%", "3P%", "2P%", "FT%", "eFG%",
    ]
    keep_team_cols = [c for c in keep_team_cols if c in team_df.columns]

    # Create a smaller team dataframe with only needed columns
    team_df_small = team_df[keep_team_cols].copy()

    # Prefix team-level numeric/stat columns with "T_" to distinguish them
    rename_map = {}
    for c in keep_team_cols:
        if c in ["Season", "SeasonEndYear", "Team", "WINS", "LOSSES"]:
            continue
        rename_map[c] = f"T_{c}"

    team_df_small = team_df_small.rename(columns=rename_map)

    # Merge player-aggregated team features with official team stats
    merged = pd.merge(
        team_from_players,
        team_df_small,
        on=["Season", "SeasonEndYear", "Team"],
        how="inner",
    )

    print(f"[INFO] Merged team-feature rows: {len(merged)}")

    # Compute team win percentage and previous season's win percentage
    merged["WIN_PCT"] = merged["WINS"] / (merged["WINS"] + merged["LOSSES"])
    merged = merged.sort_values(["Team", "SeasonEndYear"])
    merged["PREV_WIN_PCT"] = merged.groupby("Team")["WIN_PCT"].shift(1)

    return merged


def create_trend_features(df):
    """
    Add win-percentage trend-based features:
    - current season win pct
    - previous season win pct
    - rolling 3-year and 5-year averages
    - short-term trend (change in 3-year rolling average)
    """
    df = df.copy()
    df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
    df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)
    df = df.sort_values(["Team", "SeasonEndYear"])

    # Rolling 3-year average win percentage per team
    df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    # Rolling 5-year average win percentage per team
    df["WIN_PCT_5Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    # Trend: change in 3-year rolling mean over time
    df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean().diff()
    )

    # List of trend-based feature columns used for modeling
    feature_cols = ["PREV_WIN_PCT", "WIN_PCT_3Y_AVG", "WIN_PCT_5Y_AVG", "WIN_PCT_TREND"]
    return df, feature_cols


def evaluate_predictions(df, true_col="WIN_PCT", pred_col="PRED_SCORE"):
    """
    Evaluate predictions using ranking-based metrics and regression metrics.
    - Ranking metrics: exact rank match, within 1 or 2 places, combined overall score
    - Regression metrics: MAE, RMSE, R^2, Pearson r, Spearman rho
    """
    df = df.copy()
    # Rank teams by true and predicted values (higher is better)
    df = df.sort_values(true_col, ascending=False)
    df["TRUE_RANK"] = df[true_col].rank(method="min", ascending=False).astype(int)
    df["PRED_RANK"] = df[pred_col].rank(method="min", ascending=False).astype(int)
    df["RANK_DIFF"] = df["PRED_RANK"] - df["TRUE_RANK"]

    # Accuracy metrics based on rank differences
    exact_acc = (df["PRED_RANK"] == df["TRUE_RANK"]).mean()
    within1_acc = (df["RANK_DIFF"].abs() <= 1).mean()
    within2_acc = (df["RANK_DIFF"].abs() <= 2).mean()
    # Weighted overall score emphasizing within-1 accuracy
    overall_score = within1_acc * 0.7 + exact_acc * 0.3

    # Extract arrays for regression-style metrics
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    # Standard regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Correlation metrics
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    return {
        "exact_rank_acc": exact_acc,
        "within1_rank_acc": within1_acc,
        "within2_rank_acc": within2_acc,
        "overall_score": overall_score,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }


def main():
    """
    Main pipeline:
    1) Build merged team-level dataset from player and team stats.
    2) Create win-percentage trend features.
    3) Train a Random Forest regressor on historical seasons.
    4) Evaluate predictions on the latest two seasons (holdout).
    """
    data = build_team_level_dataframe()

    # Inspect numeric columns available in the merged dataset
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[INFO] Number of numeric feature columns (initial merged df): {len(numeric_cols)}")
    print(f"[INFO] Example numeric columns: {numeric_cols[:10]}")

    # Add trend-based features for modeling
    full_feat, feature_cols = create_trend_features(data)

    # Define holdout seasons as the last two seasons in the data
    seasons_sorted = sorted(full_feat["Season"].unique())
    latest_season = seasons_sorted[-1]
    second_latest_season = seasons_sorted[-2] if len(seasons_sorted) > 1 else latest_season
    holdout_seasons = [second_latest_season, latest_season]

    print(f"[INFO] Holdout seasons: {holdout_seasons}")
    print(f"[INFO] All seasons: {seasons_sorted}")

    # Training on all seasons except the last two
    train_feat = full_feat[~full_feat["Season"].isin(holdout_seasons)].copy()
    # Test on the last two seasons
    test_feat_all = full_feat[full_feat["Season"].isin(holdout_seasons)].copy()

    # Features and target for training
    X_train = train_feat[feature_cols].values
    y_train = train_feat["WIN_PCT"].values

    # Standardize feature scales for the regression model
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Random Forest regressor to predict win percentage
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS (per holdout season)")
    print("=" * 80)

    # Evaluate predictions separately for each holdout season
    for season in holdout_seasons:
        print("\n" + "=" * 50)
        print(f"SEASON: {season}")
        print("=" * 50)

        season_feat = test_feat_all[test_feat_all["Season"] == season].copy()
        if season_feat.empty:
            continue

        # Ensure all trend feature columns exist in the season subset
        for col in feature_cols:
            if col not in season_feat.columns:
                season_feat[col] = 0.0

        X_test = season_feat[feature_cols].values
        X_test_scaled = preprocessor.transform(X_test)

        # Predict win percentage for each team in the holdout season
        pred = model.predict(X_test_scaled)
        season_feat["PRED_SCORE"] = pred

        # Compute metrics comparing predicted vs true win percentage
        metrics = evaluate_predictions(season_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")

        print("\nPerformance Metrics:")
        print(f"  MAE (WIN_PCT):   {metrics['mae']:.4f}")
        print(f"  RMSE (WIN_PCT):  {metrics['rmse']:.4f}")
        print(f"  R^2:             {metrics['r2']:.4f}")
        print(f"  Pearson r:       {metrics['pearson_r']:.4f}")
        print(f"  Spearman rho:    {metrics['spearman_r']:.4f}")


if __name__ == "__main__":
    main()