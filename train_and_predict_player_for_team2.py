import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

PLAYER_CSV = "Player_stats/all_stats/player_20years.csv"
TEAM_CSV = "Team_stats/all_seasons_team_summary.csv"


def build_team_level_dataframe():
    print(f"[INFO] Loading player data from: {PLAYER_CSV}")
    player_df = pd.read_csv(PLAYER_CSV)

    if "Team" in player_df.columns:
        tot_count = (player_df["Team"] == "TOT").sum()
        print(f"[INFO] Dropped {tot_count} 'TOT' rows.")
        player_df = player_df[player_df["Team"] != "TOT"]

    if "MP" not in player_df.columns:
        raise ValueError("Player CSV must contain 'MP' column.")
    before_mp = len(player_df)
    player_df = player_df[player_df["MP"] >= 200]
    after_mp = len(player_df)
    print(f"[INFO] Dropped {before_mp - after_mp} rows with MP < 200.")
    print(f"[INFO] Remaining player-rows: {after_mp}")

    if "SeasonEndYear" not in player_df.columns:
        if "Season" not in player_df.columns:
            raise ValueError("Player CSV must contain 'Season' column.")
        player_df["SeasonEndYear"] = (
            player_df["Season"].astype(str).str.split("-").str[-1].astype(int)
        )

    group_cols = ["Season", "SeasonEndYear", "Team"]

    sum_cols = [
        "G", "GS", "MP",
        "FG", "FGA", "3P", "3PA", "2P", "2PA",
        "FT", "FTA",
        "ORB", "DRB", "TRB",
        "AST", "STL", "BLK",
        "TOV", "PF",
        "PTS", "Trp-Dbl",
    ]
    sum_cols = [c for c in sum_cols if c in player_df.columns]

    agg_dict = {col: "sum" for col in sum_cols}
    if "Age" in player_df.columns:
        agg_dict["Age"] = "mean"

    team_from_players = (
        player_df
        .groupby(group_cols)
        .agg(agg_dict)
        .reset_index()
    )

    team_rows = len(team_from_players)
    print(f"[INFO] Team-level rows from players: {team_rows}")

    eps = 1e-8
    if {"FG", "FGA"}.issubset(team_from_players.columns):
        team_from_players["FG_PCT_P"] = team_from_players["FG"] / (team_from_players["FGA"] + eps)
    if {"3P", "3PA"}.issubset(team_from_players.columns):
        team_from_players["TP_PCT_P"] = team_from_players["3P"] / (team_from_players["3PA"] + eps)
    if {"2P", "2PA"}.issubset(team_from_players.columns):
        team_from_players["TP2_PCT_P"] = team_from_players["2P"] / (team_from_players["2PA"] + eps)
    if {"FT", "FTA"}.issubset(team_from_players.columns):
        team_from_players["FT_PCT_P"] = team_from_players["FT"] / (team_from_players["FTA"] + eps)
    if {"PTS", "FGA", "FTA"}.issubset(team_from_players.columns):
        team_from_players["TS_PCT_P"] = team_from_players["PTS"] / (
            2 * (team_from_players["FGA"] + 0.44 * team_from_players["FTA"] + eps)
        )
    if {"FG", "3P", "FGA"}.issubset(team_from_players.columns):
        team_from_players["EFG_PCT_P"] = (
            team_from_players["FG"] + 0.5 * team_from_players["3P"]
        ) / (team_from_players["FGA"] + eps)
    if {"AST", "TOV"}.issubset(team_from_players.columns):
        team_from_players["AST_TO_RATIO_P"] = (
            team_from_players["AST"] / (team_from_players["TOV"] + eps)
        )

    print(f"[INFO] Loading team data from: {TEAM_CSV}")
    team_df = pd.read_csv(TEAM_CSV)

    if "SeasonEndYear" not in team_df.columns:
        if "Season" not in team_df.columns:
            raise ValueError("Team CSV must contain 'Season' and 'SeasonEndYear' or derivable info.")
        team_df["SeasonEndYear"] = (
            team_df["Season"].astype(str).str.split("-").str[-1].astype(int)
        )

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

    team_df_small = team_df[keep_team_cols].copy()
    rename_map = {}
    for c in keep_team_cols:
        if c in ["Season", "SeasonEndYear", "Team", "WINS", "LOSSES"]:
            continue
        rename_map[c] = f"T_{c}"

    team_df_small = team_df_small.rename(columns=rename_map)

    merged = pd.merge(
        team_from_players,
        team_df_small,
        on=["Season", "SeasonEndYear", "Team"],
        how="inner",
    )

    print(f"[INFO] Merged team-feature rows: {len(merged)}")

    merged["WIN_PCT"] = merged["WINS"] / (merged["WINS"] + merged["LOSSES"])
    merged = merged.sort_values(["Team", "SeasonEndYear"])
    merged["PREV_WIN_PCT"] = merged.groupby("Team")["WIN_PCT"].shift(1)

    return merged


def create_trend_features(df):
    df = df.copy()
    df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
    df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)
    df = df.sort_values(["Team", "SeasonEndYear"])

    df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["WIN_PCT_5Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean().diff()
    )

    feature_cols = ["PREV_WIN_PCT", "WIN_PCT_3Y_AVG", "WIN_PCT_5Y_AVG", "WIN_PCT_TREND"]
    return df, feature_cols


def evaluate_predictions(df, true_col="WIN_PCT", pred_col="PRED_SCORE"):
    df = df.copy()
    df = df.sort_values(true_col, ascending=False)
    df["TRUE_RANK"] = df[true_col].rank(method="min", ascending=False).astype(int)
    df["PRED_RANK"] = df[pred_col].rank(method="min", ascending=False).astype(int)
    df["RANK_DIFF"] = df["PRED_RANK"] - df["TRUE_RANK"]

    exact_acc = (df["PRED_RANK"] == df["TRUE_RANK"]).mean()
    within1_acc = (df["RANK_DIFF"].abs() <= 1).mean()
    within2_acc = (df["RANK_DIFF"].abs() <= 2).mean()
    overall_score = within1_acc * 0.7 + exact_acc * 0.3

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
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
    data = build_team_level_dataframe()

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[INFO] Number of numeric feature columns (initial merged df): {len(numeric_cols)}")
    print(f"[INFO] Example numeric columns: {numeric_cols[:10]}")

    full_feat, feature_cols = create_trend_features(data)

    seasons_sorted = sorted(full_feat["Season"].unique())
    latest_season = seasons_sorted[-1]
    second_latest_season = seasons_sorted[-2] if len(seasons_sorted) > 1 else latest_season
    holdout_seasons = [second_latest_season, latest_season]

    print(f"[INFO] Holdout seasons: {holdout_seasons}")
    print(f"[INFO] All seasons: {seasons_sorted}")

    train_feat = full_feat[~full_feat["Season"].isin(holdout_seasons)].copy()
    test_feat_all = full_feat[full_feat["Season"].isin(holdout_seasons)].copy()

    X_train = train_feat[feature_cols].values
    y_train = train_feat["WIN_PCT"].values

    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    all_season_results = []

    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS WITH BEST CONFIG (trend_only + RF + StandardScaler)")
    print("=" * 80)

    for season in holdout_seasons:
        print("\n" + "=" * 50)
        print(f"SEASON: {season}")
        print("=" * 50)

        season_feat = test_feat_all[test_feat_all["Season"] == season].copy()
        if season_feat.empty:
            continue

        for col in feature_cols:
            if col not in season_feat.columns:
                season_feat[col] = 0.0

        X_test = season_feat[feature_cols].values
        X_test_scaled = preprocessor.transform(X_test)

        pred = model.predict(X_test_scaled)
        season_feat["PRED_SCORE"] = pred

        metrics = evaluate_predictions(season_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
        all_season_results.append(metrics)

        print("\nPerformance Metrics:")
        print(f"  Exact accuracy:  {metrics['exact_rank_acc']:.4f}")
        print(f"  Within 1 rank:   {metrics['within1_rank_acc']:.4f}")
        print(f"  Within 2 ranks:  {metrics['within2_rank_acc']:.4f}")
        print(f"  MAE (WIN_PCT):   {metrics['mae']:.4f}")
        print(f"  RMSE (WIN_PCT):  {metrics['rmse']:.4f}")
        print(f"  R^2:             {metrics['r2']:.4f}")
        print(f"  Pearson r:       {metrics['pearson_r']:.4f}")
        print(f"  Spearman rho:    {metrics['spearman_r']:.4f}")

        season_feat["PRED_RANK"] = season_feat["PRED_SCORE"].rank(method="min", ascending=False).astype(int)
        season_feat["TRUE_RANK"] = season_feat["WIN_PCT"].rank(method="min", ascending=False).astype(int)
        season_feat["RANK_DIFF"] = season_feat["PRED_RANK"] - season_feat["TRUE_RANK"]

        season_sorted = season_feat.sort_values("PRED_RANK")

        print("\nFull Ranking Predictions:")
        print("-" * 70)
        print(f"{'Pred':^5} {'Actual':^6} {'Diff':^6} {'Team':^20} {'Pred':^8} {'Actual':^8}")
        print("-" * 70)

        correct_within1 = 0
        for _, row in season_sorted.iterrows():
            diff = int(row["RANK_DIFF"])
            diff_symbol = "✓" if abs(diff) <= 1 else "✗"
            if abs(diff) <= 1:
                correct_within1 += 1
            print(
                f"{row['PRED_RANK']:^5} "
                f"{row['TRUE_RANK']:^6} "
                f"{f'{diff_symbol}{diff:+2d}':^6} "
                f"{row['Team']:20s} "
                f"{row['PRED_SCORE']:^8.3f} "
                f"{row['WIN_PCT']:^8.3f}"
            )

        print("-" * 70)
        print(f"Correct within 1 rank: {correct_within1}/{len(season_sorted)} = {correct_within1 / len(season_sorted):.2%}")

    if all_season_results:
        avg_exact = np.mean([r["exact_rank_acc"] for r in all_season_results])
        avg_within1 = np.mean([r["within1_rank_acc"] for r in all_season_results])
        avg_mae = np.mean([r["mae"] for r in all_season_results])
        avg_rmse = np.mean([r["rmse"] for r in all_season_results])
        avg_r2 = np.mean([r["r2"] for r in all_season_results])
        avg_pearson = np.mean([r["pearson_r"] for r in all_season_results])
        avg_spearman = np.mean([r["spearman_r"] for r in all_season_results])

        print("\n" + "=" * 80)
        print("FINAL AVERAGE RANKING RESULTS (over holdout seasons)")
        print("=" * 80)
        print(f"Average Exact accuracy:   {avg_exact:.4f}")
        print(f"Average Within 1 rank:    {avg_within1:.4f}")
        print(f"Average MAE (WIN_PCT):    {avg_mae:.4f}")
        print(f"Average RMSE (WIN_PCT):   {avg_rmse:.4f}")
        print(f"Average R^2:              {avg_r2:.4f}")
        print(f"Average Pearson r:        {avg_pearson:.4f}")
        print(f"Average Spearman rho:     {avg_spearman:.4f}")

    print("\n" + "=" * 60)
    print("BEST CONFIG SUMMARY")
    print("=" * 60)
    print("Feature method: trend_only")
    print("Model type:     RandomForestRegressor")
    print("Preprocessor:   StandardScaler")
    print("Hyperparams:    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1")


if __name__ == "__main__":
    main()