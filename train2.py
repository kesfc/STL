import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

summary_path = "Team_stats/all_seasons_team_summary.csv"
REG_SEASON_GAMES = 82


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Team", "SeasonEndYear"])

    if "WIN_PCT" not in df.columns:
        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
    if "PREV_WIN_PCT" not in df.columns:
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"] + 1e-8))
    else:
        df["TS_PCT"] = np.nan

    if {"FG", "3P", "FGA"}.issubset(df.columns):
        df["EFG_PCT"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1e-8)
    else:
        df["EFG_PCT"] = np.nan

    if {"AST", "TOV"}.issubset(df.columns):
        df["AST_TO_RATIO"] = df["AST"] / (df["TOV"] + 1e-8)
    else:
        df["AST_TO_RATIO"] = np.nan

    if {"FG", "3P", "FGA"}.issubset(df.columns):
        df["EFG_FACTOR"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1e-8)
    else:
        df["EFG_FACTOR"] = np.nan

    if {"FGA", "FTA", "TOV"}.issubset(df.columns):
        df["TOV_FACTOR"] = 1 - df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"] + 1e-8)
    else:
        df["TOV_FACTOR"] = np.nan

    if {"ORB", "DRB"}.issubset(df.columns):
        df["OREB_FACTOR"] = df["ORB"] / (df["ORB"] + df["DRB"] + 1e-8)
    else:
        df["OREB_FACTOR"] = np.nan

    if {"FTA", "FGA"}.issubset(df.columns):
        df["FTR_FACTOR"] = df["FTA"] / (df["FGA"] + 1e-8)
    else:
        df["FTR_FACTOR"] = np.nan

    for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
        if col in df.columns:
            df[f"{col}_3Y_AVG"] = df.groupby("Team")[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        else:
            df[f"{col}_3Y_AVG"] = np.nan

    df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["WIN_PCT_5Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
        lambda x: x.rolling(3, min_periods=1).mean().diff()
    )

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


BEST_FEATURE_GROUPS = ["basic_eff", "rolling_box", "four_factors", "trend"]

BEST_FEATURE_COLS = [
    "AST_3Y_AVG",
    "AST_TO_RATIO",
    "BLK_3Y_AVG",
    "EFG_FACTOR",
    "EFG_PCT",
    "FTR_FACTOR",
    "OREB_FACTOR",
    "PREV_WIN_PCT",
    "PTS_3Y_AVG",
    "STL_3Y_AVG",
    "TOV_FACTOR",
    "TRB_3Y_AVG",
    "TS_PCT",
    "WIN_PCT_3Y_AVG",
    "WIN_PCT_5Y_AVG",
    "WIN_PCT_TREND",
]


def evaluate_predictions(df, true_col="WIN_PCT", pred_col="PRED_SCORE"):
    df = df.copy()
    df = df.dropna(subset=[true_col, pred_col])

    df = df.sort_values(true_col, ascending=False)
    df["TRUE_RANK"] = df[true_col].rank(method="min", ascending=False).astype(int)
    df["PRED_RANK"] = df[pred_col].rank(method="min", ascending=False).astype(int)
    df["RANK_DIFF"] = df["PRED_RANK"] - df["TRUE_RANK"]

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    exact_acc = (df["PRED_RANK"] == df["TRUE_RANK"]).mean()
    within1_acc = (df["RANK_DIFF"].abs() <= 1).mean()
    within2_acc = (df["RANK_DIFF"].abs() <= 2).mean()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    var_true = np.var(y_true)
    if var_true > 0:
        r2 = float(1 - np.mean((y_true - y_pred) ** 2) / var_true)
    else:
        r2 = np.nan

    if len(df) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = np.nan

    if len(df) >= 2 and np.std(df["TRUE_RANK"]) > 0 and np.std(df["PRED_RANK"]) > 0:
        spearman_rho = float(
            np.corrcoef(df["TRUE_RANK"].values, df["PRED_RANK"].values)[0, 1]
        )
    else:
        spearman_rho = np.nan

    overall_score = within1_acc * 0.7 + exact_acc * 0.3

    return {
        "exact_rank_acc": float(exact_acc),
        "within1_rank_acc": float(within1_acc),
        "within2_rank_acc": float(within2_acc),
        "overall_score": float(overall_score),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


def main():
    print(f"[INFO] Loading data from: {summary_path}")
    data = pd.read_csv(summary_path)

    if "WINS" not in data.columns or "LOSSES" not in data.columns:
        raise ValueError("CSV must contain 'WINS' and 'LOSSES' columns")

    num_cols_global = data.select_dtypes(include=[np.number]).columns
    data[num_cols_global] = data[num_cols_global].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["WINS", "LOSSES"])

    data = data.sort_values(["Team", "SeasonEndYear"])
    data["WIN_PCT"] = data["WINS"] / (data["WINS"] + data["LOSSES"])
    data["PREV_WIN_PCT"] = data.groupby("Team")["WIN_PCT"].shift(1)

    latest_season = data["Season"].max()
    unique_seasons = sorted(data["Season"].unique())
    second_latest_season = unique_seasons[-2] if len(unique_seasons) > 1 else latest_season
    holdout_seasons = [second_latest_season, latest_season]

    print(f"[INFO] Data loaded: {len(data)} rows, {data['Season'].nunique()} seasons")
    print(f"[INFO] Holdout seasons: {holdout_seasons}")
    print(f"[INFO] Training seasons: {sorted(set(data['Season']) - set(holdout_seasons))}")

    full_feat = build_feature_matrix(data)

    for df_ in (full_feat,):
        num_cols = df_.select_dtypes(include=[np.number]).columns
        df_[num_cols] = df_[num_cols].replace([np.inf, -np.inf], np.nan)
        df_.dropna(subset=["WIN_PCT"], inplace=True)

    numeric_cols = full_feat.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[INFO] Number of numeric feature columns (after feature matrix): {len(numeric_cols)}")
    print(f"[INFO] Example numeric columns: {numeric_cols[:10]}")

    train_feat = full_feat[~full_feat["Season"].isin(holdout_seasons)].copy()
    test_feat_all = full_feat[full_feat["Season"].isin(holdout_seasons)].copy()

    feature_cols = [c for c in BEST_FEATURE_COLS if c in train_feat.columns]
    missing = sorted(set(BEST_FEATURE_COLS) - set(feature_cols))
    if missing:
        print(f"[WARN] Missing feature columns (will be ignored): {missing}")
    print(f"[INFO] Using {len(feature_cols)} feature columns.")

    X_train = train_feat[feature_cols].copy()
    y_train = train_feat["WIN_PCT"].values

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    train_means = X_train.mean(axis=0)
    X_train = X_train.fillna(train_means)

    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    all_season_results = []

    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS WITH BEST CONFIG (basic_eff+rolling_box+four_factors+trend + GB + StandardScaler)")
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

        X_test = season_feat[feature_cols].copy()
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(train_means)

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
        print(f"  Spearman rho:    {metrics['spearman_rho']:.4f}")

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
        print(
            f"Correct within 1 rank: {correct_within1}/{len(season_sorted)} = "
            f"{correct_within1 / len(season_sorted):.2%}"
        )

    if all_season_results:
        avg_exact = float(np.mean([r["exact_rank_acc"] for r in all_season_results]))
        avg_within1 = float(np.mean([r["within1_rank_acc"] for r in all_season_results]))
        avg_within2 = float(np.mean([r["within2_rank_acc"] for r in all_season_results]))
        avg_mae = float(np.mean([r["mae"] for r in all_season_results]))
        avg_rmse = float(np.mean([r["rmse"] for r in all_season_results]))
        avg_r2 = float(np.mean([r["r2"] for r in all_season_results]))
        avg_pearson = float(np.mean([r["pearson_r"] for r in all_season_results]))
        avg_spearman = float(np.mean([r["spearman_rho"] for r in all_season_results]))

        print("\n" + "=" * 80)
        print("FINAL AVERAGE RANKING RESULTS (over holdout seasons)")
        print("=" * 80)
        print(f"Average Exact accuracy:   {avg_exact:.4f}")
        print(f"Average Within 1 rank:    {avg_within1:.4f}")
        print(f"Average Within 2 ranks:   {avg_within2:.4f}")
        print(f"Average MAE (WIN_PCT):    {avg_mae:.4f}")
        print(f"Average RMSE (WIN_PCT):   {avg_rmse:.4f}")
        print(f"Average R^2:              {avg_r2:.4f}")
        print(f"Average Pearson r:        {avg_pearson:.4f}")
        print(f"Average Spearman rho:     {avg_spearman:.4f}")

    print("\n" + "=" * 60)
    print("BEST CONFIG SUMMARY")
    print("=" * 60)
    print(f"Feature groups:  {BEST_FEATURE_GROUPS}")
    print("Model type:      GradientBoostingRegressor")
    print("Preprocessor:    StandardScaler")
    print("Hyperparams:     n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42")


if __name__ == "__main__":
    main()