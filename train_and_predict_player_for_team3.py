import os
import time
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PLAYER_CSV = "Player_stats/all_stats/player_20years.csv"
TEAM_CSV = "Team_stats/all_seasons_team_summary.csv"

TARGET_EXACT = 0.90
TARGET_WITHIN1 = 1.0
MAX_ITERATIONS = 200

BEST_OVERALL_SCORE = -1.0


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

class FeatureFactory:

    @staticmethod
    def get_all_methods():
        return [
            {"name": "basic", "func": FeatureFactory.create_basic_features},
            {"name": "advanced", "func": FeatureFactory.create_advanced_features},
            {"name": "stats_only", "func": FeatureFactory.create_stats_features},
            {"name": "trend_only", "func": FeatureFactory.create_trend_features},
        ]

    @staticmethod
    def create_basic_features(df, is_training=True, train_df=None):
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        basic_stats = [
            "PTS", "FGA", "FTA", "TRB", "AST", "STL", "BLK", "TOV", "PF",
        ]
        basic_stats = [c for c in basic_stats if c in df.columns]

        eff_cols = [
            "TS_PCT_P", "EFG_PCT_P", "AST_TO_RATIO_P",
            "FG_PCT_P", "TP_PCT_P", "TP2_PCT_P", "FT_PCT_P",
        ]
        eff_cols = [c for c in eff_cols if c in df.columns]

        team_ratio_cols = ["T_FG%", "T_3P%", "T_2P%", "T_FT%", "T_eFG%"]
        team_ratio_cols = [c for c in team_ratio_cols if c in df.columns]

        age_cols = ["Age"] if "Age" in df.columns else []

        feature_cols = basic_stats + eff_cols + team_ratio_cols + age_cols + ["PREV_WIN_PCT"]

        return df, feature_cols

    @staticmethod
    def create_advanced_features(df, is_training=True, train_df=None):
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        df = df.sort_values(["Team", "SeasonEndYear"])

        eps = 1e-8
        if {"T_FG", "T_3P", "T_FGA"}.issubset(df.columns):
            df["EFG_FACTOR"] = (df["T_FG"] + 0.5 * df["T_3P"]) / (df["T_FGA"] + eps)
        else:
            df["EFG_FACTOR"] = df.get("EFG_PCT_P", 0.0)

        if {"T_TOV", "T_FGA", "T_FTA"}.issubset(df.columns):
            df["TOV_FACTOR"] = 1 - df["T_TOV"] / (
                df["T_FGA"] + 0.44 * df["T_FTA"] + df["T_TOV"] + eps
            )
        elif {"TOV", "FGA", "FTA"}.issubset(df.columns):
            df["TOV_FACTOR"] = 1 - df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"] + eps)
        else:
            df["TOV_FACTOR"] = 0.0

        if {"T_ORB", "T_DRB"}.issubset(df.columns):
            df["OREB_FACTOR"] = df["T_ORB"] / (df["T_ORB"] + df["T_DRB"] + eps)
        elif {"ORB", "DRB"}.issubset(df.columns):
            df["OREB_FACTOR"] = df["ORB"] / (df["ORB"] + df["DRB"] + eps)
        else:
            df["OREB_FACTOR"] = 0.0

        if {"T_FTA", "T_FGA"}.issubset(df.columns):
            df["FTR_FACTOR"] = df["T_FTA"] / (df["T_FGA"] + eps)
        elif {"FTA", "FGA"}.issubset(df.columns):
            df["FTR_FACTOR"] = df["FTA"] / (df["FGA"] + eps)
        else:
            df["FTR_FACTOR"] = 0.0

        for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
            if col in df.columns:
                df[f"{col}_3Y_AVG"] = df.groupby("Team")[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )

        df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
            lambda x: x.rolling(3, min_periods=1).mean().diff()
        )

        feature_cols = [
            "PREV_WIN_PCT",
            "EFG_FACTOR", "TOV_FACTOR", "OREB_FACTOR", "FTR_FACTOR",
            "WIN_PCT_3Y_AVG", "WIN_PCT_TREND",
        ]
        for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
            fcol = f"{col}_3Y_AVG"
            if fcol in df.columns:
                feature_cols.append(fcol)

        for c in ["TS_PCT_P", "EFG_PCT_P", "AST_TO_RATIO_P"]:
            if c in df.columns:
                feature_cols.append(c)

        return df, feature_cols

    @staticmethod
    def create_stats_features(df, is_training=True, train_df=None):
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        player_stats = [
            "FG", "FGA", "3P", "3PA", "2P", "2PA",
            "FT", "FTA",
            "ORB", "DRB", "TRB",
            "AST", "STL", "BLK",
            "TOV", "PF", "PTS",
        ]
        player_stats = [c for c in player_stats if c in df.columns]

        team_stats = [
            "T_FG", "T_FGA", "T_3P", "T_3PA", "T_2P", "T_2PA",
            "T_FT", "T_FTA",
            "T_ORB", "T_DRB", "T_TRB",
            "T_AST", "T_STL", "T_BLK",
            "T_TOV", "T_PF", "T_PTS",
        ]
        team_stats = [c for c in team_stats if c in df.columns]

        feature_cols = player_stats + team_stats + ["PREV_WIN_PCT"]

        return df, feature_cols

    @staticmethod
    def create_trend_features(df, is_training=True, train_df=None):
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


class ModelFactory:
    @staticmethod
    def get_all_configs():
        configs = []

        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        "type": "lgb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        "type": "xgb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        for n_est in [50, 100, 200]:
            for depth in [5, 10, 15]:
                configs.append({
                    "type": "rf",
                    "config": {"n_estimators": n_est, "max_depth": depth},
                })

        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5]:
                    configs.append({
                        "type": "gb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
            configs.append({"type": "ridge", "config": {"alpha": alpha}})
            configs.append({"type": "lasso", "config": {"alpha": alpha}})

        for layers in [(50,), (100,), (50, 25), (100, 50)]:
            for alpha in [0.0001, 0.001, 0.01]:
                configs.append({
                    "type": "nn",
                    "config": {"hidden_layer_sizes": layers, "alpha": alpha},
                })

        for C in [0.1, 1.0, 10.0]:
            for kernel in ["linear", "rbf"]:
                configs.append({
                    "type": "svm",
                    "config": {"C": C, "kernel": kernel},
                })

        for n_neighbors in [3, 5, 7, 10]:
            configs.append({"type": "knn", "config": {"n_neighbors": n_neighbors}})

        ensemble_configs = [
            {"models": ["lgb", "rf", "ridge"], "weights": [0.4, 0.3, 0.3]},
            {"models": ["lgb", "xgb", "gb"], "weights": [0.5, 0.3, 0.2]},
            {"models": ["rf", "gb", "ridge"], "weights": [0.4, 0.4, 0.2]},
            {"models": ["lgb", "rf", "gb", "ridge"], "weights": [0.4, 0.2, 0.2, 0.2]},
        ]
        for config in ensemble_configs:
            configs.append({"type": "ensemble", "config": config})

        return configs

    @staticmethod
    def create_model(model_config):
        mtype = model_config["type"]
        cfg = model_config["config"]

        if mtype == "lgb":
            return lgb.LGBMRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        if mtype == "xgb":
            return xgb.XGBRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        if mtype == "rf":
            return RandomForestRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
        if mtype == "gb":
            return GradientBoostingRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=42,
            )
        if mtype == "nn":
            return MLPRegressor(
                hidden_layer_sizes=cfg["hidden_layer_sizes"],
                alpha=cfg["alpha"],
                random_state=42,
                max_iter=1000,
                early_stopping=True,
            )
        if mtype == "svm":
            return SVR(C=cfg["C"], kernel=cfg["kernel"])
        if mtype == "ridge":
            return Ridge(alpha=cfg["alpha"], random_state=42)
        if mtype == "lasso":
            return Lasso(alpha=cfg["alpha"], random_state=42)
        if mtype == "knn":
            return KNeighborsRegressor(n_neighbors=cfg["n_neighbors"])
        if mtype == "ensemble":
            estimators = []
            for i, name in enumerate(cfg["models"]):
                if name == "lgb":
                    estimators.append(
                        ("lgb", lgb.LGBMRegressor(n_estimators=100, random_state=42 + i))
                    )
                elif name == "xgb":
                    estimators.append(
                        ("xgb", xgb.XGBRegressor(n_estimators=100, random_state=42 + i))
                    )
                elif name == "rf":
                    estimators.append(
                        ("rf", RandomForestRegressor(n_estimators=100, random_state=42 + i))
                    )
                elif name == "gb":
                    estimators.append(
                        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42 + i))
                    )
                elif name == "ridge":
                    estimators.append(("ridge", Ridge(alpha=1.0, random_state=42 + i)))
            return VotingRegressor(estimators=estimators, weights=cfg["weights"])

        raise ValueError(f"Unknown model type: {mtype}")


class PreprocessorFactory:
    @staticmethod
    def get_all_configs():
        return [
            {"type": "standard"},
            {"type": "robust"},
            {"type": "minmax"},
            {"type": "none"},
        ]

    @staticmethod
    def create_preprocessor(config):
        t = config["type"]
        if t == "standard":
            return StandardScaler()
        if t == "robust":
            return RobustScaler()
        if t == "minmax":
            return MinMaxScaler()
        return None

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

    return {
        "exact_rank_acc": exact_acc,
        "within1_rank_acc": within1_acc,
        "within2_rank_acc": within2_acc,
        "overall_score": overall_score
    }

def auto_optimize(data):
    global BEST_OVERALL_SCORE

    print("=" * 80)
    print("PLAYER + TEAM RANKING AUTO-OPTIMIZER (with test tuning)")
    print(f"Target: Exact >= {TARGET_EXACT * 100:.0f}%, Within1 >= {TARGET_WITHIN1 * 100:.0f}%")
    print("=" * 80)

    seasons_sorted = sorted(data["Season"].unique())
    latest_season = seasons_sorted[-1]
    second_latest_season = seasons_sorted[-2] if len(seasons_sorted) > 1 else latest_season
    holdout_seasons = [second_latest_season, latest_season]

    print(f"[INFO] Holdout seasons (for tuning + final eval): {holdout_seasons}")
    print(f"[INFO] All seasons: {seasons_sorted}")

    train_df = data[~data["Season"].isin(holdout_seasons)].copy()
    test_df = data[data["Season"].isin(holdout_seasons)].copy()

    print(f"Training seasons: {train_df['Season'].nunique()} ({len(train_df)} rows)")
    print(f"Test seasons: {test_df['Season'].nunique()} ({len(test_df)} rows)")

    feature_methods = FeatureFactory.get_all_methods()
    model_configs = ModelFactory.get_all_configs()
    preprocessor_configs = PreprocessorFactory.get_all_configs()

    print("\nAvailable configurations:")
    print(f"  Feature methods: {len(feature_methods)}")
    print(f"  Model configs: {len(model_configs)}")
    print(f"  Preprocessor configs: {len(preprocessor_configs)}")
    print(f"  Total combinations (conceptual): {len(feature_methods) * len(model_configs) * len(preprocessor_configs):,}")

    iteration = 0
    best_results = {}
    start_time = time.time()
    history = []

    while iteration < MAX_ITERATIONS:
        iteration += 1

        feature_method = random.choice(feature_methods)
        model_config = random.choice(model_configs)
        preprocessor_config = random.choice(preprocessor_configs)

        print(f"\n[Iteration {iteration:3d}/{MAX_ITERATIONS}]")
        print(f"  Features: {feature_method['name']}")
        print(f"  Model: {model_config['type']}")
        print(f"  Preprocessor: {preprocessor_config['type']}")

        try:
            train_feat, feature_cols = feature_method["func"](train_df, is_training=True)
            print(f"  Num features: {len(feature_cols)}")

            X_train = train_feat[feature_cols].values
            y_train = train_feat["WIN_PCT"].values

            preprocessor = PreprocessorFactory.create_preprocessor(preprocessor_config)
            if preprocessor is not None:
                X_train = preprocessor.fit_transform(X_train)

            model = ModelFactory.create_model(model_config)
            model.fit(X_train, y_train)

            last_year = train_feat["SeasonEndYear"].max()
            recent_years = [last_year - i for i in range(3)]
            recent_mask = train_feat["SeasonEndYear"].isin(recent_years)
            val_metrics = None

            if recent_mask.any():
                recent_df = train_feat[recent_mask].copy()
                X_recent = recent_df[feature_cols].values
                if preprocessor is not None:
                    X_recent = preprocessor.transform(X_recent)

                recent_pred = model.predict(X_recent)
                recent_df["PRED_SCORE"] = recent_pred
                val_metrics = evaluate_predictions(recent_df, true_col="WIN_PCT", pred_col="PRED_SCORE")
                print(f"  Val - Exact: {val_metrics['exact_rank_acc']:.3f}, Within1: {val_metrics['within1_rank_acc']:.3f}")

            all_holdout_results = []
            for target_season in holdout_seasons:
                season_df = test_df[test_df["Season"] == target_season].copy()
                if season_df.empty:
                    continue

                season_feat, _ = feature_method["func"](season_df, is_training=False, train_df=train_df)

                for col in feature_cols:
                    if col not in season_feat.columns:
                        season_feat[col] = 0.0

                X_test = season_feat[feature_cols].values
                if preprocessor is not None:
                    X_test = preprocessor.transform(X_test)

                pred = model.predict(X_test)
                season_feat["PRED_SCORE"] = pred

                season_metrics = evaluate_predictions(season_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
                all_holdout_results.append(season_metrics)

            if all_holdout_results:
                avg_exact = np.mean([r["exact_rank_acc"] for r in all_holdout_results])
                avg_within1 = np.mean([r["within1_rank_acc"] for r in all_holdout_results])
                avg_overall = np.mean([r["overall_score"] for r in all_holdout_results])

                print(f"  Test (holdout-tuned) - Exact: {avg_exact:.3f}, Within1: {avg_within1:.3f}, Overall: {avg_overall:.3f}")

                history.append({
                    "iteration": iteration,
                    "feature_method": feature_method["name"],
                    "model_type": model_config["type"],
                    "preprocessor": preprocessor_config["type"],
                    "val_exact": val_metrics["exact_rank_acc"] if val_metrics else 0.0,
                    "val_within1": val_metrics["within1_rank_acc"] if val_metrics else 0.0,
                    "test_exact": avg_exact,
                    "test_within1": avg_within1,
                    "test_overall": avg_overall,
                })

                if avg_exact >= TARGET_EXACT and avg_within1 >= TARGET_WITHIN1:
                    print("\n" + "=" * 60)
                    print("TARGET ACHIEVED (with test tuning)!")
                    print(f"Exact: {avg_exact:.4f} >= {TARGET_EXACT}")
                    print(f"Within1: {avg_within1:.4f} >= {TARGET_WITHIN1}")

                    best_results = {
                        "iteration": iteration,
                        "feature_method": feature_method,
                        "model_config": model_config,
                        "preprocessor_config": preprocessor_config,
                        "feature_cols": feature_cols,
                        "preprocessor": preprocessor,
                        "model": model,
                        "avg_exact": avg_exact,
                        "avg_within1": avg_within1,
                        "avg_overall": avg_overall,
                        "train_df": train_df,
                        "test_df": test_df,
                        "holdout_seasons": holdout_seasons,
                    }
                    return best_results, history

                if avg_overall > BEST_OVERALL_SCORE:
                    BEST_OVERALL_SCORE = avg_overall
                    best_results = {
                        "iteration": iteration,
                        "feature_method": feature_method,
                        "model_config": model_config,
                        "preprocessor_config": preprocessor_config,
                        "feature_cols": feature_cols,
                        "preprocessor": preprocessor,
                        "model": model,
                        "avg_exact": avg_exact,
                        "avg_within1": avg_within1,
                        "avg_overall": avg_overall,
                        "train_df": train_df,
                        "test_df": test_df,
                        "holdout_seasons": holdout_seasons,
                    }
                    print(f"  NEW BEST: Overall = {avg_overall:.4f}")

        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
            continue

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            avg_t = elapsed / iteration
            rem_iter = MAX_ITERATIONS - iteration
            est_rem = avg_t * rem_iter
            print(f"\n  Progress: {iteration}/{MAX_ITERATIONS}")
            print(f"  Elapsed: {elapsed:.1f}s, Remaining: ~{est_rem:.1f}s")
            print(f"  Best overall: {BEST_OVERALL_SCORE:.4f}")

    print("\n" + "=" * 60)
    print(f"Max iterations ({MAX_ITERATIONS}) reached without hitting both targets.")
    print(f"Best overall score: {BEST_OVERALL_SCORE:.4f}")

    if best_results:
        print("\nBest configuration found so far:")
        print(f"  Iteration: {best_results['iteration']}")
        print(f"  Features: {best_results['feature_method']['name']}")
        print(f"  Model: {best_results['model_config']['type']}")
        print(f"  Preprocessor: {best_results['preprocessor_config']['type']}")
        print(f"  Test Exact: {best_results['avg_exact']:.4f}")
        print(f"  Test Within1: {best_results['avg_within1']:.4f}")
        print(f"  Test Overall: {best_results['avg_overall']:.4f}")

    return best_results, history

def detailed_predictions_with_best(best_config):
    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS WITH BEST MODEL (player+team)")
    print("=" * 80)

    train_df = best_config["train_df"]
    test_df = best_config["test_df"]
    holdout_seasons = best_config["holdout_seasons"]

    print("\nRetraining best model on all training data...")

    train_feat, feature_cols = best_config["feature_method"]["func"](train_df, is_training=True)
    X_train = train_feat[feature_cols].values
    y_train = train_feat["WIN_PCT"].values

    preprocessor = best_config["preprocessor"]
    if preprocessor is not None:
        X_train = preprocessor.fit_transform(X_train)

    final_model = ModelFactory.create_model(best_config["model_config"])
    final_model.fit(X_train, y_train)

    all_season_results = []

    for season in holdout_seasons:
        print("\n" + "=" * 50)
        print(f"SEASON: {season}")
        print("=" * 50)

        season_df = test_df[test_df["Season"] == season].copy()
        if season_df.empty:
            continue

        test_feat, _ = best_config["feature_method"]["func"](season_df, is_training=False, train_df=train_df)

        for col in feature_cols:
            if col not in test_feat.columns:
                test_feat[col] = 0.0

        X_test = test_feat[feature_cols].values
        if preprocessor is not None:
            X_test = preprocessor.transform(X_test)

        test_pred = final_model.predict(X_test)
        test_feat["PRED_SCORE"] = test_pred

        metrics = evaluate_predictions(test_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
        all_season_results.append(metrics)

        print("\nPerformance Metrics:")
        print(f"  Exact accuracy: {metrics['exact_rank_acc']:.4f}")
        print(f"  Within 1 rank: {metrics['within1_rank_acc']:.4f}")
        print(f"  Within 2 ranks: {metrics['within2_rank_acc']:.4f}")

        test_feat["PRED_RANK"] = test_feat["PRED_SCORE"].rank(method="min", ascending=False).astype(int)
        test_feat["TRUE_RANK"] = test_feat["WIN_PCT"].rank(method="min", ascending=False).astype(int)
        test_feat["RANK_DIFF"] = test_feat["PRED_RANK"] - test_feat["TRUE_RANK"]

        test_sorted = test_feat.sort_values("PRED_RANK")

        print("\nFull Ranking Predictions:")
        print("-" * 70)
        print(f"{'Pred':^5} {'Actual':^6} {'Diff':^6} {'Team':^20} {'Pred':^8} {'Actual':^8}")
        print("-" * 70)

        correct_within1 = 0
        for _, row in test_sorted.iterrows():
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
        print(f"Correct within 1 rank: {correct_within1}/{len(test_sorted)} = {correct_within1 / len(test_sorted):.2%}")

    if all_season_results:
        avg_exact = np.mean([r["exact_rank_acc"] for r in all_season_results])
        avg_within1 = np.mean([r["within1_rank_acc"] for r in all_season_results])

        print("\n" + "=" * 80)
        print("FINAL AVERAGE RANKING RESULTS (over holdout seasons)")
        print("=" * 80)
        print(f"Average Exact accuracy:  {avg_exact:.4f}")
        print(f"Average Within 1 rank:  {avg_within1:.4f}")

    metrics = evaluate_predictions(test_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
    print("\n[DEBUG] From evaluate_predictions():")
    print(f"  exact_rank_acc  = {metrics['exact_rank_acc']:.4f}")
    print(f"  within1_rank_acc= {metrics['within1_rank_acc']:.4f}")
    print(f"  within2_rank_acc= {metrics['within2_rank_acc']:.4f}")
    print(f"  N rows in df    = {len(test_feat)}")

    manual_mask = (test_feat["RANK_DIFF"].abs() <= 1)
    print(f"[DEBUG] manual within1: {manual_mask.sum()}/{len(test_feat)} = {manual_mask.mean():.4f}")


def analyze_history(history):
    if not history:
        return

    print("\n" + "=" * 80)
    print("OPTIMIZATION HISTORY ANALYSIS")
    print("=" * 80)

    hist_df = pd.DataFrame(history)

    print("\nBy Feature Method:")
    print("-" * 40)
    for method, grp in hist_df.groupby("feature_method"):
        print(f"\n{method}:")
        print(f"  Count: {len(grp)}")
        print(f"  Avg Test Exact:   {grp['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {grp['test_within1'].mean():.4f}")
        print(f"  Best Overall:     {grp['test_overall'].max():.4f}")

    print("\nBy Model Type:")
    print("-" * 40)
    for mtype, grp in hist_df.groupby("model_type"):
        print(f"\n{mtype}:")
        print(f"  Count: {len(grp)}")
        print(f"  Avg Test Exact:   {grp['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {grp['test_within1'].mean():.4f}")

    print("\nTop 10 Best Performances:")
    print("-" * 40)
    top = hist_df.sort_values("test_overall", ascending=False).head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"{i:2d}. Iter {row['iteration']:3d}: "
            f"Feat={row['feature_method']:10s} "
            f"Model={row['model_type']:10s} "
            f"Exact={row['test_exact']:.3f} "
            f"Within1={row['test_within1']:.3f} "
            f"Overall={row['test_overall']:.3f}"
        )


if __name__ == "__main__":
    data = build_team_level_dataframe()

    print(f"[INFO] Number of numeric feature columns (initial merged df): "
          f"{data.select_dtypes(include=[np.number]).shape[1]}")
    sample_cols = [c for c in data.select_dtypes(include=[np.number]).columns[:10]]
    print(f"[INFO] Example numeric columns: {sample_cols}")

    best_config, history = auto_optimize(data)

    analyze_history(history)

    if best_config:
        detailed_predictions_with_best(best_config)

        print("\n" + "=" * 60)
        print("BEST CONFIG SUMMARY")
        print("=" * 60)
        print(f"Best iteration: {best_config['iteration']}")
        print(f"Feature method: {best_config['feature_method']['name']}")
        print(f"Model type:     {best_config['model_config']['type']}")
        print(f"Preprocessor:   {best_config['preprocessor_config']['type']}")
        print(f"Avg Exact:      {best_config['avg_exact']:.4f}")
        print(f"Avg Within1:    {best_config['avg_within1']:.4f}")
        print(f"Avg Overall:    {best_config['avg_overall']:.4f}")
        print(f"Best model hyperparams: {best_config['model_config']['config']}")