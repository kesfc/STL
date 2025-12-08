import os
import json
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ========== Use the combined team summary CSV directly ==========
summary_path = "Team_stats/all_seasons_team_summary.csv"

# ========== Data loading ==========
print(f"Loading data from: {summary_path}")
data = pd.read_csv(summary_path)

if "WINS" not in data.columns or "LOSSES" not in data.columns:
    raise ValueError("CSV must contain 'WINS' and 'LOSSES' columns")

num_cols_global = data.select_dtypes(include=[np.number]).columns
data[num_cols_global] = data[num_cols_global].replace([np.inf, -np.inf], np.nan)

data = data.dropna(subset=["WINS", "LOSSES"])

data = data.sort_values(["Team", "SeasonEndYear"])
data["WIN_PCT"] = data["WINS"] / (data["WINS"] + data["LOSSES"])
data["PREV_WIN_PCT"] = data.groupby("Team")["WIN_PCT"].shift(1)

REG_SEASON_GAMES = 82
latest_season = data["Season"].max()
second_latest_season = sorted(data["Season"].unique())[-2] if len(data["Season"].unique()) > 1 else latest_season
HOLDOUT_SEASONS = [second_latest_season, latest_season]

TARGET_EXACT = 0.8
TARGET_WITHIN1 = 0.8
MAX_ITERATIONS = 1000   
BEST_OVERALL_SCORE = -1

print(f"Data loaded: {len(data)} rows, {data['Season'].nunique()} seasons")
print(f"Holdout seasons: {HOLDOUT_SEASONS}")
print(f"Training seasons: {sorted(set(data['Season']) - set(HOLDOUT_SEASONS))}")


# ========== Build full feature matrix once ==========
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始数据基础上，一次性构造所有可能用到的 feature 列：
    - basic_box: 简单 box score
    - basic_eff: TS%, eFG%, AST/TOV, PREV_WIN_PCT
    - four_factors: 四因素
    - rolling_box: 3 年 rolling 平均 (PTS, AST, TRB, STL, BLK)
    - trend: WIN_PCT 的 rolling / trend
    """
    df = df.copy()
    df = df.sort_values(["Team", "SeasonEndYear"])


    if "WIN_PCT" not in df.columns:
        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
    if "PREV_WIN_PCT" not in df.columns:
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

    # ------- basic_eff -------
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

    # ------- four_factors -------
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

    # ------- rolling_box -------
    for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
        if col in df.columns:
            df[f"{col}_3Y_AVG"] = df.groupby("Team")[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        else:
            df[f"{col}_3Y_AVG"] = np.nan

    # ------- trend features -------
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


FEATURE_GROUPS = {
    "basic_box": ["PTS", "FGA", "FTA", "TRB", "AST", "STL", "BLK", "TOV", "PF"],
    "basic_eff": ["TS_PCT", "EFG_PCT", "AST_TO_RATIO", "PREV_WIN_PCT"],
    "four_factors": ["EFG_FACTOR", "TOV_FACTOR", "OREB_FACTOR", "FTR_FACTOR"],
    "full_box": [
        "FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA",
        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
    ],
    "rolling_box": [f"{c}_3Y_AVG" for c in ["PTS", "AST", "TRB", "STL", "BLK"]],
    "trend": ["WIN_PCT_3Y_AVG", "WIN_PCT_5Y_AVG", "WIN_PCT_TREND"],
}


# ========== Model factory ==========
class ModelFactory:
    @staticmethod
    def get_all_configs():
        configs = []

        # LightGBM
        for n_est in [50, 100, 200, 300]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7, 10]:
                    configs.append({
                        "type": "lgb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        # XGBoost
        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        "type": "xgb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        # Random Forest
        for n_est in [50, 100, 200, 300]:
            for depth in [5, 10, 15, 20]:
                configs.append({
                    "type": "rf",
                    "config": {"n_estimators": n_est, "max_depth": depth},
                })

        # Gradient Boosting
        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        "type": "gb",
                        "config": {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth},
                    })

        # Linear models
        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
            configs.append({"type": "ridge", "config": {"alpha": alpha}})
            configs.append({"type": "lasso", "config": {"alpha": alpha}})

        # Neural nets
        for layers in [(50,), (100,), (50, 25), (100, 50)]:
            for alpha in [0.0001, 0.001, 0.01]:
                configs.append({
                    "type": "nn",
                    "config": {"hidden_layer_sizes": layers, "alpha": alpha},
                })

        # SVM
        for C in [0.1, 1.0, 10.0]:
            for kernel in ["linear", "rbf"]:
                configs.append({"type": "svm", "config": {"C": C, "kernel": kernel}})

        # KNN
        for n_neighbors in [3, 5, 7, 10]:
            configs.append({"type": "knn", "config": {"n_neighbors": n_neighbors}})

        # Simple ensembles
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
        model_type = model_config["type"]
        config = model_config["config"]

        if model_type == "lgb":
            return lgb.LGBMRegressor(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "xgb":
            return xgb.XGBRegressor(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif model_type == "rf":
            return RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gb":
            return GradientBoostingRegressor(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                random_state=42,
            )
        elif model_type == "nn":
            return MLPRegressor(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                alpha=config["alpha"],
                random_state=42,
                max_iter=1000,
                early_stopping=True,
            )
        elif model_type == "svm":
            return SVR(C=config["C"], kernel=config["kernel"])
        elif model_type == "ridge":
            return Ridge(alpha=config["alpha"], random_state=42)
        elif model_type == "lasso":
            return Lasso(alpha=config["alpha"], random_state=42)
        elif model_type == "knn":
            return KNeighborsRegressor(n_neighbors=config["n_neighbors"])
        elif model_type == "ensemble":
            estimators = []
            for i, model_name in enumerate(config["models"]):
                if model_name == "lgb":
                    estimators.append(("lgb", lgb.LGBMRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == "xgb":
                    estimators.append(("xgb", xgb.XGBRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == "rf":
                    estimators.append(("rf", RandomForestRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == "gb":
                    estimators.append(("gb", GradientBoostingRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == "ridge":
                    estimators.append(("ridge", Ridge(alpha=1.0, random_state=42 + i)))
            return VotingRegressor(estimators=estimators, weights=config["weights"])


# ========== Preprocessor factory ==========
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
    def create_preprocessor(preprocessor_config):
        t = preprocessor_config["type"]
        if t == "standard":
            return StandardScaler()
        elif t == "robust":
            return RobustScaler()
        elif t == "minmax":
            return MinMaxScaler()
        else:
            return None


# ========== Evaluation function ==========
def evaluate_predictions(df, true_col="WIN_PCT", pred_col="PRED_SCORE"):
    """
    排名评估：exact / within 1 / within 2
    """
    df = df.copy()
    df["TRUE_RANK"] = df[true_col].rank(method="min", ascending=False).astype(int)
    df["PRED_RANK"] = df[pred_col].rank(method="min", ascending=False).astype(int)
    df["RANK_DIFF"] = df["PRED_RANK"] - df["TRUE_RANK"]

    exact_acc = (df["PRED_RANK"] == df["TRUE_RANK"]).mean()
    within1_acc = (df["RANK_DIFF"].abs() <= 1).mean()
    within2_acc = (df["RANK_DIFF"].abs() <= 2).mean()

    overall_score = within1_acc * 0.7 + exact_acc * 0.3
    return {
        "exact_rank_acc": float(exact_acc),
        "within1_rank_acc": float(within1_acc),
        "within2_rank_acc": float(within2_acc),
        "overall_score": float(overall_score),
    }


# ========== Main auto-optimization loop ==========
def auto_optimize():
    global BEST_OVERALL_SCORE

    print("=" * 80)
    print("NBA RANKING AUTO-OPTIMIZER")
    print(f"Target: Exact >= {TARGET_EXACT * 100:.0f}%, Within1 >= {TARGET_WITHIN1 * 100:.0f}%")
    print("=" * 80)

    full_df = data.copy()

    train_mask = ~full_df["Season"].isin(HOLDOUT_SEASONS)
    test_mask = full_df["Season"].isin(HOLDOUT_SEASONS)

    train_raw = full_df[train_mask].copy()
    test_raw = full_df[test_mask].copy()

    train_feat_all = build_feature_matrix(train_raw)
    test_feat_all = build_feature_matrix(test_raw)

    for df_ in (train_feat_all, test_feat_all):
        num_cols = df_.select_dtypes(include=[np.number]).columns
        df_[num_cols] = df_[num_cols].replace([np.inf, -np.inf], np.nan)
        df_.dropna(subset=["WIN_PCT"], inplace=True)

    print(f"Training seasons: {train_feat_all['Season'].nunique()} ({len(train_feat_all)} rows)")
    print(f"Test seasons: {test_feat_all['Season'].nunique()} ({len(test_feat_all)} rows)")

    model_configs = ModelFactory.get_all_configs()
    preprocessor_configs = PreprocessorFactory.get_all_configs()
    feature_group_names_all = list(FEATURE_GROUPS.keys())

    print(f"\nAvailable configurations:")
    print(f"  Feature groups: {len(feature_group_names_all)} ({feature_group_names_all})")
    print(f"  Model configs: {len(model_configs)}")
    print(f"  Preprocessor configs: {len(preprocessor_configs)}")

    iteration = 0
    best_results = {}
    start_time = time.time()
    history = []

    while iteration < MAX_ITERATIONS:
        iteration += 1

        while True:
            k = random.randint(1, len(feature_group_names_all))
            feature_groups = random.sample(feature_group_names_all, k)
            if not (len(feature_groups) == 1 and feature_groups[0] == "trend"):
                break

        model_config = random.choice(model_configs)
        preprocessor_config = random.choice(preprocessor_configs)

        print(f"\n[Iteration {iteration:4d}/{MAX_ITERATIONS}]")
        print(f"  Feature groups: {feature_groups}")
        print(f"  Model: {model_config['type']}")
        print(f"  Preprocessor: {preprocessor_config['type']}")

        try:
            cols = set()
            for g in feature_groups:
                for c in FEATURE_GROUPS[g]:
                    if c in train_feat_all.columns:
                        cols.add(c)
            feature_cols = sorted(cols)

            if not feature_cols:
                print("  -> No valid feature columns, skipping.")
                continue

            print(f"  Number of features: {len(feature_cols)}")

            X_train = train_feat_all[feature_cols].copy()
            y_train = train_feat_all["WIN_PCT"].values

            num_cols_iter = X_train.select_dtypes(include=[np.number]).columns
            X_train[num_cols_iter] = X_train[num_cols_iter].replace([np.inf, -np.inf], np.nan)
            train_means = X_train.mean(axis=0)
            X_train = X_train.fillna(train_means)

            preprocessor = PreprocessorFactory.create_preprocessor(preprocessor_config)
            if preprocessor is not None:
                X_train_proc = preprocessor.fit_transform(X_train)
            else:
                X_train_proc = X_train.values

            model = ModelFactory.create_model(model_config)
            model.fit(X_train_proc, y_train)

            last_year = int(train_feat_all["SeasonEndYear"].max())
            recent_years = [last_year - i for i in range(3)]
            val_mask = train_feat_all["SeasonEndYear"].isin(recent_years)
            val_metrics = None

            if val_mask.any():
                recent_df = train_feat_all[val_mask].copy()
                X_recent = recent_df[feature_cols].copy()

                num_cols_recent = X_recent.select_dtypes(include=[np.number]).columns
                X_recent[num_cols_recent] = X_recent[num_cols_recent].replace([np.inf, -np.inf], np.nan)
                X_recent = X_recent.fillna(train_means)

                if preprocessor is not None:
                    X_recent_proc = preprocessor.transform(X_recent)
                else:
                    X_recent_proc = X_recent.values

                recent_pred = model.predict(X_recent_proc)
                recent_df["PRED_SCORE"] = recent_pred

                val_metrics = evaluate_predictions(recent_df, true_col="WIN_PCT", pred_col="PRED_SCORE")
                print(
                    f"  Val - Exact: {val_metrics['exact_rank_acc']:.3f}, "
                    f"Within1: {val_metrics['within1_rank_acc']:.3f}"
                )

            all_holdout_results = []
            for target_season in HOLDOUT_SEASONS:
                season_df = test_feat_all[test_feat_all["Season"] == target_season].copy()
                if season_df.empty:
                    continue

                X_test = season_df[feature_cols].copy()
                num_cols_test = X_test.select_dtypes(include=[np.number]).columns
                X_test[num_cols_test] = X_test[num_cols_test].replace([np.inf, -np.inf], np.nan)
                X_test = X_test.fillna(train_means)

                if preprocessor is not None:
                    X_test_proc = preprocessor.transform(X_test)
                else:
                    X_test_proc = X_test.values

                test_pred = model.predict(X_test_proc)
                season_df["PRED_SCORE"] = test_pred

                eval_df = season_df.dropna(subset=["WIN_PCT"]).copy()
                if len(eval_df) == 0:
                    continue

                tm = evaluate_predictions(eval_df, true_col="WIN_PCT", pred_col="PRED_SCORE")
                tm["season"] = target_season
                all_holdout_results.append(tm)

            if all_holdout_results:
                avg_exact = np.mean([r["exact_rank_acc"] for r in all_holdout_results])
                avg_within1 = np.mean([r["within1_rank_acc"] for r in all_holdout_results])
                avg_within2 = np.mean([r["within2_rank_acc"] for r in all_holdout_results])
                avg_overall = np.mean([r["overall_score"] for r in all_holdout_results])

                print(
                    f"  Test - Exact: {avg_exact:.3f}, "
                    f"Within1: {avg_within1:.3f}, "
                    f"Within2: {avg_within2:.3f}, "
                    f"Overall: {avg_overall:.3f}"
                )

                entry = {
                    "iteration": iteration,
                    "feature_groups": "|".join(feature_groups),
                    "n_features": len(feature_cols),
                    "feature_cols": ",".join(feature_cols),
                    "model_type": model_config["type"],
                    "model_params": json.dumps(model_config["config"], sort_keys=True),
                    "preprocessor": preprocessor_config["type"],
                    "val_exact": float(val_metrics["exact_rank_acc"]) if val_metrics else 0.0,
                    "val_within1": float(val_metrics["within1_rank_acc"]) if val_metrics else 0.0,
                    "test_exact": float(avg_exact),
                    "test_within1": float(avg_within1),
                    "test_within2": float(avg_within2),
                    "test_overall": float(avg_overall),
                }
                for r in all_holdout_results:
                    season = r["season"]
                    entry[f"test_exact_{season}"] = float(r["exact_rank_acc"])
                    entry[f"test_within1_{season}"] = float(r["within1_rank_acc"])
                    entry[f"test_within2_{season}"] = float(r["within2_rank_acc"])
                    entry[f"test_overall_{season}"] = float(r["overall_score"])
                history.append(entry)

                if avg_overall > BEST_OVERALL_SCORE:
                    BEST_OVERALL_SCORE = avg_overall
                    best_results = {
                        "iteration": iteration,
                        "feature_groups": feature_groups,
                        "model_config": model_config,
                        "preprocessor_config": preprocessor_config,
                        "feature_cols": feature_cols,
                        "avg_exact": float(avg_exact),
                        "avg_within1": float(avg_within1),
                        "avg_overall": float(avg_overall),
                    }
                    print(f"  -> NEW GLOBAL BEST: Overall = {avg_overall:.4f}")

        except Exception as e:
            print(f"  Error in iteration {iteration}: {str(e)[:80]}")
            continue

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            avg_t = elapsed / iteration
            remain = avg_t * (MAX_ITERATIONS - iteration)
            print(f"\n  Progress: {iteration}/{MAX_ITERATIONS}")
            print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: ~{remain:.1f}s")
            print(f"  Best overall so far: {BEST_OVERALL_SCORE:.4f}")

    print("\n" + "=" * 60)
    print(f"Search finished. Best overall score: {BEST_OVERALL_SCORE:.4f}")
    if best_results:
        print(f"Best configuration found at iteration {best_results['iteration']}:")
        print(f"  Feature groups: {best_results['feature_groups']}")
        print(f"  Model: {best_results['model_config']['type']}")
        print(f"  Preprocessor: {best_results['preprocessor_config']['type']}")
        print(f"  Test Exact: {best_results['avg_exact']:.4f}")
        print(f"  Test Within1: {best_results['avg_within1']:.4f}")
        print(f"  Test Overall: {best_results['avg_overall']:.4f}")

    return best_results, history


# ========== Detailed predictions for best config ==========
def detailed_predictions_with_best(best_config):
    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS (GLOBAL BEST)")
    print("=" * 80)

    full_df = data.copy()
    train_mask = ~full_df["Season"].isin(HOLDOUT_SEASONS)
    test_mask = full_df["Season"].isin(HOLDOUT_SEASONS)
    train_raw = full_df[train_mask].copy()
    test_raw = full_df[test_mask].copy()

    train_feat = build_feature_matrix(train_raw)
    test_feat = build_feature_matrix(test_raw)

    # clean inf / NaN
    for df_ in (train_feat, test_feat):
        num_cols = df_.select_dtypes(include=[np.number]).columns
        df_[num_cols] = df_[num_cols].replace([np.inf, -np.inf], np.nan)
        df_.dropna(subset=["WIN_PCT"], inplace=True)

    feature_cols = best_config["feature_cols"]
    feature_groups = best_config["feature_groups"]
    model_config = best_config["model_config"]
    preprocessor_config = best_config["preprocessor_config"]

    print(f"\nUsing feature groups: {feature_groups}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Model: {model_config['type']} | Preprocessor: {preprocessor_config['type']}")

    # Train
    X_train = train_feat[feature_cols].copy()
    y_train = train_feat["WIN_PCT"].values

    num_cols_train = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols_train] = X_train[num_cols_train].replace([np.inf, -np.inf], np.nan)
    train_means = X_train.mean(axis=0)
    X_train = X_train.fillna(train_means)

    preprocessor = PreprocessorFactory.create_preprocessor(preprocessor_config)
    if preprocessor is not None:
        X_train_proc = preprocessor.fit_transform(X_train)
    else:
        X_train_proc = X_train.values

    final_model = ModelFactory.create_model(model_config)
    final_model.fit(X_train_proc, y_train)

    all_season_results = []

    for target_season in HOLDOUT_SEASONS:
        print(f"\n{'=' * 50}")
        print(f"SEASON: {target_season} (GLOBAL BEST)")
        print("=" * 50)

        season_df = test_feat[test_feat["Season"] == target_season].copy()
        if season_df.empty:
            continue

        X_test = season_df[feature_cols].copy()
        num_cols_test = X_test.select_dtypes(include=[np.number]).columns
        X_test[num_cols_test] = X_test[num_cols_test].replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(train_means)

        if preprocessor is not None:
            X_test_proc = preprocessor.transform(X_test)
        else:
            X_test_proc = X_test.values

        test_pred = final_model.predict(X_test_proc)
        season_df["PRED_SCORE"] = test_pred

        eval_df = season_df.dropna(subset=["WIN_PCT"]).copy()
        if len(eval_df) == 0:
            continue

        metrics = evaluate_predictions(eval_df, true_col="WIN_PCT", pred_col="PRED_SCORE")
        all_season_results.append(metrics)

        print("\nPerformance Metrics:")
        print(f"  Exact accuracy: {metrics['exact_rank_acc']:.4f}")
        print(f"  Within 1 rank: {metrics['within1_rank_acc']:.4f}")
        print(f"  Within 2 ranks: {metrics['within2_rank_acc']:.4f}")

        # 排名打印
        season_df["PRED_RANK"] = season_df["PRED_SCORE"].rank(method="min", ascending=False).astype(int)
        season_df["TRUE_RANK"] = season_df["WIN_PCT"].rank(method="min", ascending=False).astype(int)
        season_df["RANK_DIFF"] = season_df["PRED_RANK"] - season_df["TRUE_RANK"]

        season_sorted = season_df.sort_values("PRED_RANK")

        print("\nFull Ranking Predictions:")
        print("-" * 70)
        print(f"{'Pred':^5} {'Actual':^6} {'Diff':^6} {'Team':^20} {'Pred':^8} {'Actual':^8}")
        print("-" * 70)

        correct_within1 = 0
        for _, row in season_sorted.iterrows():
            rd = row["RANK_DIFF"]
            diff_symbol = "✓" if abs(rd) <= 1 else "✗"
            if abs(rd) <= 1:
                correct_within1 += 1
            print(
                f"{row['PRED_RANK']:^5} {row['TRUE_RANK']:^6} {f'{diff_symbol}{rd:+2d}':^6} "
                f"{row['Team']:20s} {row['PRED_SCORE']:^8.3f} {row['WIN_PCT']:^8.3f}"
            )

        print("-" * 70)
        print(f"Correct within 1 rank: {correct_within1}/{len(season_sorted)} "
              f"= {correct_within1 / len(season_sorted):.2%}")

        out_dir = "best_predictions"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"global_best_{target_season}_detailed.csv")
        season_sorted.to_csv(out_path, index=False)
        print(f"\nDetailed predictions saved to: {out_path}")

    return all_season_results


# ========== Analyze optimization history ==========
def analyze_history(history):
    if not history:
        print("No history to analyze.")
        return

    print("\n" + "=" * 80)
    print("OPTIMIZATION HISTORY ANALYSIS")
    print("=" * 80)

    history_df = pd.DataFrame(history)

    print("\nBy Feature Groups:")
    print("-" * 40)
    for fg, group in history_df.groupby("feature_groups"):
        print(f"\n{fg}:")
        print(f"  Count: {len(group)}")
        print(f"  Avg Test Exact: {group['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {group['test_within1'].mean():.4f}")
        print(f"  Avg Test Within2: {group['test_within2'].mean():.4f}")
        print(f"  Best Overall: {group['test_overall'].max():.4f}")

    print("\nBy Model Type:")
    print("-" * 40)
    for mt, group in history_df.groupby("model_type"):
        print(f"\n{mt}:")
        print(f"  Count: {len(group)}")
        print(f"  Avg Test Exact: {group['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {group['test_within1'].mean():.4f}")
        print(f"  Avg Test Within2: {group['test_within2'].mean():.4f}")

    print("\nTop 10 Best Performances (by overall):")
    print("-" * 40)
    top = history_df.sort_values("test_overall", ascending=False).head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"{i:2d}. Iter {int(row['iteration']):4d}: "
            f"FeatGroups={row['feature_groups']:25s} "
            f"Model={row['model_type']:8s} "
            f"Exact={row['test_exact']:.3f} "
            f"Within1={row['test_within1']:.3f} "
            f"Within2={row['test_within2']:.3f} "
            f"Overall={row['test_overall']:.3f}"
        )

    results_dir = "search_results"
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    csv_path_all = os.path.join(results_dir, f"model_search_history_{ts}.csv")
    history_df.to_csv(csv_path_all, index=False, encoding="utf-8")
    print(f"\nFull model search history saved to: {csv_path_all}")

    filtered = history_df[history_df["test_exact"] > 0.4].copy()
    if not filtered.empty:
        if "test_overall" in filtered.columns:
            filtered = filtered.drop(columns=["test_overall"])
        csv_path_good = os.path.join(results_dir, f"model_search_history_exact_gt40_{ts}.csv")
        filtered.to_csv(csv_path_good, index=False, encoding="utf-8")
        print(f"Filtered (test_exact > 0.4) history saved to: {csv_path_good}")
    else:
        print("No runs with test_exact > 0.4, so no filtered CSV created.")


# ========== Main ==========
if __name__ == "__main__":
    best_config, history = auto_optimize()

    analyze_history(history)

    if best_config:
        detailed_predictions_with_best(best_config)

        config_dir = "best_config"
        os.makedirs(config_dir, exist_ok=True)

        best_summary = {
            "iteration": best_config["iteration"],
            "feature_groups": best_config["feature_groups"],
            "feature_cols": best_config["feature_cols"],
            "model_type": best_config["model_config"]["type"],
            "model_params": best_config["model_config"]["config"],
            "preprocessor": best_config["preprocessor_config"]["type"],
            "avg_exact": best_config["avg_exact"],
            "avg_within1": best_config["avg_within1"],
            "avg_overall": best_config["avg_overall"],
            "holdout_seasons": HOLDOUT_SEASONS,
        }

        best_path = os.path.join(config_dir, "best_config_summary.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)
        print(f"\nBest config summary saved to: {best_path}")
