import os
import glob
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ========== Use the combined team summary CSV directly ==========
summary_path = "Team_stats/all_seasons_team_summary.csv"  # Path to the combined team summary CSV file

# ========== Data loading ==========
print(f"Loading data from: {summary_path}")
data = pd.read_csv(summary_path)

# Ensure wins and losses information is present
if 'WINS' not in data.columns or 'LOSSES' not in data.columns:
    raise ValueError("CSV must contain 'WINS' and 'LOSSES' columns")

# Drop rows with missing win/loss values
data = data.dropna(subset=["WINS", "LOSSES"])

# Sort and compute basic win-rate metrics
data = data.sort_values(["Team", "SeasonEndYear"])
data["WIN_PCT"] = data["WINS"] / (data["WINS"] + data["LOSSES"])
data["PREV_WIN_PCT"] = data.groupby("Team")["WIN_PCT"].shift(1)

REG_SEASON_GAMES = 82
# Use the most recent seasons as holdout
latest_season = data["Season"].max()
second_latest_season = sorted(data["Season"].unique())[-2] if len(data["Season"].unique()) > 1 else latest_season
HOLDOUT_SEASONS = [second_latest_season, latest_season]  # Use the latest two seasons as test / holdout

TARGET_EXACT = 0.8  # Target exact ranking accuracy
TARGET_WITHIN1 = 0.8  # Target ranking accuracy within 1 position
MAX_ITERATIONS = 200  # Maximum number of random search iterations
BEST_OVERALL_SCORE = -1  # Best overall score observed so far

print(f"Data loaded: {len(data)} rows, {data['Season'].nunique()} seasons")
print(f"Holdout seasons: {HOLDOUT_SEASONS}")
print(f"Training seasons: {sorted(set(data['Season']) - set(HOLDOUT_SEASONS))}")


# ========== Simplified feature factory ==========
class FeatureFactory:
    """Factory class that generates different feature sets."""

    @staticmethod
    def get_all_methods():
        """Return all available feature methods."""
        return [
            {'name': 'basic', 'func': FeatureFactory.create_basic_features},
            {'name': 'advanced', 'func': FeatureFactory.create_advanced_features},
            {'name': 'stats_only', 'func': FeatureFactory.create_stats_features},
            {'name': 'trend_only', 'func': FeatureFactory.create_trend_features},
        ]

    @staticmethod
    def create_basic_features(df, is_training=True, train_df=None):
        """Create a basic feature set."""
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        # Basic box-score stats
        basic_stats = ["PTS", "FGA", "FTA", "TRB", "AST", "STL", "BLK", "TOV", "PF"]

        # Simple efficiency features
        df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"] + 1e-8))
        df["EFG_PCT"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1e-8)
        df["AST_TO_RATIO"] = df["AST"] / (df["TOV"] + 1e-8)

        feature_cols = basic_stats + ["TS_PCT", "EFG_PCT", "AST_TO_RATIO", "PREV_WIN_PCT"]

        return df, feature_cols

    @staticmethod
    def create_advanced_features(df, is_training=True, train_df=None):
        """Create a more advanced feature set."""
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        # Four factors-style features
        df["EFG_FACTOR"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1e-8)
        df["TOV_FACTOR"] = 1 - df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"] + 1e-8)
        df["OREB_FACTOR"] = df["ORB"] / (df["ORB"] + df["DRB"] + 1e-8)
        df["FTR_FACTOR"] = df["FTA"] / (df["FGA"] + 1e-8)

        # Time-series rolling stats per team
        df = df.sort_values(["Team", "SeasonEndYear"])
        for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
            if col in df.columns:
                df[f"{col}_3Y_AVG"] = df.groupby("Team")[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )

        # Win-percentage trend features
        df["WIN_PCT_3Y_AVG"] = df.groupby("Team")["WIN_PCT"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df["WIN_PCT_TREND"] = df.groupby("Team")["WIN_PCT"].transform(
            lambda x: x.rolling(3, min_periods=1).mean().diff()
        )

        feature_cols = [
            "PREV_WIN_PCT", "EFG_FACTOR", "TOV_FACTOR", "OREB_FACTOR", "FTR_FACTOR",
            "WIN_PCT_3Y_AVG", "WIN_PCT_TREND"
        ]

        for col in ["PTS", "AST", "TRB", "STL", "BLK"]:
            feature_cols.append(f"{col}_3Y_AVG")

        return df, feature_cols

    @staticmethod
    def create_stats_features(df, is_training=True, train_df=None):
        """Use only raw statistical features (box-score totals)."""
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        # All basic counting stats
        all_stats = ["FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA",
                     "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

        feature_cols = [col for col in all_stats if col in df.columns]
        feature_cols.append("PREV_WIN_PCT")

        return df, feature_cols

    @staticmethod
    def create_trend_features(df, is_training=True, train_df=None):
        """Use only trend-based features derived from win percentage."""
        df = df.copy()

        df["WIN_PCT"] = df["WINS"] / (df["WINS"] + df["LOSSES"])
        df["PREV_WIN_PCT"] = df.groupby("Team")["WIN_PCT"].shift(1)

        df = df.sort_values(["Team", "SeasonEndYear"])

        # Rolling win-percentage and trend features
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


# ========== Simplified model factory ==========
class ModelFactory:
    """Factory class that generates different model configurations."""

    @staticmethod
    def get_all_configs():
        """Return all model configurations for random search."""
        configs = []

        # LightGBM configs (several combinations)
        for n_est in [50, 100, 200, 300]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7, 10]:
                    configs.append({
                        'type': 'lgb',
                        'config': {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
                    })

        # XGBoost configs
        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        'type': 'xgb',
                        'config': {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
                    })

        # Random Forest configs
        for n_est in [50, 100, 200, 300]:
            for depth in [5, 10, 15, 20]:
                configs.append({
                    'type': 'rf',
                    'config': {'n_estimators': n_est, 'max_depth': depth}
                })

        # Gradient Boosting configs
        for n_est in [50, 100, 200]:
            for lr in [0.01, 0.05, 0.1]:
                for depth in [3, 5, 7]:
                    configs.append({
                        'type': 'gb',
                        'config': {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
                    })

        # Linear models (Ridge and Lasso with different alphas)
        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
            configs.append({'type': 'ridge', 'config': {'alpha': alpha}})
            configs.append({'type': 'lasso', 'config': {'alpha': alpha}})

        # Neural network configs
        for layers in [(50,), (100,), (50, 25), (100, 50)]:
            for alpha in [0.0001, 0.001, 0.01]:
                configs.append({
                    'type': 'nn',
                    'config': {'hidden_layer_sizes': layers, 'alpha': alpha}
                })

        # SVM configs
        for C in [0.1, 1.0, 10.0]:
            for kernel in ['linear', 'rbf']:
                configs.append({
                    'type': 'svm',
                    'config': {'C': C, 'kernel': kernel}
                })

        # KNN configs
        for n_neighbors in [3, 5, 7, 10]:
            configs.append({
                'type': 'knn',
                'config': {'n_neighbors': n_neighbors}
            })

        # Simple ensemble configs (VotingRegressor over a few base models)
        ensemble_configs = [
            {'models': ['lgb', 'rf', 'ridge'], 'weights': [0.4, 0.3, 0.3]},
            {'models': ['lgb', 'xgb', 'gb'], 'weights': [0.5, 0.3, 0.2]},
            {'models': ['rf', 'gb', 'ridge'], 'weights': [0.4, 0.4, 0.2]},
            {'models': ['lgb', 'rf', 'gb', 'ridge'], 'weights': [0.4, 0.2, 0.2, 0.2]},
        ]

        for config in ensemble_configs:
            configs.append({'type': 'ensemble', 'config': config})

        return configs

    @staticmethod
    def create_model(model_config):
        """Create a model instance from a configuration dictionary."""
        model_type = model_config['type']
        config = model_config['config']

        if model_type == 'lgb':
            return lgb.LGBMRegressor(
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                max_depth=config['max_depth'],
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif model_type == 'xgb':
            return xgb.XGBRegressor(
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                max_depth=config['max_depth'],
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            return GradientBoostingRegressor(
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                max_depth=config['max_depth'],
                random_state=42
            )
        elif model_type == 'nn':
            return MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                random_state=42,
                max_iter=1000,
                early_stopping=True
            )
        elif model_type == 'svm':
            return SVR(C=config['C'], kernel=config['kernel'])
        elif model_type == 'ridge':
            return Ridge(alpha=config['alpha'], random_state=42)
        elif model_type == 'lasso':
            return Lasso(alpha=config['alpha'], random_state=42)
        elif model_type == 'knn':
            return KNeighborsRegressor(n_neighbors=config['n_neighbors'])
        elif model_type == 'ensemble':
            # Build a VotingRegressor with the requested base models
            estimators = []
            for i, model_name in enumerate(config['models']):
                if model_name == 'lgb':
                    estimators.append(('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == 'xgb':
                    estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == 'rf':
                    estimators.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == 'gb':
                    estimators.append(('gb', GradientBoostingRegressor(n_estimators=100, random_state=42 + i)))
                elif model_name == 'ridge':
                    estimators.append(('ridge', Ridge(alpha=1.0, random_state=42 + i)))

            return VotingRegressor(estimators=estimators, weights=config['weights'])


# ========== Preprocessor factory ==========
class PreprocessorFactory:
    """Factory class for creating data preprocessors."""

    @staticmethod
    def get_all_configs():
        """Return all preprocessor configurations."""
        return [
            {'type': 'standard'},
            {'type': 'robust'},
            {'type': 'minmax'},
            {'type': 'none'},
        ]

    @staticmethod
    def create_preprocessor(preprocessor_config):
        """Create a scaler / preprocessor instance."""
        preprocessor_type = preprocessor_config['type']

        if preprocessor_type == 'standard':
            return StandardScaler()
        elif preprocessor_type == 'robust':
            return RobustScaler()
        elif preprocessor_type == 'minmax':
            return MinMaxScaler()
        else:
            return None  # No preprocessing


# ========== Evaluation function ==========
def evaluate_predictions(df, true_col="WIN_PCT", pred_col="PRED_SCORE"):
    """Evaluate ranking predictions (exact match and within-k accuracy)."""
    df = df.copy()
    df = df.sort_values(true_col, ascending=False)
    df["TRUE_RANK"] = np.arange(1, len(df) + 1)
    df["PRED_RANK"] = df[pred_col].rank(method="min", ascending=False).astype(int)
    df["RANK_DIFF"] = df["PRED_RANK"] - df["TRUE_RANK"]

    exact_acc = (df["PRED_RANK"] == df["TRUE_RANK"]).mean()
    within1_acc = (df["RANK_DIFF"].abs() <= 1).mean()
    within2_acc = (df["RANK_DIFF"].abs() <= 2).mean()

    # Overall score is a weighted combination of exact and within-1 accuracy
    overall_score = within1_acc * 0.7 + exact_acc * 0.3

    return {
        "exact_rank_acc": exact_acc,
        "within1_rank_acc": within1_acc,
        "within2_rank_acc": within2_acc,
        "overall_score": overall_score
    }


# ========== Main auto-optimization loop ==========
def auto_optimize():
    """Main loop for random search and automatic configuration optimization."""
    global BEST_OVERALL_SCORE

    print("=" * 80)
    print("NBA RANKING AUTO-OPTIMIZER")
    print(f"Target: Exact >= {TARGET_EXACT * 100:.0f}%, Within1 >= {TARGET_WITHIN1 * 100:.0f}%")
    print("=" * 80)

    # Train / test split by season
    train_df = data[~data["Season"].isin(HOLDOUT_SEASONS)].copy()
    test_df = data[data["Season"].isin(HOLDOUT_SEASONS)].copy()

    print(f"Training seasons: {train_df['Season'].nunique()} ({len(train_df)} rows)")
    print(f"Test seasons: {test_df['Season'].nunique()} ({len(test_df)} rows)")

    # Get all candidates
    feature_methods = FeatureFactory.get_all_methods()
    model_configs = ModelFactory.get_all_configs()
    preprocessor_configs = PreprocessorFactory.get_all_configs()

    print(f"\nAvailable configurations:")
    print(f"  Feature methods: {len(feature_methods)}")
    print(f"  Model configs: {len(model_configs)}")
    print(f"  Preprocessor configs: {len(preprocessor_configs)}")
    print(f"  Total combinations: {len(feature_methods) * len(model_configs) * len(preprocessor_configs):,}")

    iteration = 0
    best_results = {}
    start_time = time.time()

    # History of experiments
    history = []

    # Main random search loop
    while iteration < MAX_ITERATIONS:
        iteration += 1

        # Randomly sample a configuration
        feature_method = random.choice(feature_methods)
        model_config = random.choice(model_configs)
        preprocessor_config = random.choice(preprocessor_configs)

        print(f"\n[Iteration {iteration:3d}/{MAX_ITERATIONS}]")
        print(f"  Features: {feature_method['name']}")
        print(f"  Model: {model_config['type']}")
        print(f"  Preprocessor: {preprocessor_config['type']}")

        try:
            # 1. Feature engineering on the training set
            train_feat, feature_cols = feature_method['func'](train_df, is_training=True)
            print(f"  Features: {len(feature_cols)}")

            # 2. Prepare X/y for training
            X_train = train_feat[feature_cols].values
            y_train = train_feat["WIN_PCT"].values

            # 3. Optional preprocessing
            preprocessor = PreprocessorFactory.create_preprocessor(preprocessor_config)
            if preprocessor is not None:
                X_train = preprocessor.fit_transform(X_train)

            # 4. Train the model
            model = ModelFactory.create_model(model_config)
            model.fit(X_train, y_train)

            # 5. Validate on the most recent 3 years of the training data
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
                print(
                    f"  Val - Exact: {val_metrics['exact_rank_acc']:.3f}, Within1: {val_metrics['within1_rank_acc']:.3f}")

            # 6. Evaluate on each holdout season
            all_holdout_results = []
            for target_season in HOLDOUT_SEASONS:
                test_season_df = test_df[test_df["Season"] == target_season].copy()
                if test_season_df.empty:
                    continue

                # Create test features using the same method
                test_feat, _ = feature_method['func'](test_season_df, is_training=False, train_df=train_df)

                # Ensure feature columns are aligned between train and test
                for col in feature_cols:
                    if col not in test_feat.columns:
                        test_feat[col] = 0

                X_test = test_feat[feature_cols].values
                if preprocessor is not None:
                    X_test = preprocessor.transform(X_test)

                # Predict on holdout
                test_pred = model.predict(X_test)
                test_feat["PRED_SCORE"] = test_pred

                # Evaluate ranking quality
                test_metrics = evaluate_predictions(test_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
                all_holdout_results.append(test_metrics)

            # 7. Aggregate metrics across all holdout seasons
            if all_holdout_results:
                avg_exact = np.mean([r["exact_rank_acc"] for r in all_holdout_results])
                avg_within1 = np.mean([r["within1_rank_acc"] for r in all_holdout_results])
                avg_overall = np.mean([r["overall_score"] for r in all_holdout_results])

                print(f"  Test - Exact: {avg_exact:.3f}, Within1: {avg_within1:.3f}, Overall: {avg_overall:.3f}")

                # Log this iteration to history
                history.append({
                    'iteration': iteration,
                    'feature_method': feature_method['name'],
                    'model_type': model_config['type'],
                    'preprocessor': preprocessor_config['type'],
                    'val_exact': val_metrics['exact_rank_acc'] if val_metrics else 0,
                    'val_within1': val_metrics['within1_rank_acc'] if val_metrics else 0,
                    'test_exact': avg_exact,
                    'test_within1': avg_within1,
                    'test_overall': avg_overall
                })

                # Check if the current configuration meets target thresholds
                if avg_exact >= TARGET_EXACT and avg_within1 >= TARGET_WITHIN1:
                    print(f"\n{'=' * 60}")
                    print("TARGET ACHIEVED!")
                    print(f"Exact: {avg_exact:.4f} >= {TARGET_EXACT}")
                    print(f"Within1: {avg_within1:.4f} >= {TARGET_WITHIN1}")

                    # Save the best configuration found
                    best_results = {
                        'iteration': iteration,
                        'feature_method': feature_method,
                        'model_config': model_config,
                        'preprocessor_config': preprocessor_config,
                        'feature_cols': feature_cols,
                        'preprocessor': preprocessor,
                        'model': model,
                        'avg_exact': avg_exact,
                        'avg_within1': avg_within1,
                        'avg_overall': avg_overall,
                        'train_df': train_df,
                        'test_df': test_df
                    }

                    return best_results, history

                # Update global best if this configuration is better
                if avg_overall > BEST_OVERALL_SCORE:
                    BEST_OVERALL_SCORE = avg_overall
                    best_results = {
                        'iteration': iteration,
                        'feature_method': feature_method,
                        'model_config': model_config,
                        'preprocessor_config': preprocessor_config,
                        'feature_cols': feature_cols,
                        'preprocessor': preprocessor,
                        'model': model,
                        'avg_exact': avg_exact,
                        'avg_within1': avg_within1,
                        'avg_overall': avg_overall,
                        'train_df': train_df,
                        'test_df': test_df
                    }

                    print(f"NEW BEST: Overall = {avg_overall:.4f}")

        except Exception as e:
            # Catch any error during feature creation, training, or evaluation
            print(f"Error: {str(e)[:50]}")
            continue

        # Show progress every 10 iterations with rough time estimates
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_iter = elapsed_time / iteration
            remaining_iter = MAX_ITERATIONS - iteration
            estimated_remaining = avg_time_per_iter * remaining_iter

            print(f"\n  Progress: {iteration}/{MAX_ITERATIONS}")
            print(f"  Elapsed: {elapsed_time:.1f}s")
            print(f"  Remaining: ~{estimated_remaining:.1f}s")
            print(f"  Best overall: {BEST_OVERALL_SCORE:.4f}")

    # If maximum iterations are reached without hitting the target, return the best so far
    print(f"\n{'=' * 60}")
    print(f"Maximum iterations reached ({MAX_ITERATIONS}) without achieving target.")
    print(f"Best overall score: {BEST_OVERALL_SCORE:.4f}")

    if best_results:
        print(f"\nBest configuration found at iteration {best_results['iteration']}:")
        print(f"  Features: {best_results['feature_method']['name']}")
        print(f"  Model: {best_results['model_config']['type']}")
        print(f"  Preprocessor: {best_results['preprocessor_config']['type']}")
        print(f"  Test Exact: {best_results['avg_exact']:.4f}")
        print(f"  Test Within1: {best_results['avg_within1']:.4f}")
        print(f"  Test Overall: {best_results['avg_overall']:.4f}")

    return best_results, history


# ========== Use the best model for detailed predictions ==========
def detailed_predictions_with_best(best_config):
    """Run detailed predictions using the best configuration and print full rankings."""
    print("\n" + "=" * 80)
    print("DETAILED PREDICTIONS WITH BEST MODEL")
    print("=" * 80)

    train_df = best_config['train_df']
    test_df = best_config['test_df']

    # Retrain the best model on the full training data
    print(f"\nRetraining best model on all training data...")

    # Feature engineering using the best feature method
    train_feat, feature_cols = best_config['feature_method']['func'](train_df, is_training=True)

    X_train = train_feat[feature_cols].values
    y_train = train_feat["WIN_PCT"].values

    # Apply preprocessor (re-fit on full training data)
    preprocessor = best_config['preprocessor']
    if preprocessor is not None:
        X_train = preprocessor.fit_transform(X_train)

    # Train final model instance from the best config
    final_model = ModelFactory.create_model(best_config['model_config'])
    final_model.fit(X_train, y_train)

    # Generate detailed predictions for each holdout season
    all_season_results = []

    for target_season in HOLDOUT_SEASONS:
        print(f"\n{'=' * 50}")
        print(f"SEASON: {target_season}")
        print('=' * 50)

        test_season_df = test_df[test_df["Season"] == target_season].copy()
        if test_season_df.empty:
            continue

        # Build features for this season
        test_feat, _ = best_config['feature_method']['func'](test_season_df, is_training=False, train_df=train_df)

        # Make sure test features contain all columns used in training
        for col in feature_cols:
            if col not in test_feat.columns:
                test_feat[col] = 0

        X_test = test_feat[feature_cols].values
        if preprocessor is not None:
            X_test = preprocessor.transform(X_test)

        # Predict ranking score
        test_pred = final_model.predict(X_test)
        test_feat["PRED_SCORE"] = test_pred

        # Evaluate season-level ranking metrics
        metrics = evaluate_predictions(test_feat, true_col="WIN_PCT", pred_col="PRED_SCORE")
        all_season_results.append(metrics)

        print(f"\nPerformance Metrics:")
        print(f"  Exact accuracy: {metrics['exact_rank_acc']:.4f}")
        print(f"  Within 1 rank: {metrics['within1_rank_acc']:.4f}")
        print(f"  Within 2 ranks: {metrics['within2_rank_acc']:.4f}")

        # Add ranking columns for summary printing
        test_feat["PRED_RANK"] = test_feat["PRED_SCORE"].rank(method="min", ascending=False).astype(int)
        test_feat["TRUE_RANK"] = test_feat["WIN_PCT"].rank(method="min", ascending=False).astype(int)
        test_feat["RANK_DIFF"] = test_feat["PRED_RANK"] - test_feat["TRUE_RANK"]

        # Sort by predicted rank
        test_feat_sorted = test_feat.sort_values("PRED_RANK")

        print(f"\nFull Ranking Predictions:")
        print("-" * 70)
        print(f"{'Pred':^5} {'Actual':^6} {'Diff':^6} {'Team':^20} {'Pred':^8} {'Actual':^8}")
        print("-" * 70)

        correct_count = 0
        for _, row in test_feat_sorted.iterrows():
            rank_diff = row["RANK_DIFF"]
            diff_symbol = "✓" if abs(rank_diff) <= 1 else "✗"
            if abs(rank_diff) <= 1:
                correct_count += 1

            print(f"{row['PRED_RANK']:^5} {row['TRUE_RANK']:^6} {f'{diff_symbol}{rank_diff:+2d}':^6} "
                  f"{row['Team']:20s} {row['PRED_SCORE']:^8.3f} {row['WIN_PCT']:^8.3f}")

        print("-" * 70)
        print(f"Correct within 1 rank: {correct_count}/{len(test_feat)} = {correct_count / len(test_feat):.2%}")

        # Save detailed per-season predictions to CSV
        pred_dir = "best_predictions"
        os.makedirs(pred_dir, exist_ok=True)

        output_path = os.path.join(pred_dir, f"{target_season}_detailed.csv")
        test_feat.to_csv(output_path, index=False)
        print(f"\nDetailed predictions saved to: {output_path}")

    # Compute and print average performance across all holdout seasons
    if all_season_results:
        avg_exact = np.mean([r["exact_rank_acc"] for r in all_season_results])
        avg_within1 = np.mean([r["within1_rank_acc"] for r in all_season_results])

        print("\n" + "=" * 80)
        print("FINAL AVERAGE RESULTS")
        print("=" * 80)
        print(f"Average Exact accuracy: {avg_exact:.4f}")
        print(f"Average Within 1 rank: {avg_within1:.4f}")

        if avg_exact >= TARGET_EXACT and avg_within1 >= TARGET_WITHIN1:
            print(f"\nSUCCESS: Both targets achieved! 🎉")
        else:
            print(f"\nTargets not fully achieved:")
            print(f"  Exact: {avg_exact:.2%} (target: {TARGET_EXACT * 100:.0f}%)")
            print(f"  Within1: {avg_within1:.2%} (target: {TARGET_WITHIN1 * 100:.0f}%)")


# ========== Analyze optimization history ==========
def analyze_history(history):
    """Analyze the historical optimization results from the random search."""
    if not history:
        return

    print("\n" + "=" * 80)
    print("OPTIMIZATION HISTORY ANALYSIS")
    print("=" * 80)

    history_df = pd.DataFrame(history)

    # Group by feature method
    print("\nBy Feature Method:")
    print("-" * 40)
    for method, group in history_df.groupby('feature_method'):
        print(f"\n{method}:")
        print(f"  Count: {len(group)}")
        print(f"  Avg Test Exact: {group['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {group['test_within1'].mean():.4f}")
        print(f"  Best Overall: {group['test_overall'].max():.4f}")

    # Group by model type
    print("\nBy Model Type:")
    print("-" * 40)
    for model_type, group in history_df.groupby('model_type'):
        print(f"\n{model_type}:")
        print(f"  Count: {len(group)}")
        print(f"  Avg Test Exact: {group['test_exact'].mean():.4f}")
        print(f"  Avg Test Within1: {group['test_within1'].mean():.4f}")

    # Top-10 configurations by overall test score
    print("\nTop 10 Best Performances:")
    print("-" * 40)
    top_results = history_df.sort_values('test_overall', ascending=False).head(10)
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i:2d}. Iter {row['iteration']:3d}: "
              f"Feat={row['feature_method']:10s} "
              f"Model={row['model_type']:10s} "
              f"Exact={row['test_exact']:.3f} "
              f"Within1={row['test_within1']:.3f} "
              f"Overall={row['test_overall']:.3f}")


# ========== Main entry point ==========
if __name__ == "__main__":
    # Run the auto-optimization routine
    best_config, history = auto_optimize()

    # Analyze search history
    analyze_history(history)

    if best_config:
        # Run detailed predictions with the best configuration
        detailed_predictions_with_best(best_config)

        # Save best configuration info
        config_dir = "best_config"
        os.makedirs(config_dir, exist_ok=True)
