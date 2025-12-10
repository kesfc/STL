**NBA Team Performance Prediction**

This repository contains code for predicting NBA team wins using two approaches:

Team-only baseline model

Uses historical team win/loss records to predict next-season wins.

Player-based forecasting model

Builds player-level statistics across seasons, predicts future player performance, and aggregates player projections to estimate team strength.


**Files**

baseline_only_team.py – simple standings-based model

get_player_stats.py – processes raw player statistics

combine_all_seaon.py / combine_team.py – merges multi-season player and team data

get_WL.py – extracts win/loss labels

train.py – trains player evolution models

train_and_predict_player_for_team.py – uses player predictions to forecast team wins

scoring.py – evaluation functions

The repository also includes CSV files and PNG plots that summarize trends such as scoring by age and experience.

The paper/ folder contains the written report for the project.


**How to Run**

Prepare the data (player and team stats).

Run get_player_stats.py and the data combination scripts.

Train the model using train.py.

Generate predictions with train_and_predict_player_for_team.py.

For comparison, run the baseline model with baseline_only_team.py.

**Summary**

The project compares a simple team-based baseline with a more detailed player-based pipeline.
The player model performs better for teams with major roster changes, while the team baseline performs well for stable rosters.


**Contributors**

Shijie Chen

Weihao Li

Shuhao Gao

Anbang Liu
