# Predicting the NBA Playoffs

A machine learning model that predicts NBA playoff series outcomes from regular-season team stats, then simulates the full bracket from the first round to the Finals.

The 2025-26 predictions are also available as an [ESPN bracket entry](https://fantasy.espn.com/free-prize-games/sharer?challengeId=281&from=espn&context=FINAL_FOUR&entryId=258e93d0-3b41-11f1-8477-87400cd67949&game=null&rcIds=1.1,1.2,2.1,2.2,1.1,2.1,1.1&teamIds=25,24,8,2,25,8,25).

## Table of Contents

- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Get Started](#get-started)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)

## Motivation

I wanted to build an ML model that could predict the NBA playoffs. Playoff basketball is a good prediction problem: the matchups are known in advance, the outcomes are binary, and there is decades of detailed, public data. It is also a hard one, because small samples, injuries, and variance mean even the best teams can lose. The goal is to see how far regular-season team quality and recent form can take a model, and to test it against what actually happened.

## Project Structure

```
├── nba_data_collection.py   # Collects team stats, game logs and playoff results
├── train_h2h_model.py       # Builds head-to-head data, tunes and trains XGBoost
├── simulate_bracket.py      # Simulates a full playoff bracket with the trained model
├── datasets/                # Collected and generated CSVs (committed)
├── models/                  # Trained models (git-ignored, created by training)
└── pyproject.toml           # Dependencies (managed with uv)
```

## Get Started

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Collect data. This makes ~900 API calls and takes ~30 minutes.
uv run nba_data_collection.py

# 3. Train the model (runs a grid search; writes the models to models/)
uv run train_h2h_model.py

# 4. Simulate the bracket
uv run simulate_bracket.py
```

To simulate a different bracket, edit `SEASON`, `EAST_BRACKET` and `WEST_BRACKET` at the top of `simulate_bracket.py`. Team names must match the `TEAM_NAME` column in the dataset (for example, `LA Clippers`, not `Los Angeles Clippers`).

## Methodology

### 1. Data collection

`nba_data_collection.py` gathers three sources for the 2015-16 through 2025-26 seasons:

| Source | Data |
| --- | --- |
| [`nba_api`](https://github.com/swar/nba_api) | Regular-season advanced and base team stats, Per-team game logs (used for momentum features) |
| [Basketball Reference](https://www.basketball-reference.com) | Playoff series winners and losers for 2015-16 through 2024-25 |

The 2015-16 through 2024-25 seasons were used, because the style of basketball remains consistent with how its currently being played (2026).

### 2. Feature engineering

Twelve features describe each team:

- **Season quality (z-scored within each season):** net rating, offensive rating, defensive rating, true shooting %, assist-to-turnover ratio, rebound %, pace, win %, points per game.
- **Momentum (last 20 regular-season games):** win %, points per game, plus/minus.

Season stats are z-scored within each season, so teams are compared to their own league. This keeps the numbers comparable across years, as scoring and pace have shifted over the decade.

### 3. Head-to-head formulation

The model predicts a single series, not a championship. For every playoff series, the features of both teams are subtracted (`Team A − Team B`) to give one `_DIFF` value per feature. Each series is stored twice, once from each team's perspective, with a target of `1` if Team A won and `0` otherwise. The mirrored rows keep the classes balanced and stop the model from favoring "Team A" by position.

This produces 300 rows from 150 series (10 seasons × 15 series).

### 4. Model and tuning

An `XGBClassifier` is tuned with a 5-fold `GridSearchCV` over 972 hyperparameter combinations (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`), scoring on accuracy. Shallow trees (`max_depth=2`) win consistently, which suits a small dataset.

### 5. Evaluation

The split is chronological to avoid leaking future information: train on 2015-16 through 2023-24, test on 2024-25. Once the best settings are found, a final model is retrained on **all** available seasons and used for inference.

### 6. Bracket simulation

`simulate_bracket.py` takes the 16 playoff teams, predicts each series with the final model using that team's 2025-26 regular-season stats, and advances the winner to the next round until a champion remains.

## Results

### Model performance

Held-out accuracy on the 2024-25 playoffs, where the model predicts which team wins each series. The main metric is the tuned model's accuracy.

| Iteration | CV accuracy | Test accuracy | Notes |
| --- | --- | --- | --- |
| Baseline XGBoost (no tuning) | n/a | 61.7% | Tested on 60 rows (a larger test split) |
| + Hyperparameter tuning | 72.1% | 75.0% | Tested on 60 rows |
| + Extra features (clutch net rating, opponent 3P%, top-2 PIE) | 70.0% | 63.3% | Worse, so these features were dropped |
| **Final (tuned, chronological split)** | **73.3%** | **66.7%** | Tested on 30 rows (15 series × 2) |

Treat these numbers cautiously. With only 15 series in the final test set, one series changes accuracy by about 6.7 points. The test sets also differ between iterations, so the earlier rows are not directly comparable to the final one.

**Most important features (final model):** net rating, win %, true shooting %, offensive rating, and 20-game plus/minus. Season-long efficiency counts for far more than recent form.

### 2025-26 playoffs: predicted vs. actual

The Oklahoma City Thunder were predicted to win. The New York Knicks won, beating the San Antonio Spurs 4-1.

**Eastern Conference**

| Round | Predicted | Actual | Result |
| --- | --- | --- | --- |
| First Round | Pistons over Magic (72.0%) | Pistons over Magic, 4-3 | ✅ |
| First Round | Cavaliers over Raptors (63.7%) | Cavaliers over Raptors, 4-3 | ✅ |
| First Round | Knicks over Hawks (57.6%) | Knicks over Hawks, 4-2 | ✅ |
| First Round | Celtics over 76ers (73.6%) | 76ers over Celtics, 4-3 | ❌ |
| Semifinals | Pistons over Cavaliers (62.5%) | Cavaliers over Pistons, 4-3 | ❌ |
| Semifinals | Celtics over Knicks (55.5%) | Knicks over 76ers, 4-0 | ❌ |
| Finals | Pistons over Celtics (50.3%) | Knicks over Cavaliers, 4-0 | ❌ |

**Western Conference**

| Round | Predicted | Actual | Result |
| --- | --- | --- | --- |
| First Round | Thunder over Suns (80.8%) | Thunder over Suns, 4-0 | ✅ |
| First Round | Lakers over Rockets (62.7%) | Lakers over Rockets, 4-2 | ✅ |
| First Round | Nuggets over Timberwolves (75.6%) | Timberwolves over Nuggets, 4-2 | ❌ |
| First Round | Spurs over Trail Blazers (81.0%) | Spurs over Trail Blazers, 4-1 | ✅ |
| Semifinals | Thunder over Lakers (62.4%) | Thunder over Lakers, 4-0 | ✅ |
| Semifinals | Spurs over Nuggets (53.0%) | Spurs over Timberwolves, 4-2 | ✅ (right winner, different opponent) |
| Finals | Thunder over Spurs (54.2%) | Spurs over Thunder, 4-3 | ❌ |

**NBA Finals**

| Predicted | Actual | Result |
| --- | --- | --- |
| Thunder over Pistons (62.5%) | Knicks over Spurs, 4-1 | ❌ |

**Summary**

| Round | Correct |
| --- | --- |
| First Round | 6 / 8 |
| Conference Semifinals | 2 / 4 |
| Conference Finals | 0 / 2 |
| NBA Finals | 0 / 1 |
| **Total** | **8 / 15** |

A series counts as correct when the predicted winner won, even if the opponent was not the one predicted. Once an earlier pick was wrong, later rounds had a mismatched bracket.

**Takeaways**

- The model did well in the first round (75%), where the gap between teams is largest.
- Both first-round misses were upsets (the 76ers over the Celtics in 7, and the Timberwolves over the Nuggets), and the Celtics pick was one of the model's most confident (73.6%).
- The Thunder-Spurs conference final went to seven games. The model gave the Thunder a 54.2% edge, close to a coin flip, so the miss is consistent with its own uncertainty.
- The model did not see the Knicks coming. They were never picked past the second round, which suggests that regular-season stats alone miss things like roster health, playoff rotations and matchup effects.

<details>
<summary>Full predicted bracket</summary>

```
EASTERN CONFERENCE
First Round
  Detroit Pistons vs Orlando Magic          => Detroit Pistons (72.0%)
  Cleveland Cavaliers vs Toronto Raptors    => Cleveland Cavaliers (63.7%)
  New York Knicks vs Atlanta Hawks          => New York Knicks (57.6%)
  Boston Celtics vs Philadelphia 76ers      => Boston Celtics (73.6%)
Conference Semifinals
  Detroit Pistons vs Cleveland Cavaliers    => Detroit Pistons (62.5%)
  New York Knicks vs Boston Celtics         => Boston Celtics (55.5%)
Conference Finals
  Detroit Pistons vs Boston Celtics         => Detroit Pistons (50.3%)

WESTERN CONFERENCE
First Round
  Oklahoma City Thunder vs Phoenix Suns         => Oklahoma City Thunder (80.8%)
  Los Angeles Lakers vs Houston Rockets         => Los Angeles Lakers (62.7%)
  Denver Nuggets vs Minnesota Timberwolves      => Denver Nuggets (75.6%)
  San Antonio Spurs vs Portland Trail Blazers   => San Antonio Spurs (81.0%)
Conference Semifinals
  Oklahoma City Thunder vs Los Angeles Lakers   => Oklahoma City Thunder (62.4%)
  Denver Nuggets vs San Antonio Spurs           => San Antonio Spurs (53.0%)
Conference Finals
  Oklahoma City Thunder vs San Antonio Spurs    => Oklahoma City Thunder (54.2%)

NBA FINALS
  Detroit Pistons vs Oklahoma City Thunder      => Oklahoma City Thunder (62.5%)

PREDICTED CHAMPION: OKLAHOMA CITY THUNDER
```

</details>

## Limitations

- **Small dataset:** 150 series, so results are noisy and hyperparameter tuning can overfit.
- **Regular-season features only:** no injuries, roster changes, seeding, home-court advantage, or player-level data.
- **Binary outcome:** the model predicts a winner and a probability, not series length or game scores.
- **Playoff-only training:** the model only learns from teams that made the playoffs, so it has no view of matchups between very different teams.
