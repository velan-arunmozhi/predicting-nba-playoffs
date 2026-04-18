import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def normalize_team_name(name):
    # Map from BR names to NBA API names
    mapping = {
        "Los Angeles Clippers": "LA Clippers",
        "Charlotte Bobcats": "Charlotte Hornets",  # Just in case
    }
    return mapping.get(name, name)


def main():
    print("=" * 60)
    print("NBA Playoff Prediction - Training H2H Model")
    print("=" * 60)

    # 1. Load Data
    team_stats = pd.read_csv("datasets/nba_playoff_prediction_dataset.csv")
    playoff_results = pd.read_csv("datasets/playoff_results_2015_2025.csv")

    # Select features to use (normalized season stats + momentum features)
    feature_cols = [
        c for c in team_stats.columns if c.endswith("_norm") or "last20" in c
    ]
    print(f"Selected features ({len(feature_cols)}): {feature_cols}")

    # Build H2H Dataset
    h2h_records = []

    for _, row in playoff_results.iterrows():
        season = row["SEASON"]
        winner = normalize_team_name(row["WINNER"])
        loser = normalize_team_name(row["LOSER"])

        # Get team stats
        winner_stats = team_stats[
            (team_stats["SEASON"] == season) & (team_stats["TEAM_NAME"] == winner)
        ]
        loser_stats = team_stats[
            (team_stats["SEASON"] == season) & (team_stats["TEAM_NAME"] == loser)
        ]

        if winner_stats.empty or loser_stats.empty:
            print(f"Skipping {season} {winner} vs {loser} (missing stats)")
            continue

        winner_stats = winner_stats.iloc[0]
        loser_stats = loser_stats.iloc[0]

        # We need two rows per series (A wins, B loses) and (B loses, A wins)
        # Row 1: Team A = Winner, Team B = Loser, Target = 1
        record_1 = {"SEASON": season, "TEAM_A": winner, "TEAM_B": loser, "TARGET": 1}
        for col in feature_cols:
            record_1[f"{col}_DIFF"] = winner_stats[col] - loser_stats[col]
        h2h_records.append(record_1)

        # Row 2: Team A = Loser, Team B = Winner, Target = 0
        record_2 = {"SEASON": season, "TEAM_A": loser, "TEAM_B": winner, "TARGET": 0}
        for col in feature_cols:
            record_2[f"{col}_DIFF"] = loser_stats[col] - winner_stats[col]
        h2h_records.append(record_2)

    h2h_df = pd.DataFrame(h2h_records)
    print(f"Generated {len(h2h_df)} H2H records from {len(playoff_results)} series.")

    # Save the H2H dataset just in case
    h2h_df.to_csv("datasets/h2h_matchups_dataset.csv", index=False)

    # 3. Train / Test Split (Chronological)
    # Train on 2015-16 to 2023-24, Test on 2024-25
    train_df = h2h_df[h2h_df["SEASON"] < "2024-25"]
    test_df = h2h_df[h2h_df["SEASON"] == "2024-25"]

    # If the test set is empty (e.g., if scraping 2024 failed), fallback to random split
    if test_df.empty or len(test_df) < 10:
        print("Using random 80/20 split due to limited recent test data.")
        train_df, test_df = train_test_split(h2h_df, test_size=0.2, random_state=42)

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    X_col = [c for c in h2h_df.columns if c.endswith("_DIFF")]

    X_train = train_df[X_col]
    y_train = train_df["TARGET"]

    X_test = test_df[X_col]
    y_test = test_df["TARGET"]

    # 4. Train XGBoost
    print("\nPerforming Hyperparameter Tuning for XGBoost...")
    base_model = XGBClassifier(random_state=42, eval_metric="logloss")

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3],
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.2%}")

    model = grid_search.best_estimator_

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_h2h_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Feature Importances
    importances = pd.DataFrame(
        {"Feature": X_col, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\nTop 5 Most Important Features:")
    print(importances.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
