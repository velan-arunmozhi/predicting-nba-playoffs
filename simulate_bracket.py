import pandas as pd
import joblib

# The season to pull team stats from. Modify this to use a different season.
# Example: '2024-25' corresponds to the 2024-2025 regular season.
SEASON = "2024-25"

# Sample bracket for demonstration (you can replace these with the actual teams once set).
# Must match exactly the TEAM_NAME in team_stats_2015_2025.csv
# Note: "Los Angeles Clippers" should be "LA Clippers".
EAST_BRACKET = [
    ("Boston Celtics", "Orlando Magic"),
    ("New York Knicks", "Detroit Pistons"),
    ("Milwaukee Bucks", "Indiana Pacers"),
    ("Cleveland Cavaliers", "Miami Heat"),
]

WEST_BRACKET = [
    ("Oklahoma City Thunder", "Memphis Grizzlies"),
    ("Denver Nuggets", "LA Clippers"),
    (
        "Minnesota Timberwolves",
        "Los Angeles Lakers",
    ),
    ("Houston Rockets", "Golden State Warriors"),
]


def load_data():
    team_stats = pd.read_csv("datasets/nba_playoff_prediction_dataset.csv")
    model = joblib.load("models/xgboost_h2h_model.joblib")

    # We need to know which features the model expects
    model_features = model.feature_names_in_
    # Clean the _DIFF suffix to know which base stats we need
    base_features = [f.replace("_DIFF", "") for f in model_features]

    return team_stats, model, base_features, model_features


def predict_series(
    team_a, team_b, team_stats, model, base_features, model_features, season=SEASON
):
    """Predict the winner between two teams based on their regular season stats."""

    # Fix common naming issues just in case
    mapping = {
        "Los Angeles Clippers": "LA Clippers",
        "New Orleans Timberwolves": "New Orleans Pelicans",
    }
    team_a = mapping.get(team_a, team_a)
    team_b = mapping.get(team_b, team_b)

    stats_a = team_stats[
        (team_stats["SEASON"] == season) & (team_stats["TEAM_NAME"] == team_a)
    ]
    stats_b = team_stats[
        (team_stats["SEASON"] == season) & (team_stats["TEAM_NAME"] == team_b)
    ]

    if stats_a.empty:
        raise ValueError(f"Could not find stats for {team_a} in season {season}.")
    if stats_b.empty:
        raise ValueError(f"Could not find stats for {team_b} in season {season}.")

    stats_a = stats_a.iloc[0]
    stats_b = stats_b.iloc[0]

    # Construct feature difference row
    row = {}
    for base_f, model_f in zip(base_features, model_features):
        row[model_f] = stats_a[base_f] - stats_b[base_f]

    df_pred = pd.DataFrame([row])

    # Model predicts probability that Team A (Target=1) wins
    prob_a_wins = model.predict_proba(df_pred)[0][1]

    if prob_a_wins > 0.5:
        return team_a, prob_a_wins
    else:
        return team_b, 1.0 - prob_a_wins


def simulate_round(
    matchups, round_name, team_stats, model, base_features, model_features
):
    print(f"\n--- {round_name} ---")
    winners = []
    for team_a, team_b in matchups:
        winner, prob = predict_series(
            team_a, team_b, team_stats, model, base_features, model_features
        )
        print(f"{team_a} vs {team_b}  =>  {winner} wins ({prob:.1%} probability)")
        winners.append(winner)

    # Pair winners sequentially for the next round
    next_round_matchups = []
    for i in range(0, len(winners), 2):
        if i + 1 < len(winners):
            next_round_matchups.append((winners[i], winners[i + 1]))
        else:
            # If odd number (like Finals), just return the single winner list
            return winners

    return next_round_matchups


def main():
    print("============================================================")
    print("NBA Playoff Bracket Simulator")
    print("============================================================")

    team_stats, model, base_features, model_features = load_data()

    print("\n[ SIMULATING EASTERN CONFERENCE ]")
    east_r2 = simulate_round(
        EAST_BRACKET, "First Round", team_stats, model, base_features, model_features
    )
    east_cf = simulate_round(
        east_r2,
        "Conference Semifinals",
        team_stats,
        model,
        base_features,
        model_features,
    )
    east_champ = simulate_round(
        east_cf, "Conference Finals", team_stats, model, base_features, model_features
    )[0]

    print("\n[ SIMULATING WESTERN CONFERENCE ]")
    west_r2 = simulate_round(
        WEST_BRACKET, "First Round", team_stats, model, base_features, model_features
    )
    west_cf = simulate_round(
        west_r2,
        "Conference Semifinals",
        team_stats,
        model,
        base_features,
        model_features,
    )
    west_champ = simulate_round(
        west_cf, "Conference Finals", team_stats, model, base_features, model_features
    )[0]

    print("\n============================================================")
    print("[ NBA FINALS ]")
    finals_matchup = [(east_champ, west_champ)]
    nba_champ = simulate_round(
        finals_matchup,
        "Finals Series",
        team_stats,
        model,
        base_features,
        model_features,
    )[0]
    print("============================================================")
    print(f"!!! PROJECTED NBA CHAMPION: {nba_champ.upper()} !!!")
    print("============================================================")


if __name__ == "__main__":
    main()
