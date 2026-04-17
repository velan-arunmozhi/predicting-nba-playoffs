"""
NBA Playoff Prediction — Full Data Collection Script
=====================================================
Collects:
  1. Regular season team advanced stats (2015-2024) via nba_api
  2. Playoff series results (2015-2024) scraped from Basketball Reference
  3. Team game logs for momentum features (last-20-game rolling stats)
  4. Playoff team flags from league standings

Requirements:
    pip install nba_api pandas requests beautifulsoup4 lxml tqdm
"""

import time
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    teamgamelogs,
    leaguestandings,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2025)]  # 2015-16 to 2024-25
BR_BASE = "https://www.basketball-reference.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_DELAY = 1.5  # seconds between nba_api calls
BR_DELAY = 3.0  # seconds between Basketball Reference requests


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def safe_request(endpoint_fn, retries=3, delay=REQUEST_DELAY):
    """Call an nba_api endpoint with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            return endpoint_fn()
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(delay * 2)
    raise RuntimeError("Max retries exceeded")


def br_get(url):
    """Fetch a Basketball Reference page politely."""
    time.sleep(BR_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


# ---------------------------------------------------------------------------
# 1. Regular season team stats
#
# Correct parameter names (from nba_api docs):
#   measure_type_detailed_defense  — "Base" | "Advanced" | "Misc" | etc.
#   per_mode_detailed               — "PerGame" | "Totals" | "Per100Possessions" | etc.
#   season_type_all_star            — "Regular Season" | "Playoffs" | etc.
#   pace_adjust, plus_minus, rank   — required non-nullable: "Y" or "N"
#   last_n_games, month, opponent_team_id, period — required non-nullable ints
# ---------------------------------------------------------------------------

REQUIRED_PARAMS = dict(
    last_n_games=0,
    month=0,
    opponent_team_id=0,
    period=0,
    pace_adjust="N",
    plus_minus="N",
    rank="N",
    season_type_all_star="Regular Season",
)


def fetch_advanced_team_stats(season):
    """Pull advanced team stats for one season."""
    print(f"  Fetching advanced stats: {season}")
    df = safe_request(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            **REQUIRED_PARAMS,
        ).get_data_frames()[0]
    )
    df["SEASON"] = season
    return df


def fetch_base_team_stats(season):
    """Pull base (traditional) team stats."""
    print(f"  Fetching base stats: {season}")
    df = safe_request(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            **REQUIRED_PARAMS,
        ).get_data_frames()[0]
    )
    df["SEASON"] = season
    return df


def collect_all_team_stats():
    """Collect and merge advanced + base stats for all seasons."""
    adv_frames, base_frames = [], []

    for season in tqdm(SEASONS, desc="Team stats"):
        adv_frames.append(fetch_advanced_team_stats(season))
        base_frames.append(fetch_base_team_stats(season))

    adv = pd.concat(adv_frames, ignore_index=True)
    base = pd.concat(base_frames, ignore_index=True)

    base_cols = [
        "TEAM_ID",
        "SEASON",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "W",
        "L",
        "W_PCT",
    ]
    base_cols = [c for c in base_cols if c in base.columns]

    merged = adv.merge(
        base[base_cols],
        on=["TEAM_ID", "SEASON"],
        how="left",
        suffixes=("", "_base"),
    )
    print(f"\nTeam stats collected: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


# ---------------------------------------------------------------------------
# 2. Playoff series results from Basketball Reference
# ---------------------------------------------------------------------------


def parse_br_playoff_series(season_year):
    """
    Scrape playoff series results for one season.
    season_year = the year the season ends, e.g. 2016 for 2015-16.
    """
    url = f"{BR_BASE}/playoffs/NBA_{season_year}.html"
    print(f"  Scraping BR playoffs: NBA_{season_year}")
    season_str = f"{season_year - 1}-{str(season_year)[2:]}"

    try:
        soup = br_get(url)
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return []

    series_list = []

    # Primary: table#all_playoffs (Basketball Reference standard playoff table)
    tbl = soup.find("table", {"id": "all_playoffs"})
    if tbl:
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2 and "over" in tds[1].text:
                round_name = tds[0].text.strip()
                a_tags = tds[1].find_all("a")
                if len(a_tags) == 2:
                    games = tds[1].text.split("(")[-1].split(")")[0].strip()
                    series_list.append(
                        {
                            "SEASON": season_str,
                            "ROUND": round_name,
                            "WINNER": a_tags[0].text.strip(),
                            "LOSER": a_tags[1].text.strip(),
                            "GAMES": games,
                        }
                    )
        if series_list:
            return series_list

    # Fallback: series-summary divs
    summaries = soup.select("div.series-summary")
    if summaries:
        for summary in summaries:
            winner_td = summary.select_one("tr.winner td.left a")
            loser_td = summary.select_one("tr.loser td.left a")
            games_tds = summary.select("td.center")

            winner = winner_td.get_text(strip=True) if winner_td else ""
            loser = loser_td.get_text(strip=True) if loser_td else ""
            games = games_tds[-1].get_text(strip=True) if games_tds else ""

            heading = ""
            parent = summary.find_parent("div", class_="round")
            if parent:
                h = parent.find(["h2", "h3", "h4"])
                if h:
                    heading = h.get_text(strip=True)

            if winner:
                series_list.append(
                    {
                        "SEASON": season_str,
                        "ROUND": heading,
                        "WINNER": winner,
                        "LOSER": loser,
                        "GAMES": games,
                    }
                )
        return series_list

    # Final fallback: extract from team links in bracket divs
    for section in soup.find_all("div", class_=["round", "series"]):
        links = section.find_all("a")
        team_names = [
            a.get_text(strip=True) for a in links if "/teams/" in a.get("href", "")
        ]
        if len(team_names) >= 2:
            series_list.append(
                {
                    "SEASON": season_str,
                    "ROUND": "Unknown",
                    "WINNER": team_names[0],
                    "LOSER": team_names[1],
                    "GAMES": "",
                }
            )

    return series_list


def collect_all_playoff_results():
    """Collect playoff series results for all seasons."""
    all_series = []
    for season in tqdm(SEASONS, desc="Playoff results"):
        year = int(season.split("-")[0]) + 1
        all_series.extend(parse_br_playoff_series(year))

    df = pd.DataFrame(all_series)
    print(f"\nPlayoff series collected: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 3. Team game logs for momentum features
#    TeamGameLogs uses: team_id_nullable, season_nullable, season_type_nullable
# ---------------------------------------------------------------------------


def fetch_team_game_log(team_id, season):
    """Pull the full regular season game log for one team."""
    df = safe_request(
        lambda: teamgamelogs.TeamGameLogs(
            team_id_nullable=str(team_id),
            season_nullable=season,
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]
    )
    df["SEASON"] = season
    return df


def compute_momentum_features(game_log, n=20):
    """Compute last-N-game momentum features from a full season game log."""
    if game_log.empty:
        return {}

    gl = game_log.sort_values("GAME_DATE").reset_index(drop=True)
    gl["WIN"] = gl["WL"].apply(lambda x: 1 if x == "W" else 0)
    last_n = gl.tail(n)

    features = {f"WIN_PCT_last{n}": round(last_n["WIN"].mean(), 4)}
    for col in ["PTS", "PLUS_MINUS"]:
        if col in last_n.columns:
            features[f"{col}_last{n}"] = round(last_n[col].mean(), 4)
    return features


def collect_momentum_features(team_stats):
    """Fetch game logs and compute momentum for every team-season."""
    records = []
    rows = list(
        team_stats[["TEAM_ID", "SEASON"]].drop_duplicates().itertuples(index=False)
    )
    for row in tqdm(rows, desc="Game logs (momentum)"):
        try:
            gl = fetch_team_game_log(row.TEAM_ID, row.SEASON)
            momentum = compute_momentum_features(gl, n=20)
            momentum["TEAM_ID"] = row.TEAM_ID
            momentum["SEASON"] = row.SEASON
            records.append(momentum)
        except Exception as e:
            print(f"  Skipping {row.TEAM_ID} {row.SEASON}: {e}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Playoff flags from standings
# ---------------------------------------------------------------------------


def fetch_playoff_teams(season):
    """Return {TEAM_ID, SEASON, MADE_PLAYOFFS} from LeagueStandings."""
    print(f"  Fetching standings: {season}")
    df = safe_request(
        lambda: leaguestandings.LeagueStandings(season=season).get_data_frames()[0]
    )
    df["SEASON"] = season
    df["MADE_PLAYOFFS"] = df["PlayoffRank"].apply(
        lambda x: 1 if pd.notna(x) and int(x) > 0 else 0
    )
    return df[["TeamID", "SEASON", "MADE_PLAYOFFS"]].rename(
        columns={"TeamID": "TEAM_ID"}
    )


# ---------------------------------------------------------------------------
# 5. Normalize features by season (z-score within each season)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "NET_RATING",
    "OFF_RATING",
    "DEF_RATING",
    "TS_PCT",
    "AST_TO",
    "REB_PCT",
    "PACE",
    "W_PCT",
    "PTS",
    "PLUS_MINUS",
]


def normalize_by_season(df, cols):
    """Z-score normalize each feature within its season."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mean = df.groupby("SEASON")[col].transform("mean")
        std = df.groupby("SEASON")[col].transform("std").replace(0, 1)
        df[f"{col}_norm"] = ((df[col] - mean) / std).round(4)
    return df


# ---------------------------------------------------------------------------
# 6. Build final ML-ready dataset
# ---------------------------------------------------------------------------


def build_dataset(team_stats, playoff_results, momentum, playoff_flags):
    """Join all sources into a single ML-ready DataFrame."""
    df = team_stats.copy()

    df = df.merge(playoff_flags, on=["TEAM_ID", "SEASON"], how="left")
    df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].fillna(0).astype(int)

    if not momentum.empty:
        df = df.merge(momentum, on=["TEAM_ID", "SEASON"], how="left")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = normalize_by_season(df, available_cols)

    if not playoff_results.empty and "WINNER" in playoff_results.columns:
        win_counts = (
            playoff_results[playoff_results["WINNER"] != ""]
            .groupby(["WINNER", "SEASON"])
            .size()
            .reset_index(name="SERIES_WINS")
            .rename(columns={"WINNER": "TEAM_NAME"})
        )
        if "TEAM_NAME" in df.columns:
            df = df.merge(win_counts, on=["TEAM_NAME", "SEASON"], how="left")
            df["SERIES_WINS"] = df["SERIES_WINS"].fillna(0).astype(int)

    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs("datasets", exist_ok=True)
    print("=" * 60)
    print("NBA Playoff Prediction — Data Collection")
    print("=" * 60)

    print("\n[1/5] Collecting regular season team stats...")
    team_stats = collect_all_team_stats()
    team_stats.to_csv("datasets/team_stats_2015_2025.csv", index=False)
    print("  Saved: datasets/team_stats_2015_2025.csv")

    print("\n[2/5] Scraping playoff results from Basketball Reference...")
    playoff_results = collect_all_playoff_results()
    playoff_results.to_csv("datasets/playoff_results_2015_2025.csv", index=False)
    print("  Saved: datasets/playoff_results_2015_2025.csv")

    print("\n[3/5] Collecting playoff team flags from standings...")
    flags_list = []
    for season in tqdm(SEASONS, desc="Standings"):
        flags_list.append(fetch_playoff_teams(season))
    playoff_flags = pd.concat(flags_list, ignore_index=True)
    playoff_flags.to_csv("datasets/playoff_flags_2015_2025.csv", index=False)
    print("  Saved: datasets/playoff_flags_2015_2025.csv")

    print("\n[4/5] Collecting game logs for momentum features...")
    print("  ~900 API calls, ~30 min. Press Ctrl+C to skip.")
    try:
        momentum = collect_momentum_features(team_stats)
        momentum.to_csv("datasets/momentum_features_2015_2025.csv", index=False)
        print("  Saved: datasets/momentum_features_2015_2025.csv")
    except KeyboardInterrupt:
        print("  Skipped — momentum features will be absent.")
        momentum = pd.DataFrame()

    print("\n[5/5] Building final ML-ready dataset...")
    dataset = build_dataset(team_stats, playoff_results, momentum, playoff_flags)
    dataset.to_csv("datasets/nba_playoff_prediction_dataset.csv", index=False)
    print("  Saved: datasets/nba_playoff_prediction_dataset.csv")

    print("\n" + "=" * 60)
    print("Collection complete. Output files:")
    print("  team_stats_2015_2025.csv            raw advanced + base stats")
    print("  playoff_results_2015_2025.csv       series winners / losers")
    print("  playoff_flags_2015_2025.csv         which teams made playoffs")
    print("  momentum_features_2015_2025.csv     last-20-game rolling stats")
    print("  nba_playoff_prediction_dataset.csv  final ML-ready dataset")
    print("=" * 60)

    norm_cols = [c for c in dataset.columns if c.endswith("_norm")]
    print(f"\nNormalized feature columns ({len(norm_cols)}): {norm_cols}")
    print(f"Targets: MADE_PLAYOFFS, SERIES_WINS")
    print(f"Shape:   {dataset.shape}")


if __name__ == "__main__":
    main()
