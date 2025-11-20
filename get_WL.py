import os
import time
import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",

    "Buffalo Braves": "LAC",
    "San Diego Clippers": "LAC",

    "New Jersey Nets": "BKN",
    "New York Nets": "BKN",

    "New Orleans Hornets": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOP",
    "New Orleans Jazz": "UTA",

    "Kansas City Kings": "SAC",
    "Kansas City-Omaha Kings": "SAC",

    "Vancouver Grizzlies": "MEM",

    "Seattle SuperSonics": "OKC",

    "Washington Bullets": "WAS",
    "Capital Bullets": "WAS",

    "Charlotte Bobcats": "CHA",
}


def fetch_season_wl_br(season_end_year: int,
                       max_retries: int = 5,
                       base_sleep: int = 5) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_standings.html"

    html = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                html = resp.text
                break
            elif resp.status_code == 429:
                wait = base_sleep * attempt
                print(f"[{season_end_year}] HTTP 429 Too Many Requests, wait {wait}s try attempt {attempt} )...")
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except requests.RequestException as e:
            wait = base_sleep * attempt
            print(f"[{season_end_year}] error {e}, wait {wait}s try attempt {attempt}...")
            time.sleep(wait)

    if html is None:
        raise RuntimeError(f"fail {url}")

    tables = pd.read_html(html)

    all_rows = []

    for df in tables:
        if "W" not in df.columns or "L" not in df.columns:
            continue

        team_col = df.columns[0]
        tmp = df[[team_col, "W", "L"]].copy()
        tmp[team_col] = (
            tmp[team_col]
            .astype(str)
            .str.replace("*", "", regex=False)
            .str.strip()
        )

        unknown = sorted(set(tmp[team_col].unique()) - set(TEAM_ABBR.keys()))
        if unknown:
            print(f"[warning] {season_end_year} unknown team: {unknown}")

        tmp["TEAM_ABBR"] = tmp[team_col].map(TEAM_ABBR)
        tmp = tmp[tmp["TEAM_ABBR"].notna()]
        if tmp.empty:
            continue

        tmp = tmp[["TEAM_ABBR", "W", "L"]].copy()
        tmp.columns = ["TEAM_ABBR", "WINS", "LOSSES"]
        all_rows.append(tmp)


    merged = pd.concat(all_rows, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["TEAM_ABBR"], keep="first")


    merged = merged.sort_values("TEAM_ABBR").reset_index(drop=True)
    return merged


if __name__ == "__main__":
    SEASON_END_YEARS = range(1968, 2026)  

    out_dir = os.path.join("Team_stats", "WL")
    os.makedirs(out_dir, exist_ok=True)

    for year in SEASON_END_YEARS:
        y1 = year - 1
        df = fetch_season_wl_br(year)
        out_path = os.path.join(out_dir, f"{year}-{y1}.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> saved to {out_path}")
        time.sleep(3)
