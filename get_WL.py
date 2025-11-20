import pandas as pd
import os
import time

TEAM_ABBR = {
    # ---- Current 30 teams ----
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

    # ---- Defunct BAA/NBA teams ----
    "Anderson Packers": "AND",
    "Baltimore Bullets (1947-1955)": "BLB",  # original Bullets, folded in 1954-55
    "Chicago Stags": "CHS",
    "Cleveland Rebels": "CLR",
    "Denver Nuggets (1949-1950)": "DNV",     # original Nuggets
    "Detroit Falcons": "DTF",
    "Indianapolis Jets": "INJ",
    "Indianapolis Olympians": "INO",
    "Pittsburgh Ironmen": "PIT",
    "Providence Steamrollers": "PRO",
    "Sheboygan Red Skins": "SHE",
    "St. Louis Bombers": "SLB",
    "Toronto Huskies": "TRH",
    "Washington Capitols": "WSC",
    "Waterloo Hawks": "WAT",

    # ---- Relocated / old names of current franchises ----
    # Hawks franchise
    "Tri-Cities Blackhawks": "TRI",
    "Milwaukee Hawks": "MLH",
    "St. Louis Hawks": "STL",

    # Pistons franchise
    "Fort Wayne Pistons": "FTW",

    # Royals / Kings franchise
    "Rochester Royals": "ROC",
    "Cincinnati Royals": "CIN",
    "Kansas City-Omaha Kings": "KCO",
    "Kansas City Kings": "KCK",

    # Lakers franchise
    "Minneapolis Lakers": "MPL",

    # Warriors franchise
    "Philadelphia Warriors": "PHW",
    "San Francisco Warriors": "SFW",

    # Bullets / Wizards franchise
    "Chicago Packers": "CHP",
    "Chicago Zephyrs": "CHP",
    "Baltimore Bullets": "BAL",          # 1963–1973 Bullets (Wizards lineage)
    "Capital Bullets": "CAP",
    "Washington Bullets": "WAS",

    # Nationals / 76ers franchise
    "Syracuse Nationals": "SYR",

    # Rockets franchise
    "San Diego Rockets": "SDR",

    # Braves / Clippers franchise
    "Buffalo Braves": "BUF",
    "San Diego Clippers": "SDC",

    # Jazz franchise
    "New Orleans Jazz": "NOR",

    # Grizzlies franchise
    "Vancouver Grizzlies": "VAN",

    # Sonics / Thunder franchise
    "Seattle SuperSonics": "SEA",

    # Nets franchise
    "New Jersey Nets": "NJN",
    "New York Nets": "NYN",

    # Charlotte / New Orleans mess
    "Charlotte Bobcats": "CHN",
    "New Orleans Hornets": "NOH",
    "New Orleans/Oklahoma City Hornets": "NOK",
}


def fetch_season_wl_br(season_end_year: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_standings.html"

    tables = pd.read_html(url)

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
        unknown = sorted(
            set(tmp[team_col].unique()) - set(TEAM_ABBR.keys())
        )
        if unknown:
            print(f"[warning] {season_end_year} unknown team: {unknown}")

        tmp["TEAM_ABBR"] = tmp[team_col].map(TEAM_ABBR)
        tmp = tmp[tmp["TEAM_ABBR"].notna()]

        if tmp.empty:
            continue

        tmp = tmp[["TEAM_ABBR", "W", "L"]].copy()
        all_rows.append(tmp)
    merged = pd.concat(all_rows, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["TEAM_ABBR"], keep="first")

    merged["WINS"] = merged["W"].astype(int)
    merged["LOSSES"] = merged["L"].astype(int)
    out = merged[["TEAM_ABBR", "WINS", "LOSSES"]].sort_values("TEAM_ABBR").reset_index(drop=True)
    return out


if __name__ == "__main__":
    SEASON_END_YEARS = range(2025, 2027)  

    out_root = os.path.join("Team_stats", "WL")
    os.makedirs(out_root, exist_ok=True)  

    for year in SEASON_END_YEARS:
        df = fetch_season_wl_br(year)  
        y1 = year - 1
        out_path = os.path.join(out_root, f"{y1}-{year}.csv")
        df.to_csv(out_path, index=False)
        print(f"{year} saved to {out_path}")
        time.sleep(3)
