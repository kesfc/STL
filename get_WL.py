import pandas as pd
import os

TEAM_ABBR = {
    # ---- 当前 30 支队（你原来的那一部分）----
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

    # ---- 历史名字：Clippers / Braves ----
    # Los Angeles Clippers, San Diego Clippers, Buffalo Braves :contentReference[oaicite:0]{index=0}
    "Buffalo Braves": "LAC",
    "San Diego Clippers": "LAC",

    # ---- 历史名字：Nets 系 ----
    # Brooklyn Nets, New Jersey Nets, New York Nets :contentReference[oaicite:1]{index=1}
    "New Jersey Nets": "BKN",
    "New York Nets": "BKN",

    # ---- 历史名字：New Orleans 系 ----
    # New Orleans Pelicans, New Orleans/Oklahoma City Hornets, New Orleans Hornets :contentReference[oaicite:2]{index=2}
    "New Orleans Hornets": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOP",

    # ---- 历史名字：Jazz 系 ----
    # New Orleans Jazz, Utah Jazz :contentReference[oaicite:3]{index=3}
    "New Orleans Jazz": "UTA",

    # ---- 历史名字：Kings 系 ----
    # Sacramento Kings, Kansas City Kings, Kansas City-Omaha Kings :contentReference[oaicite:4]{index=4}
    "Kansas City Kings": "SAC",
    "Kansas City-Omaha Kings": "SAC",

    # ---- 历史名字：Grizzlies 系 ----
    # Vancouver Grizzlies 迁到 Memphis :contentReference[oaicite:5]{index=5}
    "Vancouver Grizzlies": "MEM",

    # ---- 历史名字：Sonics / Thunder 系 ----
    # Seattle SuperSonics 迁到 Oklahoma City Thunder :contentReference[oaicite:6]{index=6}
    "Seattle SuperSonics": "OKC",

    # ---- 历史名字：Wizards / Bullets 系 ----
    # Baltimore Bullets -> Capital Bullets -> Washington Bullets -> Washington Wizards :contentReference[oaicite:7]{index=7}
    "Washington Bullets": "WAS",
    "Capital Bullets": "WAS",  # 理论上 73-74 才会用到，如果你往前多抓一两年也不会崩

    # ---- 历史名字：Charlotte 系 ----
    # Charlotte Hornets 历史被 NBA 认定为一个 franchise，2004-2014 期间叫 Charlotte Bobcats :contentReference[oaicite:8]{index=8}
    "Charlotte Bobcats": "CHA",
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
    SEASON_END_YEARS = range(1968, 2026)  

    out_root = os.path.join("Team_stats", "WL")
    os.makedirs(out_root, exist_ok=True)  

    for year in SEASON_END_YEARS:
        df = fetch_season_wl_br(year)  
        y1 = year - 1
        out_path = os.path.join(out_root, f"{year}-{y1}.csv")
        df.to_csv(out_path, index=False)
        print(f"{year} saved to {out_path}")
