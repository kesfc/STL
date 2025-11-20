import pandas as pd
import os
import time

def fetch_season_player_totals(season_end_year: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_totals.html"
    print(f"Fetching {url} ...")

    df = pd.read_html(url, header=0)[0]

    df = df[df["Rk"] != "Rk"].copy()

    df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce").astype("Int64")
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()
    df["Tm"] = df["Tm"].astype(str).str.strip()

    non_numeric = {"Player", "Pos", "Tm"}
    for col in df.columns:
        if col in non_numeric:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    SEASON_END_YEARS = range(1968, 2026)

    out_root_player = os.path.join("Player_stats", "all_stats")
    os.makedirs(out_root_player, exist_ok=True)

    for year in SEASON_END_YEARS:
        try:
            df_player = fetch_season_player_totals(year)
        except Exception as e:
            print(f"[error] {year} fail: {e}")
            continue
        y1 = year - 1
        out_path = os.path.join(out_root_player, f"{y1}-{year}.csv")
        df_player.to_csv(out_path, index=False)
        print(f"Player totals {year} saved to {out_path}")
        time.sleep(3) 
