import pandas as pd
import os
import time


def fetch_season_player_totals(season_end_year: int) -> pd.DataFrame:
    # Build the Basketball-Reference URL for a given season's player totals page
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_totals.html"
    print(f"Fetching {url} ...")

    # Read all HTML tables on the page and take the first one as the stats table
    df = pd.read_html(url, header=0)[0]

    # Remove repeated header rows inside the table where "Rk" appears as a data value
    df = df[df["Rk"] != "Rk"].copy()

    # Clean up basic columns:
    # - Convert "Rk" to integer (nullable Int64 type)
    # - Strip extra whitespace in string columns
    df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce").astype("Int64")
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()
    df["Tm"] = df["Tm"].astype(str).str.strip()

    # Convert all other columns (except non-numeric ones) to numeric, coercing errors to NaN
    non_numeric = {"Player", "Pos", "Tm"}
    for col in df.columns:
        if col in non_numeric:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Return cleaned per-player totals for this season
    return df


if __name__ == "__main__":
    # Range of season ending years to fetch (e.g., 1967-1968 up through 2024-2025)
    SEASON_END_YEARS = range(1968, 2026)

    # Output root directory for all player stats CSVs
    out_root_player = os.path.join("Player_stats", "all_stats")
    os.makedirs(out_root_player, exist_ok=True)

    # Loop over each season end year and fetch/save player totals
    for year in SEASON_END_YEARS:
        try:
            # Download and parse player totals table for the given year
            df_player = fetch_season_player_totals(year)
        except Exception as e:
            # Print error and skip this year if anything goes wrong during fetch/parse
            print(f"[error] {year} fail: {e}")
            continue

        # Build season label like "1967-1968" from the ending year
        y1 = year - 1
        out_path = os.path.join(out_root_player, f"{y1}-{year}.csv")

        # Save cleaned player totals to CSV
        df_player.to_csv(out_path, index=False)
        print(f"Player totals {year} saved to {out_path}")

        # Sleep between requests to be polite to the remote server
        time.sleep(3)
