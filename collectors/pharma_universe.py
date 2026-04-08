"""
collectors/pharma_universe.py
Fetches all pharma companies from SEC EDGAR using SIC codes.
Run: python -m collectors.pharma_universe
"""

import requests
import pandas as pd
import time
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

logger.add("logs/pharma_universe.log", rotation="10 MB")

PHARMA_SIC_CODES = {
    "2830", "2833", "2834", "2835", "2836",
    "8731", "8734", "5122", "2860", "2890"
}

SEC_HEADERS = {
    "User-Agent": "Jyotiraditya Garikipati jyothiraditya.g29@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}


def fetch_sec_company_list():
    logger.info("Fetching pharma companies from SEC EDGAR using SIC codes...")

    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["data"], columns=data["fields"])
    df["cik"] = df["cik"].astype(str).str.zfill(10)

    print(f"Total companies in SEC registry: {len(df)}")
    print("Now checking SIC codes for each company...")
    print("This takes 15-20 minutes but catches ALL pharma companies...")

    pharma_rows = []
    checked = 0

    for _, row in df.iterrows():
        cik      = str(row["cik"]).zfill(10)
        ticker   = str(row.get("ticker",   "")).strip()
        name     = str(row.get("name",     "")).strip()
        exchange = str(row.get("exchange", "")).strip()

        if exchange not in ["Nasdaq", "NYSE", "ARCA", "BATS"]:
            continue

        try:
            sub_url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
            sub_resp = requests.get(
                sub_url,
                headers={"User-Agent": "Jyotiraditya Garikipati jyothiraditya.g29@gmail.com"},
                timeout=10
            )

            if sub_resp.status_code == 200:
                sub_data        = sub_resp.json()
                sic             = str(sub_data.get("sic", ""))
                sic_description = sub_data.get("sicDescription", "")

                if sic in PHARMA_SIC_CODES:
                    pharma_rows.append({
                        "ticker":          ticker,
                        "name":            name,
                        "cik":             cik,
                        "exchange":        exchange,
                        "sic":             sic,
                        "sic_description": sic_description,
                    })

            checked += 1
            if checked % 500 == 0:
                print(f"  Checked {checked} companies, found {len(pharma_rows)} pharma so far...")

            time.sleep(0.1)

        except Exception:
            continue

    df_pharma = pd.DataFrame(pharma_rows)
    logger.info(f"Found {len(df_pharma)} pharma companies using SIC codes")
    return df_pharma


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("PHARMA UNIVERSE COLLECTOR — SIC CODE VERSION")
    print("=" * 60)

    df = fetch_sec_company_list()

    out_path = f"{RAW_DATA_DIR}/pharma_companies.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total pharma companies found: {len(df)}")
    print(f"\nTop 20 companies:")
    print(df[["ticker","name","exchange","sic_description"]].head(20).to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()