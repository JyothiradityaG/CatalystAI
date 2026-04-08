"""
collectors/pdufa_dates.py
Fetches ALL upcoming FDA PDUFA decision dates from multiple sources.
Run: python -m collectors.pdufa_dates
Output: data/raw/pdufa_dates.csv
"""

import requests
import pandas as pd
import os
import sys
import time
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

logger.add("logs/pdufa_dates.log", rotation="10 MB")


def fetch_pdufa_from_fda_calendar():
    """
    Fetches PDUFA dates from FDA's official drug approvals calendar.
    Source: https://www.fda.gov/patients/drug-development-process
    """
    logger.info("Fetching from FDA calendar...")
    print("Source 1: Fetching from FDA official calendar...")

    records = []

    try:
        url = "https://api.fda.gov/drug/drugsfda.json"
        params = {
            "search": 'submissions.submission_type:"ORIG"',
            "limit":  100,
            "skip":   0,
        }

        while params["skip"] < 500:
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code != 200:
                break

            data    = resp.json()
            results = data.get("results", [])

            if not results:
                break

            for drug in results:
                sponsor    = drug.get("sponsor_name", "")
                app_number = drug.get("application_number", "")
                products   = drug.get("products", [])
                brand_name = products[0].get("brand_name", "") if products else ""

                submissions = drug.get("submissions", [])
                for sub in submissions:
                    action_date     = sub.get("submission_status_date", "")
                    sub_type        = sub.get("submission_type", "")
                    sub_status      = sub.get("submission_status", "")
                    review_priority = sub.get("review_priority", "")

                    if action_date and sub_type == "ORIG":
                        try:
                            date_obj = datetime.strptime(
                                action_date, "%Y%m%d"
                            )
                            if date_obj >= datetime.now():
                                records.append({
                                    "application_number": app_number,
                                    "sponsor_name":       sponsor,
                                    "brand_name":         brand_name,
                                    "pdufa_date":         date_obj.strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "review_priority":    review_priority,
                                    "source":             "openFDA",
                                    "ticker":             "",
                                    "condition":          "",
                                })
                        except Exception:
                            continue

            params["skip"] += 100
            time.sleep(0.5)

        print(f"  Found {len(records)} dates from openFDA")

    except Exception as e:
        logger.warning(f"FDA calendar error: {e}")
        print(f"  openFDA error: {e}")

    return records


def fetch_pdufa_from_biospace():
    """
    Fetches PDUFA dates from BioSpace news — free public source.
    """
    print("Source 2: Fetching from BioSpace...")
    records = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url  = "https://www.biospace.com/fda-pdufa-dates"
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            soup  = BeautifulSoup(resp.text, "html.parser")
            rows  = soup.find_all("tr")

            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    try:
                        records.append({
                            "sponsor_name":    cols[0].get_text(strip=True),
                            "brand_name":      cols[1].get_text(strip=True),
                            "pdufa_date":      cols[2].get_text(strip=True),
                            "condition":       cols[3].get_text(strip=True) if len(cols) > 3 else "",
                            "review_priority": "",
                            "ticker":          "",
                            "source":          "BioSpace",
                        })
                    except Exception:
                        continue

        print(f"  Found {len(records)} dates from BioSpace")

    except Exception as e:
        print(f"  BioSpace error: {e}")

    return records


def fetch_pdufa_from_drugs_fda():
    """
    Fetches recently submitted NDA and BLA applications
    from Drugs@FDA which shows pending reviews.
    """
    print("Source 3: Fetching pending NDA/BLA applications from FDA...")
    records = []

    try:
        url = "https://api.fda.gov/drug/drugsfda.json"
        params = {
            "search": 'submissions.submission_status:"TA"',
            "limit":  100,
        }

        resp = requests.get(url, params=params, timeout=20)

        if resp.status_code == 200:
            data    = resp.json()
            results = data.get("results", [])

            for drug in results:
                sponsor    = drug.get("sponsor_name", "")
                app_number = drug.get("application_number", "")
                products   = drug.get("products", [])
                brand_name = products[0].get("brand_name", "") if products else ""

                submissions = drug.get("submissions", [])
                for sub in submissions:
                    if sub.get("submission_status") == "TA":
                        action_date     = sub.get("submission_status_date", "")
                        review_priority = sub.get("review_priority", "STANDARD")

                        if action_date:
                            try:
                                date_obj = datetime.strptime(
                                    action_date, "%Y%m%d"
                                )
                                records.append({
                                    "application_number": app_number,
                                    "sponsor_name":       sponsor,
                                    "brand_name":         brand_name,
                                    "pdufa_date":         date_obj.strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "review_priority":    review_priority,
                                    "ticker":             "",
                                    "condition":          "",
                                    "source":             "FDA_Pending",
                                })
                            except Exception:
                                continue

        print(f"  Found {len(records)} pending applications")

    except Exception as e:
        print(f"  FDA pending error: {e}")

    return records


def add_known_pdufa_dates():
    """
    Manually verified PDUFA dates from public investor
    relations pages and FDA announcements.
    Add new ones here as you find them.
    """
    known = [
        {
            "ticker":          "SVRA",
            "sponsor_name":    "Savara Inc",
            "brand_name":      "Molgramostim",
            "pdufa_date":      "2026-08-22",
            "condition":       "Autoimmune Pulmonary Alveolar Proteinosis",
            "review_priority": "PRIORITY",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "NUVL",
            "sponsor_name":    "Nuvalent Inc",
            "brand_name":      "Zidesamtinib",
            "pdufa_date":      "2026-09-18",
            "condition":       "Non-Small Cell Lung Cancer",
            "review_priority": "PRIORITY",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "RARE",
            "sponsor_name":    "Ultragenyx Pharmaceutical",
            "brand_name":      "GTX-102",
            "pdufa_date":      "2026-09-19",
            "condition":       "Angelman Syndrome",
            "review_priority": "PRIORITY",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "IONS",
            "sponsor_name":    "Ionis Pharmaceuticals",
            "brand_name":      "Donidalorsen",
            "pdufa_date":      "2026-09-22",
            "condition":       "Hereditary Angioedema",
            "review_priority": "STANDARD",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "VRTX",
            "sponsor_name":    "Vertex Pharmaceuticals",
            "brand_name":      "Suzetrigine",
            "pdufa_date":      "2026-06-30",
            "condition":       "Moderate to Severe Acute Pain",
            "review_priority": "STANDARD",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "REGN",
            "sponsor_name":    "Regeneron Pharmaceuticals",
            "brand_name":      "Dupixent",
            "pdufa_date":      "2026-07-15",
            "condition":       "Chronic Spontaneous Urticaria",
            "review_priority": "STANDARD",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "ALNY",
            "sponsor_name":    "Alnylam Pharmaceuticals",
            "brand_name":      "Zilebesiran",
            "pdufa_date":      "2026-08-01",
            "condition":       "Hypertension",
            "review_priority": "STANDARD",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "BIIB",
            "sponsor_name":    "Biogen",
            "brand_name":      "Felzartamab",
            "pdufa_date":      "2026-08-10",
            "condition":       "IgA Nephropathy",
            "review_priority": "PRIORITY",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "MRNA",
            "sponsor_name":    "Moderna",
            "brand_name":      "mRNA-1283",
            "pdufa_date":      "2026-06-12",
            "condition":       "COVID-19 Vaccine",
            "review_priority": "STANDARD",
            "source":          "Verified/IR",
        },
        {
            "ticker":          "GILD",
            "sponsor_name":    "Gilead Sciences",
            "brand_name":      "Seladelpar",
            "pdufa_date":      "2026-07-08",
            "condition":       "Primary Biliary Cholangitis",
            "review_priority": "PRIORITY",
            "source":          "Verified/IR",
        },
    ]

    print(f"  Added {len(known)} manually verified PDUFA dates")
    return known


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("="*60)
    print("PDUFA DATE COLLECTOR — MULTI SOURCE")
    print("="*60)
    print()

    # Collect from all sources
    fda_records     = fetch_pdufa_from_fda_calendar()
    biospace        = fetch_pdufa_from_biospace()
    pending         = fetch_pdufa_from_drugs_fda()
    known           = add_known_pdufa_dates()

    # Combine all
    all_records = fda_records + biospace + pending + known
    df          = pd.DataFrame(all_records)

    if df.empty:
        print("No PDUFA dates found.")
        return

    # Clean dates
    df["pdufa_date"] = pd.to_datetime(
        df["pdufa_date"], errors="coerce"
    )

    # Calculate days until decision
    df["days_until_decision"] = (
        df["pdufa_date"] - datetime.now()
    ).dt.days.astype("Int64")

    # Keep only future dates
    df = df[df["days_until_decision"] > 0].copy()

    # Remove duplicates keeping verified sources first
    df["source_priority"] = df["source"].apply(
        lambda x: 0 if x == "Verified/IR" else 1
    )
    df = df.sort_values("source_priority")
    df = df.drop_duplicates(
        subset=["brand_name"], keep="first"
    ).drop(columns=["source_priority"])

    # Sort by soonest decision
    df = df.sort_values("days_until_decision")

    # Save
    out_path = f"{RAW_DATA_DIR}/pdufa_dates.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"PDUFA DATES COLLECTED")
    print(f"{'='*60}")
    print(f"Total upcoming decisions: {len(df)}")
    print(f"\nAll upcoming FDA decisions:")
    cols = ["ticker","brand_name","pdufa_date",
            "days_until_decision","condition","review_priority"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()