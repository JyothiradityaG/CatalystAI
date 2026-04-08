"""
collectors/fda_approvals.py
 
Step 3: Fetch historical FDA drug approval decisions from openFDA API.
This is the training data for our prediction model.
 
Run:
    python -m collectors.fda_approvals
 
Output:
    data/raw/fda_approvals.csv
"""
 
import requests
import pandas as pd
import time
import os
import sys
from tqdm import tqdm
from loguru import logger
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR
 
logger.add("logs/fda_approvals.log", rotation="10 MB")
 
FDA_DRUG_URL = "https://api.fda.gov/drug/drugsfda.json"
 
 
def fetch_fda_approvals(limit=100, total=5000):
    """
    Fetches historical FDA drug approval records from openFDA.
    Returns a flat list of approval records.
    """
    all_records = []
    skip = 0
 
    with tqdm(total=total, desc="Fetching FDA approvals") as pbar:
        while skip < total:
            params = {
                "limit": limit,
                "skip":  skip,
            }
 
            try:
                resp = requests.get(FDA_DRUG_URL, params=params, timeout=20)
 
                if resp.status_code == 404:
                    logger.info("Reached end of FDA records")
                    break
 
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
 
                if not results:
                    break
 
                for drug in results:
                    sponsor = drug.get("sponsor_name", "")
                    submissions = drug.get("submissions", [])
                    products = drug.get("products", [])
 
                    # Get product details
                    product_names = []
                    dosage_forms = []
                    routes = []
                    for p in products:
                        product_names.append(p.get("brand_name", ""))
                        dosage_forms.append(p.get("dosage_form", ""))
                        routes.append(p.get("route", ""))
 
                    # Get application number
                    app_num = drug.get("application_number", "")
 
                    # Process each submission
                    for sub in submissions:
                        sub_type = sub.get("submission_type", "")
                        sub_num = sub.get("submission_number", "")
                        sub_status = sub.get("submission_status", "")
                        sub_date = sub.get("submission_status_date", "")
                        review_priority = sub.get("review_priority", "")
                        sub_class = sub.get("submission_class_code", "")
                        sub_class_desc = sub.get("submission_class_code_description", "")
 
                        # We want original approvals (ORIG) and efficacy supplements
                        if sub_type not in ["ORIG", "EFFICACY"]:
                            continue
 
                        all_records.append({
                            "application_number": app_num,
                            "sponsor_name":       sponsor,
                            "brand_names":        ", ".join(set(filter(None, product_names))),
                            "dosage_forms":       ", ".join(set(filter(None, dosage_forms))),
                            "routes":             ", ".join(set(filter(None, routes))),
                            "submission_type":    sub_type,
                            "submission_number":  sub_num,
                            "submission_status":  sub_status,
                            "submission_date":    sub_date,
                            "review_priority":    review_priority,
                            "submission_class":   sub_class,
                            "submission_class_desc": sub_class_desc,
                            # Label: 1 = approved, 0 = not approved
                            "approved": 1 if sub_status == "AP" else 0,
                        })
 
                skip += limit
                pbar.update(len(results))
                time.sleep(0.5)
 
            except Exception as e:
                logger.warning(f"Error at skip={skip}: {e}")
                skip += limit
                time.sleep(1)
                continue
 
    return all_records
 
 
def fetch_fda_by_sponsor(sponsor_name):
    """
    Fetch FDA applications for a specific sponsor/company.
    Used to enrich our pharma company list with FDA history.
    """
    params = {
        "search": f'sponsor_name:"{sponsor_name}"',
        "limit":  100,
    }
 
    try:
        resp = requests.get(FDA_DRUG_URL, params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
        return []
    except Exception as e:
        logger.warning(f"Error fetching FDA data for {sponsor_name}: {e}")
        return []
 
 
def compute_approval_stats(df):
    """
    Computes approval rate statistics by various dimensions.
    These become features for our ML model.
    """
    stats = {}
 
    # Overall approval rate
    total = len(df)
    approved = df["approved"].sum()
    stats["overall_approval_rate"] = round(approved / total, 3) if total > 0 else 0
 
    # Approval rate by review priority
    priority_stats = df.groupby("review_priority")["approved"].agg(["mean", "count"])
    stats["by_priority"] = priority_stats.to_dict()
 
    # Approval rate by submission class
    class_stats = df.groupby("submission_class_desc")["approved"].agg(["mean", "count"])
    stats["by_class"] = class_stats.to_dict()
 
    return stats
 
 
def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
 
    print("Fetching FDA drug approval history from openFDA...")
    print("This fetches up to 5,000 historical drug submissions...\n")
 
    records = fetch_fda_approvals(limit=100, total=5000)
 
    if not records:
        print("No records fetched. Check internet connection.")
        return
 
    df = pd.DataFrame(records)
 
    # Clean up dates
    df["submission_date"] = pd.to_datetime(
        df["submission_date"], format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["submission_date"].dt.year
 
    # Save raw data
    out_path = f"{RAW_DATA_DIR}/fda_approvals.csv"
    df.to_csv(out_path, index=False)
 
    # Print summary
    print(f"\n{'='*60}")
    print(f"FDA APPROVAL DATA COLLECTED")
    print(f"{'='*60}")
    print(f"Total submissions:     {len(df)}")
    print(f"Total approved (AP):   {df['approved'].sum()}")
    print(f"Overall approval rate: {df['approved'].mean():.1%}")
    print(f"\nApproval rate by review priority:")
    priority = df.groupby("review_priority")["approved"].agg(["mean","count"])
    priority.columns = ["approval_rate", "count"]
    priority["approval_rate"] = priority["approval_rate"].map("{:.1%}".format)
    print(priority.to_string())
    print(f"\nTop sponsors by submissions:")
    print(df["sponsor_name"].value_counts().head(10).to_string())
    print(f"\nSaved to: {out_path}")
 
 
if __name__ == "__main__":
    main()