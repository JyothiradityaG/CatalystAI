import requests
import pandas as pd
import time
import os
import sys
from tqdm import tqdm
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

logger.add("logs/clinical_trials.log", rotation="10 MB")

CT_API = "https://clinicaltrials.gov/api/v2/studies"
TARGET_PHASES = {"PHASE2", "PHASE3"}


def fetch_trials_for_company(company_name, ticker):
    trials = []

    try:
        params = {
            "query.spons": company_name,
            "pageSize": 40,
            "format": "json",
        }

        resp = requests.get(CT_API, params=params, timeout=20)

        if resp.status_code != 200:
            return trials

        data = resp.json()
        studies = data.get("studies", [])

        for study in studies:
            proto = study.get("protocolSection", {})
            design_mod = proto.get("designModule", {})
            phases = design_mod.get("phases", [])

            # Filter to Phase 2 and Phase 3 only
            matched_phases = [p for p in phases if p in TARGET_PHASES]
            if not matched_phases:
                continue

            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            enroll = design_mod.get("enrollmentInfo", {})

            status = status_mod.get("overallStatus", "")
            if status not in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"]:
                continue

            interventions = arms_mod.get("interventions", [])
            drug_names = [
                i.get("name", "") for i in interventions
                if i.get("type", "").upper() == "DRUG"
            ]

            trials.append({
                "ticker": ticker,
                "company_name": company_name,
                "nct_id": id_mod.get("nctId", ""),
                "title": id_mod.get("briefTitle", ""),
                "phase": ", ".join(matched_phases),
                "status": status,
                "conditions": ", ".join(cond_mod.get("conditions", [])),
                "drug_names": ", ".join(drug_names),
                "enrollment": enroll.get("count", 0),
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                "sponsor": proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
            })

        time.sleep(0.4)

    except Exception as e:
        logger.warning(f"Error for {company_name}: {e}")

    return trials


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    companies_path = f"{RAW_DATA_DIR}/pharma_companies.csv"
    df = pd.read_csv(companies_path)
    print(f"Loaded {len(df)} pharma companies")
    print("Fetching Phase 2 and Phase 3 trials...\n")

    all_trials = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching trials"):
        company_name = str(row.get("name", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if not company_name:
            continue
        trials = fetch_trials_for_company(company_name, ticker)
        all_trials.extend(trials)
        time.sleep(0.3)

    trials_df = pd.DataFrame(all_trials)

    if trials_df.empty:
        print("No trials found.")
        return

    trials_df = trials_df.drop_duplicates(subset=["nct_id", "ticker"])
    out_path = f"{RAW_DATA_DIR}/clinical_trials.csv"
    trials_df.to_csv(out_path, index=False)

    print(f"\nTotal trials: {len(trials_df)}")
    print(f"Unique companies with trials: {trials_df['ticker'].nunique()}")
    print(f"Saved to: {out_path}")
    print(trials_df[["ticker","drug_names","phase","status","conditions"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()