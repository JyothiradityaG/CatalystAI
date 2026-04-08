"""
collectors/pipeline_risk.py
Analyzes pipeline risk for each company.
Run: python -m collectors.pipeline_risk
Output: data/processed/pipeline_risk.csv
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED = "data/processed"
MODELS    = "data/models"

VERIFIED_RISK = {
    "REGN": {"rating": "LOW",    "note": "Dupixent - established drug new indication"},
    "ALNY": {"rating": "LOW",    "note": "Zilebesiran - Priority review"},
    "BIIB": {"rating": "LOW",    "note": "Felzartamab - Priority review"},
    "RARE": {"rating": "MEDIUM", "note": "GTX-102 - Priority review rare disease"},
    "IONS": {"rating": "LOW",    "note": "Donidalorsen - Standard review"},
    "MRNA": {"rating": "LOW",    "note": "mRNA-1283 - Standard review"},
    "VRTX": {"rating": "LOW",    "note": "Suzetrigine - established company"},
    "GILD": {"rating": "LOW",    "note": "Seladelpar - Priority review"},
    "SVRA": {"rating": "MEDIUM", "note": "Molgramostim - Priority review rare disease"},
    "NUVL": {"rating": "LOW",    "note": "Zidesamtinib - Priority review"},
}


def analyze_pipeline_risk():
    scores_path = f"{MODELS}/scores.csv"
    if not os.path.exists(scores_path):
        print("scores.csv not found. Run train_model.py first.")
        return None

    df = pd.read_csv(scores_path)
    print(f"Analyzing pipeline risk for {df['ticker'].nunique()} companies...")

    risk_rows = []

    for ticker, group in df.groupby("ticker"):

        company_name = group["company_name"].iloc[0] \
            if "company_name" in group.columns else ticker

        total_drugs   = len(group)
        avg_score_all = group["approval_score"].mean().round(1)
        max_score     = group["approval_score"].max().round(1)
        min_score     = group["approval_score"].min().round(1)

        if "pdufa_date" in group.columns:
            pdufa_group = group[group["pdufa_date"].notna()].copy()
        else:
            pdufa_group = pd.DataFrame()

        has_pdufa   = len(pdufa_group) > 0
        pdufa_count = len(pdufa_group)

        risky_pdufa_drugs = []
        safe_pdufa_drugs  = []
        all_pdufa_info    = []
        conflict_warning  = ""
        avg_pdufa_score   = 0
        n_risky           = 0
        n_safe            = 0

        if has_pdufa:
            for _, row in pdufa_group.iterrows():
                score = float(row.get("approval_score", 50))
                drug  = str(row.get("pdufa_drug_name", "Unknown"))
                date  = str(row.get("pdufa_date", ""))
                days  = row.get("days_until_decision", 999)

                try:
                    days = int(days)
                except:
                    days = 999

                all_pdufa_info.append({
                    "drug":  drug,
                    "date":  date,
                    "days":  days,
                    "score": score,
                })

                if score <= 30:
                    risky_pdufa_drugs.append(drug)
                elif score >= 80:
                    safe_pdufa_drugs.append(drug)

            all_pdufa_info = sorted(
                all_pdufa_info, key=lambda x: x["days"]
            )

            for i in range(len(all_pdufa_info)):
                for j in range(i + 1, len(all_pdufa_info)):
                    early = all_pdufa_info[i]
                    later = all_pdufa_info[j]
                    if (early["score"] <= 30 and
                            later["score"] >= 80 and
                            early["days"] < later["days"]):
                        conflict_warning = (
                            f"CONFLICT: {early['drug']} "
                            f"({early['days']}d score {early['score']:.0f}) "
                            f"comes BEFORE "
                            f"{later['drug']} "
                            f"({later['days']}d score {later['score']:.0f})"
                        )

            n_risky         = len(risky_pdufa_drugs)
            n_safe          = len(safe_pdufa_drugs)
            avg_pdufa_score = pdufa_group["approval_score"].mean().round(1)

            if ticker in VERIFIED_RISK:
                risk_rating = VERIFIED_RISK[ticker]["rating"]
            elif conflict_warning != "":
                risk_rating = "HIGH"
            elif n_risky == 0:
                risk_rating = "LOW"
            elif n_risky <= 1 and n_safe >= 1:
                risk_rating = "MEDIUM"
            else:
                risk_rating = "HIGH"

        else:
            risk_rating = "NO PDUFA"

        next_pdufa_date  = ""
        next_pdufa_days  = None
        next_pdufa_drug  = ""
        next_pdufa_score = None

        if has_pdufa and all_pdufa_info:
            next_event       = all_pdufa_info[0]
            next_pdufa_date  = next_event["date"]
            next_pdufa_days  = next_event["days"]
            next_pdufa_drug  = next_event["drug"]
            next_pdufa_score = next_event["score"]

        verified_note = VERIFIED_RISK.get(ticker, {}).get("note", "")

        risk_rows.append({
            "ticker":            ticker,
            "company_name":      company_name,
            "total_drugs":       total_drugs,
            "avg_score_all":     avg_score_all,
            "max_score":         max_score,
            "min_score":         min_score,
            "pdufa_count":       pdufa_count,
            "safe_pdufa_drugs":  n_safe,
            "risky_pdufa_drugs": n_risky,
            "avg_pdufa_score":   avg_pdufa_score,
            "next_pdufa_drug":   next_pdufa_drug,
            "next_pdufa_date":   next_pdufa_date,
            "next_pdufa_days":   next_pdufa_days,
            "next_pdufa_score":  next_pdufa_score,
            "risk_rating":       risk_rating,
            "conflict_warning":  conflict_warning,
            "verified_note":     verified_note,
        })

    risk_df = pd.DataFrame(risk_rows)

    has_pdufa_df = risk_df[risk_df["pdufa_count"] > 0].copy()
    has_pdufa_df = has_pdufa_df.sort_values("next_pdufa_days")
    no_pdufa_df  = risk_df[risk_df["pdufa_count"] == 0].copy()
    risk_df      = pd.concat([has_pdufa_df, no_pdufa_df])

    out_path = f"{PROCESSED}/pipeline_risk.csv"
    risk_df.to_csv(out_path, index=False)

    pdufa_companies = risk_df[risk_df["pdufa_count"] > 0]

    print(f"\n{'='*60}")
    print(f"PIPELINE RISK ANALYSIS")
    print(f"{'='*60}")
    print(f"Companies analyzed:     {len(risk_df)}")
    print(f"Companies with PDUFA:   {len(pdufa_companies)}")
    print(f"Low risk:               {len(pdufa_companies[pdufa_companies['risk_rating']=='LOW'])}")
    print(f"Medium risk:            {len(pdufa_companies[pdufa_companies['risk_rating']=='MEDIUM'])}")
    print(f"High risk:              {len(pdufa_companies[pdufa_companies['risk_rating']=='HIGH'])}")

    print(f"\nAll companies with upcoming PDUFA dates:")
    display = [
        "ticker", "next_pdufa_drug", "next_pdufa_days",
        "next_pdufa_score", "risk_rating", "verified_note"
    ]
    available = [c for c in display if c in pdufa_companies.columns]
    print(pdufa_companies[available].to_string(index=False))

    conflicts = risk_df[risk_df["conflict_warning"] != ""]
    if not conflicts.empty:
        print(f"\nConflict Warnings:")
        for _, row in conflicts.iterrows():
            print(f"  {row['ticker']}: {row['conflict_warning']}")

    print(f"\nSaved to: {out_path}")
    return risk_df


if __name__ == "__main__":
    analyze_pipeline_risk()