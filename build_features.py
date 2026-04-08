"""
build_features.py
Combines all datasets into one feature table for the ML model.
Now includes PDUFA dates.
Run: python build_features.py
Output: data/processed/features.csv
"""

import pandas as pd
import os

RAW       = "data/raw"
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

print("Loading data...")
companies = pd.read_csv(f"{RAW}/pharma_companies.csv")
trials    = pd.read_csv(f"{RAW}/clinical_trials.csv")
fda       = pd.read_csv(f"{RAW}/fda_approvals.csv")

print(f"Companies: {len(companies)}")
print(f"Trials:    {len(trials)}")
print(f"FDA records: {len(fda)}")

# ── FDA approval rate per sponsor ─────────────────────────────────────────────
fda_stats = fda.groupby("sponsor_name").agg(
    total_submissions=("approved", "count"),
    total_approved=("approved", "sum"),
).reset_index()
fda_stats["historical_approval_rate"] = (
    fda_stats["total_approved"] / fda_stats["total_submissions"]
).round(3)

# ── Merge trials with companies ───────────────────────────────────────────────
df = trials.merge(
    companies[["ticker", "name", "exchange"]],
    on="ticker", how="left"
)

# ── Phase numeric feature ─────────────────────────────────────────────────────
df["phase_num"] = df["phase"].apply(
    lambda x: 3 if "PHASE3" in str(x) else 2 if "PHASE2" in str(x) else 0
)

# ── Enrollment as numeric ─────────────────────────────────────────────────────
df["enrollment"] = pd.to_numeric(
    df["enrollment"], errors="coerce"
).fillna(0)

# ── Status flags ──────────────────────────────────────────────────────────────
df["is_recruiting"] = (df["status"] == "RECRUITING").astype(int)
df["is_active"]     = (df["status"] == "ACTIVE_NOT_RECRUITING").astype(int)
df["is_completed"]  = (df["status"] == "COMPLETED").astype(int)

# ── Match sponsor name to FDA history ─────────────────────────────────────────
def match_fda_rate(company_name):
    name       = str(company_name).upper()
    first_word = name.split()[0] if name else ""
    match      = fda_stats[
        fda_stats["sponsor_name"].str.upper().str.contains(
            first_word, na=False
        )
    ]
    if not match.empty:
        return match.iloc[0]["historical_approval_rate"]
    return fda_stats["historical_approval_rate"].mean()

print("\nMatching companies to FDA history...")
df["historical_fda_rate"] = df["company_name"].apply(match_fda_rate)

# ── Condition complexity score ────────────────────────────────────────────────
hard_conditions = [
    "cancer", "oncol", "tumor", "carcinoma", "leukemia",
    "lymphoma", "glioma", "alzheimer", "parkinson"
]

def condition_complexity(conditions):
    c = str(conditions).lower()
    return sum(1 for h in hard_conditions if h in c)

df["condition_complexity"] = df["conditions"].apply(condition_complexity)

# ── Merge PDUFA dates ─────────────────────────────────────────────────────────
pdufa_path = f"{RAW}/pdufa_dates.csv"
if os.path.exists(pdufa_path):
    print("Merging PDUFA dates...")
    pdufa_df = pd.read_csv(pdufa_path)
    pdufa_df = pdufa_df[[
        "ticker", "brand_name", "pdufa_date",
        "days_until_decision", "condition", "review_priority"
    ]].rename(columns={
        "brand_name": "pdufa_drug_name",
        "condition":  "pdufa_condition",
    })
    df = df.merge(pdufa_df, on="ticker", how="left")
    pdufa_count = df["pdufa_date"].notna().sum()
    print(f"PDUFA dates merged: {pdufa_count} drugs have upcoming decisions")
else:
    print("No PDUFA dates file found — skipping merge")
    df["pdufa_date"]           = None
    df["days_until_decision"]  = None
    df["pdufa_drug_name"]      = None
    df["pdufa_condition"]      = None
    df["review_priority"]      = None

# ── Select final feature columns ──────────────────────────────────────────────
feature_cols = [
    "ticker", "company_name", "nct_id", "title",
    "drug_names", "phase", "phase_num", "status",
    "conditions", "enrollment",
    "is_recruiting", "is_active", "is_completed",
    "historical_fda_rate", "condition_complexity",
    "start_date", "completion_date", "sponsor",
    "pdufa_date", "days_until_decision",
    "pdufa_drug_name", "pdufa_condition", "review_priority",
]

available = [c for c in feature_cols if c in df.columns]
features  = df[available].copy()

out_path  = f"{PROCESSED}/features.csv"
features.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"FEATURE TABLE BUILT")
print(f"{'='*60}")
print(f"Total drug trials:         {len(features)}")
print(f"Phase 2 trials:            {len(features[features['phase_num']==2])}")
print(f"Phase 3 trials:            {len(features[features['phase_num']==3])}")
print(f"Drugs with PDUFA dates:    {features['pdufa_date'].notna().sum()}")
print(f"Avg enrollment:            {features['enrollment'].mean():.0f}")
print(f"\nTop 10 companies by trial count:")
print(features['ticker'].value_counts().head(10).to_string())
print(f"\nUpcoming PDUFA decisions in dashboard:")
pdufa_view = features[features['pdufa_date'].notna()][
    ["ticker","pdufa_drug_name","pdufa_date","days_until_decision"]
].drop_duplicates().sort_values("days_until_decision")
print(pdufa_view.to_string(index=False))
print(f"\nSaved to: {out_path}")