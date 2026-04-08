"""
train_model.py
Trains XGBoost model to predict FDA approval probability.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

PROCESSED = "data/processed"
MODELS    = "data/models"
os.makedirs(MODELS, exist_ok=True)

print("Loading features...")
df = pd.read_csv(f"{PROCESSED}/features.csv")
print(f"Total trials: {len(df)}")

# ── Training label ────────────────────────────────────────────────────────────
df["label"] = (
    (df["phase_num"] == 3) & (df["is_completed"] == 1)
).astype(int)

print(f"Positive labels (Ph3 completed): {df['label'].sum()}")
print(f"Negative labels:                  {(df['label']==0).sum()}")

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "phase_num",
    "enrollment",
    "is_recruiting",
    "is_active",
    "is_completed",
    "historical_fda_rate",
    "condition_complexity",
]

X = df[FEATURES].fillna(0)
y = df["label"]

# ── Train test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")

# ── Train model ───────────────────────────────────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"5-Fold CV AUC:  {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── Feature importance ────────────────────────────────────────────────────────
print(f"\nFeature Importance:")
importance = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.to_string(index=False))

# ── Save model ────────────────────────────────────────────────────────────────
model_path = f"{MODELS}/approval_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# ── Score all trials ──────────────────────────────────────────────────────────
print("\nScoring all active trials...")
X_all = df[FEATURES].fillna(0)
df["approval_probability"] = model.predict_proba(X_all)[:, 1]
df["approval_score"]       = (df["approval_probability"] * 100).round(1)

# ── Output columns including PDUFA dates ─────────────────────────────────────
wanted_cols = [
    "ticker", "company_name", "drug_names", "conditions",
    "phase", "status", "enrollment",
    "approval_score", "approval_probability",
    "historical_fda_rate", "condition_complexity",
    "start_date", "completion_date", "nct_id",
    "pdufa_date", "days_until_decision",
    "pdufa_drug_name", "pdufa_condition", "review_priority",
]

output_cols = [c for c in wanted_cols if c in df.columns]
scores      = df[output_cols].sort_values(
    "approval_score", ascending=False
)

scores_path = f"{MODELS}/scores.csv"
scores.to_csv(scores_path, index=False)

print(f"\n{'='*60}")
print("TOP 20 DRUGS BY APPROVAL PROBABILITY")
print(f"{'='*60}")
top20 = scores.head(20)[
    ["ticker","drug_names","phase","status","approval_score","conditions"]
]
print(top20.to_string(index=False))
print(f"\nDrugs with PDUFA dates: {scores['pdufa_date'].notna().sum()}")
print(f"All scores saved to: {scores_path}")