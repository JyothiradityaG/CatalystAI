"""
dashboard/app.py
CatalystAI — FDA Approval Prediction Dashboard v2
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(
    page_title="CatalystAI — FDA Approval Predictor",
    page_icon="💊",
    layout="wide"
)

@st.cache_data
def load_data():
    scores_path = "data/models/scores.csv"
    if not os.path.exists(scores_path):
        return None
    df = pd.read_csv(scores_path)
    df["approval_score"] = df["approval_score"].round(1)
    df["signal"] = df["approval_score"].apply(
        lambda x: "STRONG BUY CALL" if x >= 80
        else "CONSIDER CALL"        if x >= 60
        else "CONSIDER PUT"         if x <= 30
        else "NEUTRAL"
    )
    return df

@st.cache_data
def load_risk():
    risk_path = "data/processed/pipeline_risk.csv"
    if not os.path.exists(risk_path):
        return None
    return pd.read_csv(risk_path)

df   = load_data()
risk = load_risk()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💊 CatalystAI — FDA Drug Approval Predictor")
st.markdown(
    "Predicts FDA approval probability for Phase 2 and Phase 3 "
    "clinical trials — linked to options trading signals."
)

if df is None:
    st.error("No data found. Run train_model.py first.")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trials",         len(df))
col2.metric("Phase 3 Trials",
            len(df[df["phase"].str.contains("PHASE3", na=False)]))
col3.metric("High Probability >80", len(df[df["approval_score"] >= 80]))
col4.metric("Low Probability <30",  len(df[df["approval_score"] <= 30]))
col5.metric("With PDUFA Dates",
            int(df["pdufa_date"].notna().sum())
            if "pdufa_date" in df.columns else 0)

st.divider()

# ── PDUFA COUNTDOWN ───────────────────────────────────────────────────────────
st.subheader("Upcoming FDA Decision Dates")

if "pdufa_date" in df.columns:
    pdufa_df = df[df["pdufa_date"].notna()].copy()
    pdufa_df = pdufa_df.sort_values("approval_score", ascending=False)
    pdufa_df = pdufa_df.drop_duplicates(
        subset=["ticker", "pdufa_drug_name", "pdufa_date"],
        keep="first"
    )
    pdufa_df["pdufa_date"] = pd.to_datetime(
        pdufa_df["pdufa_date"], errors="coerce"
    )
    pdufa_df = pdufa_df.sort_values("days_until_decision")

    if not pdufa_df.empty:
        for _, row in pdufa_df.head(10).iterrows():
            days  = int(row.get("days_until_decision", 0))
            score = float(row.get("approval_score", 0))
            sig   = row.get("signal", "NEUTRAL")

            if sig == "STRONG BUY CALL":
                color = "#E1F5EE"; tc = "#085041"
                trade = "BUY CALL OPTIONS"
            elif sig == "CONSIDER CALL":
                color = "#EAF3DE"; tc = "#27500A"
                trade = "CONSIDER CALL"
            elif sig == "CONSIDER PUT":
                color = "#FCEBEB"; tc = "#791F1F"
                trade = "BUY PUT OPTIONS"
            else:
                color = "#F1EFE8"; tc = "#5F5E5A"
                trade = "NEUTRAL"

            pdufa_date_str = pd.to_datetime(
                row.get("pdufa_date")
            ).strftime("%B %d, %Y") if pd.notna(
                row.get("pdufa_date")
            ) else "TBD"

            st.markdown(
                f"""
                <div style="background:{color};border-radius:12px;
                padding:14px 18px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                <span style="font-size:18px;font-weight:500;
                color:{tc};">{row.get('ticker','')} — 
                {row.get('pdufa_drug_name','')}</span><br>
                <span style="font-size:13px;color:{tc};opacity:0.8;">
                {row.get('pdufa_condition','')}</span>
                </div>
                <div style="text-align:right;">
                <span style="font-size:22px;font-weight:500;
                color:{tc};">{days} days</span><br>
                <span style="font-size:12px;color:{tc};opacity:0.8;">
                {pdufa_date_str}</span>
                </div>
                <div style="text-align:center;">
                <span style="font-size:20px;font-weight:500;
                color:{tc};">Score: {score}</span><br>
                <span style="font-size:12px;font-weight:500;
                color:{tc};">{trade}</span>
                </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No upcoming PDUFA dates found.")
else:
    st.info("Run pdufa_dates.py and build_features.py to see PDUFA dates.")

st.divider()

# ── PIPELINE RISK ANALYSIS ────────────────────────────────────────────────────
st.subheader("Pipeline Risk Analysis")
st.caption(
    "Flags companies with a risky FDA decision coming BEFORE "
    "a high-scoring decision — protects your options position."
)

if risk is not None:
    pdufa_risk = risk[risk["pdufa_count"] > 0].copy()

    if not pdufa_risk.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Low Risk",
                  len(pdufa_risk[pdufa_risk["risk_rating"] == "LOW"]))
        c2.metric("Medium Risk",
                  len(pdufa_risk[pdufa_risk["risk_rating"] == "MEDIUM"]))
        c3.metric("High Risk",
                  len(pdufa_risk[pdufa_risk["risk_rating"] == "HIGH"]))

        for _, row in pdufa_risk.iterrows():
            rating  = row.get("risk_rating", "")
            ticker  = row.get("ticker", "")
            drug    = row.get("next_pdufa_drug", "")
            days    = row.get("next_pdufa_days", 0)
            score   = row.get("next_pdufa_score", 0)
            note    = row.get("verified_note", "")
            warning = str(row.get("conflict_warning", ""))

            if rating == "LOW":
                bg = "#E1F5EE"; tc = "#085041"; badge = "LOW RISK"
            elif rating == "MEDIUM":
                bg = "#FAEEDA"; tc = "#633806"; badge = "MEDIUM RISK"
            else:
                bg = "#FCEBEB"; tc = "#791F1F"; badge = "HIGH RISK"

            warning_html = ""
            if warning and warning != "nan":
                warning_html = (
                    f'<br><span style="font-size:11px;color:#791F1F;'
                    f'font-weight:500;">WARNING: {warning}</span>'
                )

            st.markdown(
                f"""
                <div style="background:{bg};border-radius:10px;
                padding:12px 16px;margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:6px;">
                <div>
                <span style="font-size:15px;font-weight:500;
                color:{tc};">{ticker} — {drug}</span><br>
                <span style="font-size:12px;color:{tc};opacity:0.8;">
                {note}</span>{warning_html}
                </div>
                <div style="text-align:right;">
                <span style="font-size:18px;font-weight:500;
                color:{tc};">{int(days) if days else 0} days</span><br>
                <span style="font-size:11px;font-weight:500;
                color:{tc};">{badge}</span>
                </div>
                <div style="text-align:center;">
                <span style="font-size:16px;font-weight:500;
                color:{tc};">Score: {score}</span>
                </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Run pipeline_risk.py to see risk analysis.")

st.divider()

# ── COMPANY PIPELINE VIEW ─────────────────────────────────────────────────────
st.subheader("Company Pipeline View")
st.caption(
    "All drugs grouped by company — sorted by days until FDA decision. "
    "Shows full pipeline at a glance."
)

if "pdufa_date" in df.columns:
    company_pdufa = df[df["pdufa_date"].notna()].copy()

    # Keep only highest scoring row per ticker per drug per date
    company_pdufa = company_pdufa.sort_values(
        "approval_score", ascending=False
    )
    company_pdufa = company_pdufa.drop_duplicates(
        subset=["ticker", "pdufa_drug_name", "pdufa_date"],
        keep="first"
    )
    company_pdufa = company_pdufa.sort_values(
        ["ticker", "days_until_decision"]
    )

    if not company_pdufa.empty:
        tickers = company_pdufa["ticker"].unique()

        for ticker in tickers:
            ticker_df  = company_pdufa[
                company_pdufa["ticker"] == ticker
            ].copy()
            company_nm = ticker_df["company_name"].iloc[0] \
                if "company_name" in ticker_df.columns else ticker
            drug_count = len(ticker_df)

            st.markdown(
                f"""
                <div style="border-left:4px solid #185FA5;
                padding:8px 14px;margin-bottom:4px;margin-top:14px;">
                <span style="font-size:16px;font-weight:500;
                color:#185FA5;">{ticker}</span>
                <span style="font-size:13px;color:#888;
                margin-left:8px;">{company_nm}</span>
                <span style="font-size:12px;color:#888;
                margin-left:8px;">— {drug_count} decision(s)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            for i, (_, row) in enumerate(ticker_df.iterrows()):
                score = float(row.get("approval_score", 0))
                days  = int(row.get("days_until_decision", 0))
                sig   = row.get("signal", "NEUTRAL")
                drug  = str(row.get("pdufa_drug_name", "Unknown"))
                cond  = str(row.get("pdufa_condition", ""))
                date  = str(row.get("pdufa_date", ""))
                rev   = str(row.get("review_priority", ""))

                try:
                    date_fmt = pd.to_datetime(date).strftime("%b %d %Y")
                except:
                    date_fmt = date

                if sig == "STRONG BUY CALL":
                    bg = "#E1F5EE"; tc = "#085041"
                    sb = "BUY CALL"; sc = "#085041"
                elif sig == "CONSIDER CALL":
                    bg = "#EAF3DE"; tc = "#27500A"
                    sb = "CONSIDER CALL"; sc = "#27500A"
                elif sig == "CONSIDER PUT":
                    bg = "#FCEBEB"; tc = "#791F1F"
                    sb = "BUY PUT"; sc = "#791F1F"
                else:
                    bg = "#F1EFE8"; tc = "#5F5E5A"
                    sb = "NEUTRAL"; sc = "#5F5E5A"

                is_last   = (i == len(ticker_df) - 1)
                prefix    = "└──" if is_last else "├──"
                border_r  = "0 0 10px 10px" if is_last else "0"

                st.markdown(
                    f"""
                    <div style="background:{bg};
                    border-radius:{border_r};
                    padding:10px 14px 10px 24px;
                    margin-bottom:2px;">
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:6px;">
                    <div style="flex:2;min-width:180px;">
                    <span style="font-size:13px;color:#888;
                    margin-right:6px;">{prefix}</span>
                    <span style="font-size:14px;font-weight:500;
                    color:{tc};">{drug}</span><br>
                    <span style="font-size:11px;color:{tc};
                    opacity:0.75;margin-left:24px;">{cond}</span>
                    </div>
                    <div style="text-align:center;min-width:90px;">
                    <span style="font-size:13px;color:{tc};">
                    {date_fmt}</span>
                    </div>
                    <div style="text-align:center;min-width:60px;">
                    <span style="font-size:14px;font-weight:500;
                    color:{tc};">{days}d</span>
                    </div>
                    <div style="text-align:center;min-width:80px;">
                    <span style="font-size:14px;font-weight:500;
                    color:{tc};">Score: {score}</span>
                    </div>
                    <div style="text-align:center;min-width:110px;">
                    <span style="font-size:12px;font-weight:500;
                    padding:3px 10px;border-radius:20px;
                    background:white;color:{sc};">{sb}</span>
                    </div>
                    <div style="text-align:center;min-width:80px;">
                    <span style="font-size:11px;color:{tc};
                    opacity:0.7;">{rev}</span>
                    </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No drugs with PDUFA dates found.")
else:
    st.info("Run pdufa_dates.py and build_features.py first.")

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

with col1:
    phase_filter = st.multiselect(
        "Phase",
        options=["PHASE2","PHASE3"],
        default=["PHASE2","PHASE3"]
    )
with col2:
    status_filter = st.multiselect(
        "Trial Status",
        options=df["status"].unique().tolist(),
        default=df["status"].unique().tolist()
    )
with col3:
    signal_filter = st.multiselect(
        "Trading Signal",
        options=["STRONG BUY CALL","CONSIDER CALL",
                 "NEUTRAL","CONSIDER PUT"],
        default=["STRONG BUY CALL","CONSIDER CALL",
                 "NEUTRAL","CONSIDER PUT"]
    )

score_range = st.slider(
    "Approval Score Range",
    min_value=0, max_value=100, value=(0, 100)
)

pdufa_only = st.checkbox(
    "Show only drugs with PDUFA dates", value=False
)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[
    df["phase"].str.contains("|".join(phase_filter), na=False) &
    df["status"].isin(status_filter) &
    df["signal"].isin(signal_filter) &
    df["approval_score"].between(score_range[0], score_range[1])
].copy()

if pdufa_only and "pdufa_date" in filtered.columns:
    filtered = filtered[filtered["pdufa_date"].notna()]

st.caption(f"Showing {len(filtered)} trials")

# ── Score distribution ────────────────────────────────────────────────────────
st.subheader("Approval Score Distribution")
fig = px.histogram(
    filtered, x="approval_score", nbins=20,
    color_discrete_sequence=["#185FA5"],
    labels={"approval_score": "Approval Score"}
)
fig.update_layout(
    height=250, margin=dict(t=10, b=10),
    plot_bgcolor="white", paper_bgcolor="white"
)
st.plotly_chart(fig, use_container_width=True)

# ── Main table ────────────────────────────────────────────────────────────────
st.subheader("Drug Pipeline — All Scored Trials")

base_cols  = [
    "ticker","company_name","drug_names","conditions",
    "phase","status","enrollment","approval_score","signal"
]
pdufa_cols = ["pdufa_date","days_until_decision"]
disp_cols  = base_cols + [
    c for c in pdufa_cols if c in filtered.columns
]
avail_cols = [c for c in disp_cols if c in filtered.columns]

def color_signal(val):
    colors = {
        "STRONG BUY CALL": "background-color:#E1F5EE;color:#085041",
        "CONSIDER CALL":   "background-color:#EAF3DE;color:#27500A",
        "NEUTRAL":         "background-color:#F1EFE8;color:#5F5E5A",
        "CONSIDER PUT":    "background-color:#FCEBEB;color:#791F1F",
    }
    return colors.get(val, "")

def color_score(val):
    try:
        v = float(val)
        if v >= 80:   return "color:#085041;font-weight:bold"
        elif v <= 30: return "color:#791F1F;font-weight:bold"
        return ""
    except:
        return ""

styled = (
    filtered[avail_cols]
    .sort_values("approval_score", ascending=False)
    .style
    .map(color_signal, subset=["signal"])
    .map(color_score,  subset=["approval_score"])
)
st.dataframe(styled, use_container_width=True, height=500)

st.divider()

# ── Top opportunities ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Call Candidates")
    top_calls = filtered[filtered["approval_score"] >= 80][
        ["ticker","drug_names","approval_score","conditions"]
    ].head(10)
    st.dataframe(top_calls, use_container_width=True)

with col2:
    st.subheader("Top Put Candidates")
    top_puts = filtered[filtered["approval_score"] <= 30][
        ["ticker","drug_names","approval_score","conditions"]
    ].sort_values("approval_score").head(10)
    st.dataframe(top_puts, use_container_width=True)

st.divider()

# ── Company score chart ───────────────────────────────────────────────────────
st.subheader("Average Approval Score by Company")
company_scores = (
    filtered.groupby("ticker")["approval_score"]
    .mean().round(1)
    .sort_values(ascending=False)
    .head(20).reset_index()
)
fig2 = px.bar(
    company_scores, x="ticker", y="approval_score",
    color="approval_score",
    color_continuous_scale=["#E24B4A","#888780","#1D9E75"],
    labels={"ticker":"Company","approval_score":"Avg Score"},
)
fig2.update_layout(
    height=300, margin=dict(t=10, b=10),
    plot_bgcolor="white", paper_bgcolor="white",
    coloraxis_showscale=False
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "CatalystAI v2 — Built for Cove Care Partners — "
    "Quantitative Innovation"
)