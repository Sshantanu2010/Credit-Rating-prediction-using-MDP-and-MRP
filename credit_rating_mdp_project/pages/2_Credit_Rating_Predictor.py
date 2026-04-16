"""
Page 2 — Credit Rating Predictor: company search, manual input, CSV upload,
prediction, confidence chart, feature importance, and company comparison.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    FINE_RATINGS, SECTORS, NUMERIC_FEATURES,
    RATING_TO_BUCKET, get_risk_category, get_color_for_rating,
    get_rating_outlook, get_recommendation,
)
from src.data_preprocessing import load_dataset, lookup_company, clean_dataframe, load_transition_matrix
from src.train_rating_model import is_model_trained, train_and_select
from src.predict_rating import predict_single, predict_batch, explain_rating
from src.feature_engineering import load_preprocessing_artifacts
from src.mrp_model import MRP
from src.mdp_model import MDP

st.set_page_config(page_title="Rating Predictor", page_icon="📊", layout="wide")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0066cc,#003d7a);padding:1.5rem 2rem;border-radius:12px;color:white;margin-bottom:1.5rem;">
<h1 style="color:white;margin:0;">📊 Credit Rating Predictor</h1>
<p style="color:#b3d4fc;">Search a company, enter financials, or upload CSV</p>
</div>
""", unsafe_allow_html=True)

# Ensure model exists
if not is_model_trained():
    with st.spinner("Training models for the first time …"):
        train_and_select()

# Load dataset for lookup
dataset = load_dataset()


# ═══════════════════════════════════════════════════════════════
# SHARED: display prediction results  (defined BEFORE usage)
# ═══════════════════════════════════════════════════════════════
def _show_prediction(result: dict, company_data: dict, name: str):
    """Render prediction outcome with charts and MRP/MDP summary."""
    st.divider()
    st.markdown(f"## Results for **{name}**")

    # ── Key metrics row ──
    risk = result['risk_category']
    risk_colors = {
        'Low Risk': '#28a745', 'Moderate Risk': '#ffc107',
        'High Risk': '#fd7e14', 'Default Risk': '#dc3545',
    }
    rc = risk_colors.get(risk, '#6c757d')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Rating", result['predicted_rating'])
    c2.metric("Bucket", result['predicted_bucket'])
    c3.metric("Confidence", f"{result['confidence']*100:.1f}%")
    c4.markdown(
        f"<div style='background:{rc};color:white;padding:12px 16px;"
        f"border-radius:8px;text-align:center;font-weight:bold;'>"
        f"{risk}</div>", unsafe_allow_html=True
    )

    # ── Probability distribution ──
    probs = result['probabilities']
    prob_df = pd.DataFrame({'Rating': list(probs.keys()),
                            'Probability': list(probs.values())})
    prob_df = prob_df.sort_values('Probability', ascending=True).tail(10)

    fig_prob = px.bar(prob_df, x='Probability', y='Rating', orientation='h',
                      title="Class Probability Distribution (top 10)",
                      color='Probability',
                      color_continuous_scale='Blues')
    fig_prob.update_layout(template="plotly_white", height=380)

    # ── Feature importance ──
    fi = result.get('feature_importance', {})
    fi_df = pd.DataFrame({'Feature': list(fi.keys()),
                          'Importance': list(fi.values())})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)

    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance",
                     color='Importance', color_continuous_scale='Viridis')
    fig_fi.update_layout(template="plotly_white", height=380)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_prob, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Explainability ──
    explanations = explain_rating(result)
    with st.expander("💡 Rating Explanation", expanded=True):
        for e in explanations:
            st.markdown(f"- {e}")

    # ── Quick MRP + MDP summary ──
    bucket = result['predicted_bucket']
    try:
        tm = load_transition_matrix()
        mrp = MRP(tm, gamma=0.95)
        mrp.compute_state_values()
        summary = mrp.migration_summary(bucket, horizon=1)
        interp = summary['interpretation']

        mdp = MDP(tm.values, gamma=0.95)
        mdp.value_iteration()
        actions = mdp.get_policy_for_rating(bucket)

        outlook = get_rating_outlook(bucket, tm.loc[bucket].values)

        st.divider()
        st.markdown("### 🔄 Quick MRP & MDP Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stay in " + bucket, f"{interp['stay_probability']*100:.1f}%")
        m2.metric("Upgrade Prob", f"{interp['upgrade_probability']*100:.1f}%")
        m3.metric("Downgrade Prob", f"{interp['downgrade_probability']*100:.1f}%")
        m4.metric("Default Prob", f"{interp['default_probability']*100:.2f}%")

        st.markdown(f"**Rating Outlook:** {outlook}")
        st.markdown("**MDP Recommended Actions:**")
        for exp, act in actions.items():
            emoji = {'Increase': '📈', 'Hold': '📊', 'Reduce': '📉', 'Exit': '🚫'}
            st.write(f"  Exposure = {exp} → {emoji.get(act, '')} **{act}**")

        rec = get_recommendation(bucket, actions.get('Medium', 'Hold'), outlook)
        st.info(rec)
    except Exception as e:
        st.warning(f"MRP/MDP summary unavailable: {e}")

    # Store in session for reports
    st.session_state['last_prediction'] = {
        'company_name': name,
        'company_data': company_data,
        'result': result,
    }


# ── Tabs ──────────────────────────────────────────────────────
tab_search, tab_manual, tab_csv, tab_compare = st.tabs([
    "🔍 Company Search", "✏️ Manual Input", "📁 CSV Upload", "⚖️ Compare Companies"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Company Search
# ═══════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### Search an Indian Company")
    company_name = st.text_input(
        "Enter company name (e.g. Tata Motors, Infosys, Reliance)",
        key="company_search",
    )

    if company_name:
        match = lookup_company(company_name, dataset)
        if match is not None:
            st.success(f"✅ Found **{match['company_name']}** in the dataset!")
            with st.expander("📋 Auto-loaded financial data", expanded=True):
                fin_cols = ['sector'] + NUMERIC_FEATURES + ['credit_rating']
                display_data = {c: str(match[c]) for c in fin_cols if c in match.index}
                st.json(display_data)

            company_dict = match.to_dict()
            result = predict_single(company_dict)
            _show_prediction(result, company_dict, match['company_name'])
        else:
            st.warning(
                f"❌ **{company_name}** not found in the dataset.  "
                "Please use the **Manual Input** tab to enter financials."
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Manual Input
# ═══════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown("### Enter Company Financials Manually")
    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_name   = st.text_input("Company Name", "My Company Ltd")
            m_sector = st.selectbox("Sector", SECTORS)
            m_rev    = st.number_input("Revenue (₹ Cr)", value=15000.0, step=500.0)
            m_ebitda = st.number_input("EBITDA (₹ Cr)", value=3000.0, step=100.0)
            m_ebitda_m = st.number_input("EBITDA Margin (%)", value=20.0, step=1.0)
            m_npm    = st.number_input("Net Profit Margin (%)", value=10.0, step=1.0)
            m_de     = st.number_input("Debt-to-Equity", value=0.8, step=0.1)
        with c2:
            m_ic     = st.number_input("Interest Coverage", value=5.0, step=0.5)
            m_cr     = st.number_input("Current Ratio", value=1.5, step=0.1)
            m_qr     = st.number_input("Quick Ratio", value=1.1, step=0.1)
            m_roa    = st.number_input("ROA (%)", value=7.0, step=0.5)
            m_roe    = st.number_input("ROE (%)", value=14.0, step=0.5)
            m_ocf    = st.number_input("Operating Cash Flow (₹ Cr)", value=2000.0, step=100.0)
        with c3:
            m_td     = st.number_input("Total Debt (₹ Cr)", value=12000.0, step=500.0)
            m_dscr   = st.number_input("DSCR", value=2.0, step=0.1)
            m_ph     = st.number_input("Promoter Holding (%)", value=52.0, step=1.0)
            m_age    = st.number_input("Company Age (yrs)", value=25, step=1)
            m_mc     = st.number_input("Market Cap (₹ Cr)", value=30000.0, step=1000.0)
            m_macro  = st.selectbox("External Macro Risk Flag", [0, 1])
            m_qual   = st.slider("Qualitative Risk Score (1-10)", 1.0, 10.0, 4.0, 0.5)

        submitted = st.form_submit_button("🚀 Predict Rating", use_container_width=True)

    if submitted:
        company_dict = {
            'company_name': m_name, 'sector': m_sector,
            'revenue': m_rev, 'ebitda': m_ebitda,
            'ebitda_margin': m_ebitda_m, 'net_profit_margin': m_npm,
            'debt_to_equity': m_de, 'interest_coverage': m_ic,
            'current_ratio': m_cr, 'quick_ratio': m_qr,
            'roa': m_roa, 'roe': m_roe,
            'operating_cash_flow': m_ocf, 'total_debt': m_td,
            'dscr': m_dscr, 'promoter_holding': m_ph,
            'company_age': m_age, 'market_cap': m_mc,
            'external_macro_risk_flag': m_macro,
            'qualitative_risk_score': m_qual,
        }
        result = predict_single(company_dict)
        _show_prediction(result, company_dict, m_name)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — CSV Upload
# ═══════════════════════════════════════════════════════════════
with tab_csv:
    st.markdown("### Upload a CSV of companies")
    up = st.file_uploader("Upload CSV", type=['csv'], key="csv_upload")
    if up:
        df_up = pd.read_csv(up)
        st.write(f"Loaded **{len(df_up)}** rows.")
        st.dataframe(df_up.head(), use_container_width=True)
        if st.button("Predict All", key="predict_batch"):
            with st.spinner("Predicting …"):
                df_result = predict_batch(df_up)
            st.success("Done!")
            st.dataframe(df_result[['company_name', 'predicted_rating',
                                     'risk_category', 'confidence']].head(20),
                         use_container_width=True)

            # Rating distribution
            fig = px.histogram(df_result, x='predicted_rating',
                               category_orders={'predicted_rating': FINE_RATINGS},
                               color='predicted_rating',
                               title="Predicted Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = df_result.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — Company Comparison
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Compare 2–3 Companies Side-by-Side")
    all_names = sorted(dataset['company_name'].unique())
    selected = st.multiselect("Select companies", all_names, default=all_names[:3],
                              max_selections=3, key="compare_select")
    if selected:
        compare_results = []
        for name in selected:
            row = lookup_company(name, dataset)
            if row is not None:
                res = predict_single(row.to_dict())
                res['company_name'] = name
                res['sector'] = row.get('sector', '—')
                compare_results.append(res)

        if compare_results:
            cols = st.columns(len(compare_results))
            for i, res in enumerate(compare_results):
                with cols[i]:
                    risk = res['risk_category']
                    color_map = {'Low Risk': '🟢', 'Moderate Risk': '🟡',
                                 'High Risk': '🟠', 'Default Risk': '🔴'}
                    st.markdown(f"#### {res['company_name']}")
                    st.metric("Predicted Rating", res['predicted_rating'])
                    st.metric("Confidence", f"{res['confidence']*100:.1f}%")
                    st.write(f"{color_map.get(risk, '')} {risk}")
                    st.write(f"Sector: {res['sector']}")

            # Side-by-side confidence comparison
            fig = go.Figure()
            for res in compare_results:
                ratings = list(res['probabilities'].keys())
                probs = list(res['probabilities'].values())
                fig.add_trace(go.Bar(
                    name=res['company_name'], x=ratings, y=probs
                ))
            fig.update_layout(
                title="Confidence Distribution Comparison",
                barmode='group', xaxis_title="Rating", yaxis_title="Probability",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
