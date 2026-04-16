"""
Page 3 — MRP Transition Model: transition matrix heatmap, multi-step
transitions, state values, default probability over time, and gamma sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.utils import BUCKETED_RATINGS, get_rating_outlook, RATING_COLORS
from src.data_preprocessing import load_transition_matrix
from src.mrp_model import MRP, DEFAULT_REWARDS

st.set_page_config(page_title="MRP Transition Model", page_icon="🔄", layout="wide")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#00796b,#004d40);padding:1.5rem 2rem;border-radius:12px;color:white;margin-bottom:1.5rem;">
<h1 style="color:white;margin:0;">🔄 Markov Reward Process — Transition Model</h1>
<p style="color:#b2dfdb;">Rating migration dynamics, state values, and default probabilities</p>
</div>
""", unsafe_allow_html=True)

# ── Load transition matrix ────────────────────────────────────
tm = load_transition_matrix()

# ── Sidebar controls ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### MRP Settings")
    gamma = st.slider("Discount factor (γ)", 0.50, 0.99, 0.95, 0.01, key="mrp_gamma")
    horizon = st.slider("Transition horizon (years)", 1, 20, 1, key="mrp_horizon")
    max_years_dp = st.slider("Default-prob horizon", 5, 30, 15, key="mrp_dp_years")
    focus_state = st.selectbox("Focus state", BUCKETED_RATINGS, index=2, key="mrp_focus")

# Build MRP
mrp = MRP(tm, gamma=gamma)
mrp.compute_state_values()

# ═══════════════════════════════════════════════════════════════
# 1. Transition Matrix Heatmap
# ═══════════════════════════════════════════════════════════════
st.markdown("## 1️⃣ Annual Transition Matrix")

col1, col2 = st.columns([3, 2])
with col1:
    fig_tm = go.Figure(data=go.Heatmap(
        z=np.round(tm.values, 4).tolist(),
        x=list(BUCKETED_RATINGS),
        y=list(BUCKETED_RATINGS),
        text=np.round(tm.values, 3).tolist(),
        texttemplate="%{text}",
        colorscale='Blues',
        hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.4f}<extra></extra>",
    ))
    fig_tm.update_layout(
        title="One-Year Transition Probability Matrix",
        xaxis_title="To",
        yaxis_title="From",
        xaxis=dict(type='category'),
        yaxis=dict(type='category', autorange='reversed'),
        height=480,
        template="plotly_white",
    )
    st.plotly_chart(fig_tm, use_container_width=True)

with col2:
    st.markdown("### Key Properties")
    st.markdown("""
    - Each row sums to **1.0** (complete probability distribution)
    - **Diagonal** = probability of staying in the same rating
    - **D row** is all zeros except D→D = 1.0 (**absorbing state**)
    - Off-diagonal entries represent **upgrade** (left) and **downgrade** (right) risk
    """)
    st.markdown("### Reward Vector")
    reward_df = pd.DataFrame({
        'State': BUCKETED_RATINGS,
        'Reward (₹ Cr)': [DEFAULT_REWARDS[s] for s in BUCKETED_RATINGS],
    })
    st.dataframe(reward_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 2. Multi-Step Transition
# ═══════════════════════════════════════════════════════════════
st.markdown(f"## 2️⃣ Multi-Step Transition Matrix (k = {horizon} year{'s' if horizon > 1 else ''})")

Pk = mrp.multi_step_transition_df(horizon)

fig_pk = go.Figure(data=go.Heatmap(
    z=np.round(Pk.values, 4).tolist(),
    x=list(BUCKETED_RATINGS),
    y=list(BUCKETED_RATINGS),
    text=np.round(Pk.values, 3).tolist(),
    texttemplate="%{text}",
    colorscale='Oranges',
    hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.4f}<extra></extra>",
))
fig_pk.update_layout(
    title=f"{horizon}-Year Transition Probabilities",
    xaxis_title="To",
    yaxis_title="From",
    xaxis=dict(type='category'),
    yaxis=dict(type='category', autorange='reversed'),
    height=460,
    template="plotly_white",
)
st.plotly_chart(fig_pk, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 3. State Values
# ═══════════════════════════════════════════════════════════════
st.markdown("## 3️⃣ State Value Function  V(s)")

col_v1, col_v2 = st.columns([3, 2])
with col_v1:
    sv_df = mrp.get_state_value_df()
    fig_sv = go.Figure()
    for i, row in sv_df.iterrows():
        fig_sv.add_trace(go.Bar(
            x=[row['State']],
            y=[row['Value']],
            marker_color=RATING_COLORS[i],
            name=row['State'],
            showlegend=False,
            hovertemplate=f"{row['State']}: {row['Value']:.2f}<extra></extra>",
        ))
    fig_sv.update_layout(
        title=f"State Values (γ = {gamma})",
        xaxis=dict(type='category', title='State'),
        yaxis=dict(title='Value'),
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_sv, use_container_width=True)

with col_v2:
    st.markdown(r"""
    ### Interpretation
    
    **V(s)** represents the **long-term discounted expected economic value**
    of holding a position in a company currently rated *s*.
    
    $$V = (I - \gamma P)^{-1} R$$
    
    - Higher V → more attractive from a risk-adjusted return perspective
    - D has the most negative value (large default loss, no recovery)
    - AAA has the highest value (high coupon, minimal loss)
    """)
    st.dataframe(sv_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 4. Migration Summary for Focus State
# ═══════════════════════════════════════════════════════════════
st.markdown(f"## 4️⃣ Rating Migration — Starting from **{focus_state}**")

summary = mrp.migration_summary(focus_state, horizon=horizon)
dist = summary['distribution']
interp = summary['interpretation']

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stay Probability", f"{interp['stay_probability']*100:.1f}%")
m2.metric("Upgrade Probability", f"{interp['upgrade_probability']*100:.1f}%")
m3.metric("Downgrade Probability", f"{interp['downgrade_probability']*100:.1f}%")
m4.metric("Default Probability", f"{interp['default_probability']*100:.2f}%")

outlook = get_rating_outlook(focus_state, tm.loc[focus_state].values)
st.write(f"**Rating Outlook:** {outlook}")

states_list = list(dist.keys())
probs_list = list(dist.values())
fig_mig = go.Figure()
for i, (s, p) in enumerate(zip(states_list, probs_list)):
    fig_mig.add_trace(go.Bar(
        x=[s], y=[p],
        marker_color=RATING_COLORS[i],
        name=s, showlegend=False,
        hovertemplate=f"{s}: {p:.4f}<extra></extra>",
    ))
fig_mig.update_layout(
    title=f"Migration Distribution from {focus_state} ({horizon}-yr)",
    xaxis=dict(type='category', title='Target State'),
    yaxis=dict(title='Probability'),
    template="plotly_white",
    height=380,
)
st.plotly_chart(fig_mig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 5. Default Probability Over Time
# ═══════════════════════════════════════════════════════════════
st.markdown("## 5️⃣ Default Probability Over Time")

dp_df = mrp.default_probability_over_time(max_years=max_years_dp)

fig_dp = go.Figure()
for i, state in enumerate(BUCKETED_RATINGS):
    state_data = dp_df[dp_df['Starting_State'] == state]
    fig_dp.add_trace(go.Scatter(
        x=state_data['Year'].tolist(),
        y=state_data['Default_Probability'].tolist(),
        mode='lines',
        name=state,
        line=dict(color=RATING_COLORS[i], width=2),
        hovertemplate=f"{state} — Year %{{x}}: %{{y:.4f}}<extra></extra>",
    ))
fig_dp.update_layout(
    title="Cumulative Probability of Default by Starting State",
    xaxis_title="Year",
    yaxis_title="P(Default)",
    template="plotly_white",
    height=450,
)
st.plotly_chart(fig_dp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 6. Gamma Sensitivity
# ═══════════════════════════════════════════════════════════════
with st.expander("📐 Gamma Sensitivity Analysis"):
    gs_df = mrp.gamma_sensitivity()
    fig_gs = go.Figure()
    for i, state in enumerate(BUCKETED_RATINGS):
        state_data = gs_df[gs_df['State'] == state]
        fig_gs.add_trace(go.Scatter(
            x=state_data['gamma'].tolist(),
            y=state_data['Value'].tolist(),
            mode='lines+markers',
            name=state,
            line=dict(color=RATING_COLORS[i], width=2),
        ))
    fig_gs.update_layout(
        title="State Values across Discount Factors",
        xaxis_title="γ",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_gs, use_container_width=True)
    st.markdown("""
    **Why γ matters:** A higher γ places more weight on future cash flows.  
    As γ → 1, the value difference between safe (AAA) and risky (C) states widens
    because the long-term cost of default compounds.
    """)
