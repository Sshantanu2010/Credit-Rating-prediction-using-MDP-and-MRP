"""
Page 4 — MDP Optimizer: optimal policy, policy heatmap, value table,
convergence plot, exposure recommendation, and gamma sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.utils import BUCKETED_RATINGS, EXPOSURE_LEVELS, MDP_ACTIONS, RATING_COLORS
from src.data_preprocessing import load_transition_matrix
from src.mdp_model import MDP

st.set_page_config(page_title="MDP Optimizer", page_icon="🎯", layout="wide")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#6a1b9a,#4a148c);padding:1.5rem 2rem;border-radius:12px;color:white;margin-bottom:1.5rem;">
<h1 style="color:white;margin:0;">🎯 MDP — Capital Allocation Optimizer</h1>
<p style="color:#ce93d8;">Optimal exposure policies via Value Iteration &amp; Policy Iteration</p>
</div>
""", unsafe_allow_html=True)

# ── Load data & sidebar ──────────────────────────────────────
tm = load_transition_matrix()

with st.sidebar:
    st.markdown("### MDP Settings")
    gamma = st.slider("Discount factor (γ)", 0.50, 0.99, 0.95, 0.01, key="mdp_gamma")
    method = st.radio("Solution method", ["Value Iteration", "Policy Iteration"], key="mdp_method")

# ── Build and solve MDP ──────────────────────────────────────
mdp = MDP(tm.values, gamma=gamma)

if method == "Value Iteration":
    V, policy, history = mdp.value_iteration()
else:
    V, policy, history = mdp.policy_iteration()

policy_table = mdp.get_policy_table(policy)
value_table  = mdp.get_value_table(V)

# ═══════════════════════════════════════════════════════════════
# 1. Optimal Policy Table
# ═══════════════════════════════════════════════════════════════
st.markdown("## 1️⃣ Optimal Policy Table")

col1, col2 = st.columns([3, 2])
with col1:
    def _color_action(val):
        return {
            'Increase': 'background-color: #c8e6c9; color: #1b5e20;',
            'Hold':     'background-color: #fff9c4; color: #f57f17;',
            'Reduce':   'background-color: #ffe0b2; color: #e65100;',
            'Exit':     'background-color: #ffcdd2; color: #b71c1c;',
        }.get(val, '')

    st.dataframe(
        policy_table.style.map(_color_action),
        use_container_width=True, height=350,
    )

with col2:
    st.markdown("""
    ### Reading the Policy
    
    Each cell shows the **optimal action** for a given `(Rating, Exposure)` state.
    
    | Action | Symbol | Meaning |
    |--------|--------|---------|
    | **Increase** | 📈 | Risk-reward favours growth |
    | **Hold** | 📊 | Current allocation is optimal |
    | **Reduce** | 📉 | Deterioration risk outweighs return |
    | **Exit** | 🚫 | Unacceptable risk; wind down |
    
    **Key insight:** For high-rated issuers you should typically *Increase* or *Hold*.
    For low-rated issuers the model advises *Reduce* or *Exit*.
    """)

# ═══════════════════════════════════════════════════════════════
# 2. Policy Heatmap
# ═══════════════════════════════════════════════════════════════
st.markdown("## 2️⃣ Policy Heatmap")

# Encode actions numerically for colour mapping
action_code = {'Increase': 3, 'Hold': 2, 'Reduce': 1, 'Exit': 0}
policy_numeric = policy_table.replace(action_code)

# Reindex to maintain rating order
policy_numeric = policy_numeric.reindex(BUCKETED_RATINGS)
policy_text = policy_table.reindex(BUCKETED_RATINGS)

# Ensure columns are in the correct Exposure order
policy_numeric = policy_numeric[EXPOSURE_LEVELS]
policy_text = policy_text[EXPOSURE_LEVELS]

fig_ph = go.Figure(data=go.Heatmap(
    z=policy_numeric.values.tolist(),
    x=list(EXPOSURE_LEVELS),
    y=list(BUCKETED_RATINGS),
    text=policy_text.values.tolist(),
    texttemplate="%{text}",
    textfont={"size": 13},
    colorscale=[
        [0.0,  '#ef5350'],   # Exit
        [0.33, '#ffa726'],   # Reduce
        [0.66, '#ffee58'],   # Hold
        [1.0,  '#66bb6a'],   # Increase
    ],
    showscale=False,
    hovertemplate="Rating: %{y}<br>Exposure: %{x}<br>Action: %{text}<extra></extra>",
))
fig_ph.update_layout(
    title="Optimal Action per State",
    xaxis_title="Exposure Level",
    yaxis_title="Credit Rating",
    xaxis=dict(type='category'),
    yaxis=dict(type='category', autorange="reversed"),
    template="plotly_white",
    height=480,
)
st.plotly_chart(fig_ph, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 3. State Value Table
# ═══════════════════════════════════════════════════════════════
st.markdown("## 3️⃣ State Value Table  V*(s)")

col_a, col_b = st.columns([3, 2])
with col_a:
    vt = value_table.reindex(BUCKETED_RATINGS)
    vt = vt[EXPOSURE_LEVELS]

    fig_vt = go.Figure(data=go.Heatmap(
        z=vt.values.tolist(),
        x=list(EXPOSURE_LEVELS),
        y=list(BUCKETED_RATINGS),
        text=np.round(vt.values, 1).tolist(),
        texttemplate="%{text}",
        colorscale='RdYlGn',
        hovertemplate="Rating: %{y}<br>Exposure: %{x}<br>Value: %{z:.2f}<extra></extra>",
    ))
    fig_vt.update_layout(
        title="Optimal State Values",
        xaxis_title="Exposure Level",
        yaxis_title="Credit Rating",
        xaxis=dict(type='category'),
        yaxis=dict(type='category', autorange="reversed"),
        template="plotly_white",
        height=460,
    )
    st.plotly_chart(fig_vt, use_container_width=True)

with col_b:
    st.markdown(r"""
    ### Interpretation
    
    $V^*(s)$ is the **maximum long-term discounted reward** attainable
    from state $s$ when following the optimal policy.
    
    - High-rated + Low-exposure states have high value (safe + upside room)
    - D-states have deeply negative values (default loss dominates)
    - Higher exposure amplifies both reward **and** risk
    """)
    st.dataframe(vt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 4. Per-Rating Recommendation Lookup
# ═══════════════════════════════════════════════════════════════
st.markdown("## 4️⃣ Per-Rating Exposure Recommendation")

lookup_rating = st.selectbox("Select a rating to inspect:", BUCKETED_RATINGS, index=3)
actions_for_rating = mdp.get_policy_for_rating(lookup_rating, policy)

cols = st.columns(len(EXPOSURE_LEVELS))
for i, exp in enumerate(EXPOSURE_LEVELS):
    act = actions_for_rating[exp]
    with cols[i]:
        emoji = {'Increase': '📈', 'Hold': '📊', 'Reduce': '📉', 'Exit': '🚫'}
        st.metric(f"Exposure = {exp}", f"{emoji.get(act, '')} {act}")
        st.write(mdp.interpret_action(act))

# ═══════════════════════════════════════════════════════════════
# 5. Gamma Sensitivity
# ═══════════════════════════════════════════════════════════════
with st.expander("📐 Gamma Sensitivity — How Discount Factor Affects Policy"):
    gs_df = mdp.gamma_sensitivity()
    # Show pivot for Medium exposure
    pivot = gs_df[gs_df['Exposure'] == 'Medium'].pivot(
        index='Rating', columns='gamma', values='Action'
    ).reindex(BUCKETED_RATINGS)
    st.dataframe(pivot, use_container_width=True)
    st.markdown("""
    **Observation:** As γ increases (more patient decision-maker), the model
    becomes more conservative for risky ratings — preferring to *Reduce* or
    *Exit* earlier because the long-term cost of staying in a deteriorating
    credit compounds.
    """)

# ═══════════════════════════════════════════════════════════════
# 6. MDP Theory Recap
# ═══════════════════════════════════════════════════════════════
with st.expander("📚 MDP Theory — Why This Matters for Lending Decisions"):
    st.markdown(r"""
### Why model capital allocation as an MDP?

Banks and asset managers must continuously decide **how much exposure** to
maintain to a given borrower whose credit quality evolves **stochastically**.

| Traditional Approach | MDP Approach |
|----------------------|--------------|
| Static risk limits per rating | Dynamic, forward-looking policy |
| Ignores transition dynamics | Explicitly models rating drift |
| One-shot decision | Infinite-horizon optimisation |
| Rules-of-thumb | Mathematically optimal under model |

### Bellman Optimality Equation

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V^*(s') \right]$$

- **Value Iteration** computes $V^*$ by iterating on the Bellman update.
- **Policy Iteration** alternates between evaluating a policy and improving it.

### Why D is absorbing

Once a company defaults, the lender realises the loss; there is no
"un-defaulting" within the model horizon.  This makes D a **terminal state**
that the optimizer strongly avoids.
    """)
