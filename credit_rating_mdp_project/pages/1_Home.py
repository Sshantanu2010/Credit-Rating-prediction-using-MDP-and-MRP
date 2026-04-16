"""
Page 1 — Home: project overview, Indian credit-rating explainer, and workflow.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(page_title="Home — Credit Rating System", page_icon="🏠", layout="wide")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0066cc,#003d7a);padding:2rem;border-radius:12px;color:white;margin-bottom:1.5rem;">
<h1 style="color:white;margin:0;">🏠 Indian Corporate Credit Rating System</h1>
<p style="color:#b3d4fc;margin:0.3rem 0 0;">MRP & MDP–based Prediction and Capital Allocation Optimization</p>
</div>
""", unsafe_allow_html=True)

# ── Problem Statement ─────────────────────────────────────────
st.markdown("## 📋 Problem Statement")
st.markdown("""
Credit ratings are the backbone of the Indian fixed-income market.  
Rating agencies such as **CRISIL, ICRA, CARE,** and **India Ratings** assign grades
that determine a company's borrowing cost and institutional investability.

This project provides an **end-to-end analytical framework** that:

1. **Predicts** a company's credit rating from its financial ratios using supervised ML.
2. **Models** how that rating is likely to evolve over time via a **Markov Reward Process (MRP)**.
3. **Optimises** lending / investment exposure using a **Markov Decision Process (MDP)**.
""")

# ── Indian Credit Rating Scale ────────────────────────────────
st.markdown("## 🇮🇳 Indian Credit Rating Scale")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Investment Grade")
    st.markdown("""
| Rating | Meaning |
|--------|---------|
| **AAA** | Highest safety; lowest credit risk |
| **AA+, AA, AA−** | High safety; very low credit risk |
| **A+, A, A−** | Adequate safety; low credit risk |
| **BBB+, BBB, BBB−** | Moderate safety; moderate credit risk |
""")

with col2:
    st.markdown("### Speculative / Sub-Investment Grade")
    st.markdown("""
| Rating | Meaning |
|--------|---------|
| **BB+, BB, BB−** | Moderate risk; inadequate safety |
| **B+, B, B−** | High risk; substantial credit risk |
| **C** | Very high risk / near-default |
| **D** | Default — issuer has failed to meet obligations |
""")

st.info(
    "For the MRP / MDP analysis, fine-grained ratings are **bucketed** into 8 states: "
    "AAA, AA, A, BBB, BB, B, C, D.  **D is an absorbing state** (once in default, no recovery in this model)."
)

# ── Workflow Diagram ──────────────────────────────────────────
st.markdown("## 🔄 System Workflow")
st.markdown("""
```
┌──────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│  Company Search  │────▶│  Rating Prediction │────▶│   MRP Transition    │
│  (Name / Manual) │     │  (LR / RF / XGB)   │     │   Analysis          │
└──────────────────┘     └────────────────────┘     └─────────────────────┘
                                                              │
                                                              ▼
                         ┌────────────────────┐     ┌─────────────────────┐
                         │   PDF / CSV Export  │◀───│   MDP Optimizer     │
                         │   & Stress Testing  │    │   (Val / Pol Iter)  │
                         └────────────────────┘     └─────────────────────┘
```
""")

# ── Mathematical Foundation ───────────────────────────────────
st.markdown("## 📐 Mathematical Foundation")

tab1, tab2 = st.tabs(["Markov Reward Process (MRP)", "Markov Decision Process (MDP)"])

with tab1:
    st.markdown(r"""
### Markov Reward Process

A **Markov Reward Process** is defined by the tuple $\langle \mathcal{S}, P, R, \gamma \rangle$:

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | State space: {AAA, AA, A, BBB, BB, B, C, D} |
| $P$ | Transition matrix — $P_{ij} = \Pr(s_{t+1}=j \mid s_t=i)$ |
| $R$ | Reward vector — expected economic return per state |
| $\gamma$ | Discount factor $\in (0,1]$ |

**Bellman equation** (matrix form):

$$V = R + \gamma \, P \, V \quad\Longrightarrow\quad V = (I - \gamma P)^{-1} R$$

**Key property — D is absorbing:**  $P_{D,D} = 1$, meaning once a company defaults
it remains in default (no spontaneous recovery within this model horizon).

**Multi-step transition:** $P^{(k)} = P^k$ gives the probability of moving
between any two states in exactly $k$ years.
""")

with tab2:
    st.markdown(r"""
### Markov Decision Process

An **MDP** extends the MRP by introducing **actions** $\mathcal{A}$ and making
transitions / rewards action-dependent: $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$.

| Component | This project |
|-----------|-------------|
| **State** | `(credit_rating, exposure_level)` |
| **Actions** | Increase, Hold, Reduce, Exit |
| **Reward** | Coupon income − default loss − capital charge |
| **Transition** | Rating evolves stochastically; exposure adapts deterministically |

**Bellman Optimality Equation:**

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \right]$$

**Solved via:**
- **Value Iteration:** iteratively update $V$ until convergence.
- **Policy Iteration:** alternate between policy evaluation and improvement.

**Why MDP for credit?**  It formalises the bank's / fund's decision of how much
exposure to maintain given that the issuer's credit quality is a stochastic process.
""")

# ── Tech Stack ────────────────────────────────────────────────
st.markdown("## 🛠️ Technology Stack")
cols = st.columns(4)
with cols[0]:
    st.markdown("**ML / Data**\n- Pandas\n- NumPy\n- Scikit-learn\n- XGBoost\n- SHAP")
with cols[1]:
    st.markdown("**RL Logic**\n- Custom MRP\n- Custom MDP\n- Value Iteration\n- Policy Iteration")
with cols[2]:
    st.markdown("**Visualisation**\n- Plotly\n- Matplotlib\n- Seaborn")
with cols[3]:
    st.markdown("**App & Reports**\n- Streamlit\n- FPDF2")
