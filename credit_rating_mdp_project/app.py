"""
app.py — Main entry-point for the Streamlit multi-page application.

Run with:
    streamlit run app.py

This file bootstraps the data / model artefacts on first launch and
sets up the global page configuration.  Individual pages live in pages/.
"""

import sys, os

# Ensure project root is on the Python path so `from src.xxx` works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.data_preprocessing import initialise_data
from src.train_rating_model import is_model_trained, train_and_select

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Credit Rating — MRP & MDP",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a2742 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e6f0 !important;
    }
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #003d7a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #b3d4fc; margin: 0.3rem 0 0 0; font-size: 1rem; }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# ── Bootstrap ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Generating synthetic dataset …")
def _bootstrap_data():
    initialise_data()
    return True

@st.cache_resource(show_spinner="Training credit-rating models (one-time) …")
def _bootstrap_model():
    if not is_model_trained():
        train_and_select()
    return True

_bootstrap_data()
_bootstrap_model()

# ── Sidebar branding ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Rating System")
    st.caption("Indian Corporate Credit Rating Prediction\n& Capital Allocation via MRP / MDP")
    st.divider()
    st.markdown("**Navigation** — use the pages above ☝️")
    st.divider()
    st.caption("© 2026 · Built with Streamlit")

# ── Landing content (shown on the root page) ─────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Indian Corporate Credit Rating Prediction</h1>
    <p>Capital Allocation Optimization using Markov Reward & Decision Processes</p>
</div>
""", unsafe_allow_html=True)

st.info("👈 **Select a page from the sidebar** to get started.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📊 Predict Ratings")
    st.write("Enter a company name or upload financials to predict its Indian credit rating with ML models.")
with col2:
    st.markdown("### 🔄 MRP Analysis")
    st.write("Model rating transitions, compute state values, and assess default probabilities over time.")
with col3:
    st.markdown("### 🎯 MDP Optimizer")
    st.write("Find the optimal exposure / capital allocation policy using Value & Policy Iteration.")
