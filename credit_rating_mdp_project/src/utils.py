"""
utils.py — Utility functions and constants for the Indian Credit Rating System.

This module centralises every constant, mapping, and small helper used across
the project so that downstream modules never hard-code rating labels, file
paths, or colour palettes.
"""

import os
import numpy as np

# ════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# ════════════════════════════════════════════════════════════════
# INDIAN CREDIT-RATING CONSTANTS
# ════════════════════════════════════════════════════════════════

# Fine-grained notch-level ratings (CRISIL / ICRA / CARE / India Ratings)
FINE_RATINGS = [
    'AAA', 'AA+', 'AA', 'AA-',
    'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-',
    'BB+', 'BB', 'BB-',
    'B+', 'B', 'B-',
    'C', 'D'
]

# Bucketed ratings for MRP / MDP state-space
BUCKETED_RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'C', 'D']

# Map every fine rating to its bucket
RATING_TO_BUCKET = {
    'AAA': 'AAA',
    'AA+': 'AA', 'AA': 'AA', 'AA-': 'AA',
    'A+': 'A',  'A': 'A',   'A-': 'A',
    'BBB+': 'BBB', 'BBB': 'BBB', 'BBB-': 'BBB',
    'BB+': 'BB',  'BB': 'BB',   'BB-': 'BB',
    'B+': 'B',   'B': 'B',    'B-': 'B',
    'C': 'C',
    'D': 'D'
}

# Ordinal encodings (higher = safer)
RATING_ORDINAL = {r: i for i, r in enumerate(reversed(FINE_RATINGS))}
BUCKET_ORDINAL = {r: i for i, r in enumerate(reversed(BUCKETED_RATINGS))}

# Indian corporate sectors
SECTORS = [
    'Banking & Financial Services', 'IT & Technology', 'Steel & Metals',
    'Automobile', 'Pharmaceuticals', 'Energy & Oil', 'Telecom',
    'Infrastructure & Construction', 'FMCG', 'Cement',
    'Real Estate', 'Power & Utilities', 'Chemicals',
    'Textiles', 'Aviation', 'Media & Entertainment'
]

# ════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ════════════════════════════════════════════════════════════════
NUMERIC_FEATURES = [
    'revenue', 'ebitda', 'ebitda_margin', 'net_profit_margin',
    'debt_to_equity', 'interest_coverage', 'current_ratio', 'quick_ratio',
    'roa', 'roe', 'operating_cash_flow', 'total_debt', 'dscr',
    'promoter_holding', 'company_age', 'market_cap',
    'external_macro_risk_flag', 'qualitative_risk_score'
]
CATEGORICAL_FEATURES = ['sector']
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = 'credit_rating'

# Exposure levels for MDP
EXPOSURE_LEVELS = ['Low', 'Medium', 'High']
MDP_ACTIONS = ['Increase', 'Hold', 'Reduce', 'Exit']

# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_risk_category(rating: str) -> str:
    """Map a credit rating (fine or bucketed) to a risk category."""
    bucket = RATING_TO_BUCKET.get(rating, rating)
    if bucket in ('AAA', 'AA'):
        return 'Low Risk'
    if bucket in ('A', 'BBB'):
        return 'Moderate Risk'
    if bucket in ('BB', 'B', 'C'):
        return 'High Risk'
    if bucket == 'D':
        return 'Default Risk'
    return 'Unknown'


def get_rating_outlook(current_bucket: str,
                       transition_probs: np.ndarray,
                       bucket_list: list = None) -> str:
    """
    Derive a Positive / Stable / Negative outlook from the transition row.
    """
    if bucket_list is None:
        bucket_list = BUCKETED_RATINGS
    if current_bucket == 'D':
        return 'Default'
    idx = bucket_list.index(current_bucket)
    upgrade_prob   = float(np.sum(transition_probs[:idx]))   if idx > 0 else 0.0
    downgrade_prob = float(np.sum(transition_probs[idx+1:])) if idx < len(bucket_list)-1 else 0.0
    if upgrade_prob > downgrade_prob * 1.5:
        return 'Positive'
    if downgrade_prob > upgrade_prob * 1.5:
        return 'Negative'
    return 'Stable'


def get_recommendation(rating_bucket: str, optimal_action: str,
                       outlook: str) -> str:
    """Produce a human-readable lending / investment recommendation."""
    if get_risk_category(rating_bucket) == 'Default Risk':
        return ("⚠️ DEFAULT WATCHLIST — Immediate review required | "
                "Recommendation: Exit all exposure immediately")
    parts = []
    action_map = {
        'Exit':     "🔴 Exit Exposure — High risk of deterioration",
        'Reduce':   "🟠 Reduce Lending Exposure — Elevated risk",
        'Hold':     "🟡 Maintain Current Exposure — Risk within tolerance",
        'Increase': "🟢 Increase Exposure — Favourable risk-reward profile",
    }
    parts.append(action_map.get(optimal_action, optimal_action))
    if outlook == 'Negative':
        parts.append("⚠️ HIGH DOWNGRADE WATCH")
    elif outlook == 'Positive':
        parts.append("✅ Positive outlook — potential upgrade candidate")
    return " | ".join(parts)


def get_color_for_rating(rating: str) -> str:
    """Visual colour per bucket for chart consistency."""
    palette = {
        'AAA': '#006400', 'AA': '#228B22', 'A': '#32CD32',
        'BBB': '#FFD700', 'BB': '#FFA500', 'B': '#FF6347',
        'C':   '#DC143C', 'D':  '#8B0000',
    }
    return palette.get(RATING_TO_BUCKET.get(rating, rating), '#808080')


RATING_COLORS = [get_color_for_rating(r) for r in BUCKETED_RATINGS]


def ensure_dirs():
    """Create project sub-directories if they do not yet exist."""
    for d in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
        os.makedirs(d, exist_ok=True)
