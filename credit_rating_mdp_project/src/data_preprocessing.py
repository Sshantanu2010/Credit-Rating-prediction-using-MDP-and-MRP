"""
data_preprocessing.py — Synthetic Indian corporate credit-dataset generator
and loading / preprocessing utilities.

When real company-level credit data is unavailable (most CRISIL / ICRA data is
behind pay-walls), this module produces a realistic synthetic dataset of
~1 000 Indian companies with financial ratios calibrated to their credit
rating.  It also generates a realistic annual transition matrix.
"""

import os
import numpy as np
import pandas as pd
from src.utils import (
    FINE_RATINGS, BUCKETED_RATINGS, RATING_TO_BUCKET,
    SECTORS, NUMERIC_FEATURES, TARGET_COLUMN,
    DATA_DIR, ensure_dirs
)

# ═══════════════════════════════════════════════════════════════
# WELL-KNOWN INDIAN COMPANY NAMES (seed list)
# ═══════════════════════════════════════════════════════════════
_SEED_COMPANIES = {
    'Banking & Financial Services': [
        'HDFC Bank', 'ICICI Bank', 'State Bank of India', 'Kotak Mahindra Bank',
        'Axis Bank', 'IndusInd Bank', 'Punjab National Bank', 'Bank of Baroda',
        'Canara Bank', 'IDFC First Bank', 'Federal Bank', 'Bandhan Bank',
        'Yes Bank', 'RBL Bank', 'South Indian Bank', 'Bajaj Finance',
        'Bajaj Finserv', 'Muthoot Finance', 'Shriram Finance', 'LIC Housing Finance',
        'HDFC Life Insurance', 'SBI Life Insurance', 'ICICI Lombard',
        'Manappuram Finance', 'Cholamandalam Investment',
    ],
    'IT & Technology': [
        'Tata Consultancy Services', 'Infosys', 'Wipro', 'HCL Technologies',
        'Tech Mahindra', 'LTIMindtree', 'Mphasis', 'Persistent Systems',
        'Coforge', 'Cyient', 'L&T Technology Services', 'Birlasoft',
        'Zensar Technologies', 'NIIT Technologies', 'Tata Elxsi',
    ],
    'Steel & Metals': [
        'Tata Steel', 'JSW Steel', 'Hindalco Industries', 'Vedanta',
        'Steel Authority of India', 'Jindal Steel & Power', 'National Aluminium',
        'Hindustan Zinc', 'Ratnamani Metals', 'APL Apollo Tubes',
    ],
    'Automobile': [
        'Tata Motors', 'Maruti Suzuki', 'Mahindra & Mahindra', 'Bajaj Auto',
        'Hero MotoCorp', 'Eicher Motors', 'Ashok Leyland', 'TVS Motor',
        'MRF', 'Apollo Tyres', 'Escorts Kubota', 'Ola Electric',
    ],
    'Pharmaceuticals': [
        'Sun Pharmaceutical', 'Dr Reddys Laboratories', 'Cipla', 'Lupin',
        'Aurobindo Pharma', 'Divi\'s Laboratories', 'Biocon', 'Torrent Pharma',
        'Zydus Lifesciences', 'Alkem Laboratories', 'Glenmark Pharmaceuticals',
        'Mankind Pharma', 'Natco Pharma', 'Laurus Labs',
    ],
    'Energy & Oil': [
        'Reliance Industries', 'Oil & Natural Gas Corporation', 'Indian Oil Corporation',
        'Bharat Petroleum', 'Hindustan Petroleum', 'GAIL India',
        'Petronet LNG', 'Adani Total Gas', 'Gujarat Gas', 'Indraprastha Gas',
    ],
    'Telecom': [
        'Bharti Airtel', 'Vodafone Idea', 'Jio Platforms',
        'Indus Towers', 'Tata Communications', 'Sterlite Technologies',
        'HFCL', 'Tejas Networks',
    ],
    'Infrastructure & Construction': [
        'Larsen & Toubro', 'Adani Ports', 'Adani Enterprises',
        'IRB Infrastructure', 'NCC Limited', 'KNR Constructions',
        'PNC Infratech', 'Dilip Buildcon', 'HG Infra Engineering',
        'Kalpataru Projects', 'GMR Airports Infrastructure',
    ],
    'FMCG': [
        'Hindustan Unilever', 'ITC', 'Nestle India', 'Britannia Industries',
        'Dabur India', 'Marico', 'Godrej Consumer Products', 'Colgate-Palmolive India',
        'Emami', 'Tata Consumer Products', 'Procter & Gamble India',
    ],
    'Cement': [
        'UltraTech Cement', 'Ambuja Cements', 'ACC', 'Shree Cement',
        'Dalmia Bharat', 'Ramco Cements', 'JK Cement', 'Birla Corporation',
        'India Cements', 'Nuvoco Vistas',
    ],
    'Real Estate': [
        'DLF', 'Godrej Properties', 'Prestige Estates', 'Oberoi Realty',
        'Brigade Enterprises', 'Macrotech Developers', 'Sobha Limited',
        'Phoenix Mills', 'Sunteck Realty', 'Mahindra Lifespace',
    ],
    'Power & Utilities': [
        'NTPC', 'Power Grid Corporation', 'Adani Power', 'Adani Green Energy',
        'Tata Power', 'JSW Energy', 'Torrent Power', 'NHPC',
        'SJVN', 'CESC',
    ],
    'Chemicals': [
        'Pidilite Industries', 'SRF Limited', 'Aarti Industries', 'Deepak Nitrite',
        'Clean Science and Technology', 'Navin Fluorine', 'Gujarat Fluorochemicals',
        'Atul Limited', 'Balaji Amines', 'NOCIL',
    ],
    'Textiles': [
        'Page Industries', 'Arvind Limited', 'Raymond', 'Welspun Living',
        'Trident Limited', 'Vardhman Textiles', 'KPR Mill',
    ],
    'Aviation': [
        'InterGlobe Aviation (IndiGo)', 'SpiceJet', 'Akasa Air',
        'Air India (Tata)', 'Jet Airways',
    ],
    'Media & Entertainment': [
        'Zee Entertainment', 'PVR INOX', 'Sun TV Network',
        'TV18 Broadcast', 'Saregama India', 'Tips Industries',
    ],
}

# ═══════════════════════════════════════════════════════════════
# FINANCIAL-RATIO RANGES PER BUCKETED RATING
# Each key maps to (mean, std) for a truncated-normal draw.
# ═══════════════════════════════════════════════════════════════
_RATIO_PROFILES = {
    # -------- AAA --------
    'AAA': dict(
        revenue=(80000, 40000), ebitda=(24000, 12000),
        ebitda_margin=(30, 7), net_profit_margin=(20, 5),
        debt_to_equity=(0.25, 0.12), interest_coverage=(15, 5),
        current_ratio=(2.8, 0.6), quick_ratio=(2.0, 0.5),
        roa=(14, 3), roe=(22, 5),
        operating_cash_flow=(18000, 9000), total_debt=(15000, 8000),
        dscr=(5.0, 1.5), promoter_holding=(62, 8),
        company_age=(55, 20), market_cap=(200000, 100000),
    ),
    # -------- AA --------
    'AA': dict(
        revenue=(50000, 25000), ebitda=(14000, 7000),
        ebitda_margin=(25, 6), net_profit_margin=(16, 4),
        debt_to_equity=(0.5, 0.2), interest_coverage=(10, 3),
        current_ratio=(2.2, 0.5), quick_ratio=(1.6, 0.4),
        roa=(11, 3), roe=(18, 4),
        operating_cash_flow=(11000, 6000), total_debt=(22000, 12000),
        dscr=(3.8, 1.0), promoter_holding=(58, 10),
        company_age=(42, 15), market_cap=(100000, 60000),
    ),
    # -------- A --------
    'A': dict(
        revenue=(30000, 18000), ebitda=(7500, 4500),
        ebitda_margin=(20, 5), net_profit_margin=(12, 4),
        debt_to_equity=(0.8, 0.3), interest_coverage=(6, 2),
        current_ratio=(1.8, 0.4), quick_ratio=(1.3, 0.35),
        roa=(8, 2.5), roe=(15, 4),
        operating_cash_flow=(6000, 4000), total_debt=(28000, 15000),
        dscr=(2.8, 0.8), promoter_holding=(52, 12),
        company_age=(32, 12), market_cap=(50000, 30000),
    ),
    # -------- BBB --------
    'BBB': dict(
        revenue=(15000, 10000), ebitda=(3000, 2000),
        ebitda_margin=(15, 5), net_profit_margin=(8, 3),
        debt_to_equity=(1.2, 0.4), interest_coverage=(3.5, 1.2),
        current_ratio=(1.4, 0.3), quick_ratio=(1.0, 0.25),
        roa=(5, 2), roe=(11, 3),
        operating_cash_flow=(2500, 1800), total_debt=(20000, 12000),
        dscr=(1.8, 0.5), promoter_holding=(48, 12),
        company_age=(24, 10), market_cap=(20000, 15000),
    ),
    # -------- BB --------
    'BB': dict(
        revenue=(8000, 5000), ebitda=(1200, 800),
        ebitda_margin=(10, 4), net_profit_margin=(4, 3),
        debt_to_equity=(1.8, 0.6), interest_coverage=(2.0, 0.8),
        current_ratio=(1.15, 0.25), quick_ratio=(0.8, 0.2),
        roa=(3, 1.5), roe=(7, 3),
        operating_cash_flow=(800, 600), total_debt=(16000, 10000),
        dscr=(1.2, 0.3), promoter_holding=(42, 14),
        company_age=(18, 8), market_cap=(8000, 6000),
    ),
    # -------- B --------
    'B': dict(
        revenue=(4000, 3000), ebitda=(400, 350),
        ebitda_margin=(5, 4), net_profit_margin=(1, 3),
        debt_to_equity=(2.5, 0.8), interest_coverage=(1.0, 0.5),
        current_ratio=(0.9, 0.2), quick_ratio=(0.6, 0.15),
        roa=(1, 1.5), roe=(3, 4),
        operating_cash_flow=(200, 300), total_debt=(12000, 8000),
        dscr=(0.8, 0.25), promoter_holding=(35, 15),
        company_age=(12, 6), market_cap=(3000, 2500),
    ),
    # -------- C --------
    'C': dict(
        revenue=(2000, 1500), ebitda=(-100, 300),
        ebitda_margin=(-5, 6), net_profit_margin=(-10, 5),
        debt_to_equity=(4.0, 1.5), interest_coverage=(0.3, 0.3),
        current_ratio=(0.65, 0.15), quick_ratio=(0.35, 0.12),
        roa=(-5, 3), roe=(-8, 5),
        operating_cash_flow=(-200, 400), total_debt=(10000, 7000),
        dscr=(0.4, 0.2), promoter_holding=(28, 12),
        company_age=(8, 5), market_cap=(1000, 800),
    ),
    # -------- D --------
    'D': dict(
        revenue=(800, 600), ebitda=(-500, 400),
        ebitda_margin=(-18, 8), net_profit_margin=(-25, 8),
        debt_to_equity=(7.0, 3.0), interest_coverage=(-1.0, 0.5),
        current_ratio=(0.4, 0.15), quick_ratio=(0.2, 0.1),
        roa=(-12, 4), roe=(-30, 10),
        operating_cash_flow=(-600, 500), total_debt=(8000, 5000),
        dscr=(0.1, 0.15), promoter_holding=(18, 10),
        company_age=(6, 4), market_cap=(400, 350),
    ),
}

# Rating distribution target (sums to ~1000)
_FINE_RATING_WEIGHTS = {
    'AAA': 40, 'AA+': 50, 'AA': 65, 'AA-': 55,
    'A+': 75, 'A': 90, 'A-': 70,
    'BBB+': 80, 'BBB': 75, 'BBB-': 60,
    'BB+': 50, 'BB': 45, 'BB-': 35,
    'B+': 35, 'B': 30, 'B-': 25,
    'C': 25, 'D': 25,
}


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def _all_seed_names() -> list:
    """Flatten seed companies into a list of (name, sector) tuples."""
    out = []
    for sector, names in _SEED_COMPANIES.items():
        for n in names:
            out.append((n, sector))
    return out


def _make_extra_names(n: int, rng: np.random.Generator) -> list:
    """Generate additional synthetic Indian company names."""
    prefixes = [
        'Bharat', 'Desh', 'Shakti', 'Sagar', 'Param', 'Vijay',
        'Aarav', 'Surya', 'Indus', 'Ganga', 'Kaveri', 'Narmada',
        'Himalaya', 'Sahyadri', 'Vindhya', 'Deccan', 'Konkan',
        'Ganesh', 'Laxmi', 'Shri', 'Jai', 'Nav', 'Prem',
        'Raj', 'Arun', 'Kiran', 'Priya', 'Anand', 'Vivek',
        'Arjun', 'Krishna', 'Radha', 'Meera', 'Ashoka',
        'Chanakya', 'Vikram', 'Saraswati', 'Durga', 'Parvati',
    ]
    suffixes = [
        'Industries', 'Enterprises', 'Corporation', 'Technologies',
        'Infra', 'Solutions', 'Limited', 'Holdings', 'Capital',
        'Engineering', 'Chemicals', 'Textiles', 'Power',
        'Steel', 'Pharma', 'Logistics', 'Agri', 'Finance',
        'Polymers', 'Ceramics', 'Foods', 'Exports', 'Ventures',
    ]
    names = set()
    while len(names) < n:
        name = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
        names.add(name)
    return list(names)


def generate_synthetic_dataset(n_companies: int = 1000,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Indian corporate credit dataset.

    Args:
        n_companies: target number of companies
        seed: random seed for reproducibility

    Returns:
        DataFrame with company_name, sector, all financial features,
        and credit_rating.
    """
    rng = np.random.default_rng(seed)

    # --- Assign fine-grained ratings proportionally ---
    weights = np.array([_FINE_RATING_WEIGHTS[r] for r in FINE_RATINGS], dtype=float)
    weights /= weights.sum()
    ratings = rng.choice(FINE_RATINGS, size=n_companies, p=weights)

    # --- Assign company names ---
    seed_pool = _all_seed_names()
    rng.shuffle(seed_pool)
    extra_names = _make_extra_names(max(0, n_companies - len(seed_pool)), rng)
    all_names_sectors: list[tuple] = []

    # First, use seed names
    for i, rating in enumerate(ratings):
        if i < len(seed_pool):
            all_names_sectors.append(seed_pool[i])
        else:
            extra_idx = i - len(seed_pool)
            name = extra_names[extra_idx] if extra_idx < len(extra_names) else f"Company_{i}"
            sector = rng.choice(SECTORS)
            all_names_sectors.append((name, sector))

    # --- Generate financial features per company ---
    rows = []
    for i, rating in enumerate(ratings):
        name, sector = all_names_sectors[i]
        bucket = RATING_TO_BUCKET[rating]
        profile = _RATIO_PROFILES[bucket]

        row = {'company_name': name, 'sector': sector}
        for feat, (mu, sigma) in profile.items():
            val = rng.normal(mu, sigma)
            # Apply sensible floors
            if feat in ('revenue', 'market_cap', 'total_debt', 'company_age'):
                val = max(val, 10)
            elif feat in ('current_ratio', 'quick_ratio'):
                val = max(val, 0.05)
            elif feat == 'promoter_holding':
                val = np.clip(val, 0, 90)
            elif feat == 'company_age':
                val = max(int(val), 1)
            elif feat == 'debt_to_equity':
                val = max(val, 0.01)
            row[feat] = round(float(val), 2)

        # Derived / extra flags
        row['external_macro_risk_flag'] = int(rng.random() < (0.1 if bucket in ('AAA','AA','A') else 0.35))
        row['qualitative_risk_score'] = round(float(np.clip(
            rng.normal(
                {'AAA':1.5,'AA':2.5,'A':3.5,'BBB':5,'BB':6.5,'B':7.5,'C':8.5,'D':9.5}[bucket],
                0.8
            ), 1, 10)), 1)
        row['credit_rating'] = rating
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════
# TRANSITION MATRIX GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_transition_matrix() -> pd.DataFrame:
    """
    Create a realistic 8×8 annual credit-rating transition matrix
    for Indian corporates (bucketed ratings).

    Each row sums to 1.  D is absorbing.
    Inspired by published CRISIL / S&P one-year transition studies.
    """
    labels = BUCKETED_RATINGS  # AAA … D
    P = np.array([
        # AAA     AA      A       BBB     BB      B       C       D
        [0.9100, 0.0700, 0.0100, 0.0050, 0.0030, 0.0010, 0.0005, 0.0005],  # AAA
        [0.0300, 0.8900, 0.0500, 0.0150, 0.0080, 0.0040, 0.0020, 0.0010],  # AA
        [0.0050, 0.0300, 0.8700, 0.0600, 0.0200, 0.0080, 0.0040, 0.0030],  # A
        [0.0010, 0.0050, 0.0400, 0.8400, 0.0700, 0.0250, 0.0120, 0.0070],  # BBB
        [0.0005, 0.0010, 0.0050, 0.0500, 0.8000, 0.0800, 0.0400, 0.0235],  # BB
        [0.0000, 0.0005, 0.0010, 0.0050, 0.0500, 0.7500, 0.1200, 0.0735],  # B
        [0.0000, 0.0000, 0.0005, 0.0010, 0.0050, 0.0300, 0.7000, 0.2635],  # C
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # D
    ])
    return pd.DataFrame(P, index=labels, columns=labels)


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE (LOAD / SAVE)
# ═══════════════════════════════════════════════════════════════

def save_dataset(df: pd.DataFrame, filename: str = "indian_companies_credit_data.csv"):
    ensure_dirs()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


def load_dataset(filename: str = "indian_companies_credit_data.csv") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        df = generate_synthetic_dataset()
        save_dataset(df, filename)
    return pd.read_csv(path)


def save_transition_matrix(df: pd.DataFrame,
                           filename: str = "transition_matrix.csv"):
    ensure_dirs()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    return path


def load_transition_matrix(filename: str = "transition_matrix.csv") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        df = generate_transition_matrix()
        save_transition_matrix(df, filename)
    return pd.read_csv(path, index_col=0)


def lookup_company(company_name: str, df: pd.DataFrame = None) -> pd.Series | None:
    """
    Case-insensitive fuzzy lookup by company name.
    Returns the first matching row or None.
    """
    if df is None:
        df = load_dataset()
    mask = df['company_name'].str.lower().str.contains(company_name.lower(), na=False)
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


# ═══════════════════════════════════════════════════════════════
# DATA CLEANING & BASIC PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, handle missing values, ensure correct dtypes."""
    df = df.copy()
    df.drop_duplicates(subset='company_name', keep='first', inplace=True)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    if 'sector' in df.columns:
        df['sector'].fillna('Unknown', inplace=True)
    if TARGET_COLUMN in df.columns:
        df = df[df[TARGET_COLUMN].isin(FINE_RATINGS)]
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# TOP-LEVEL: initialise data on first import
# ═══════════════════════════════════════════════════════════════

def initialise_data():
    """Generate and persist all data artefacts if they don't exist yet."""
    ensure_dirs()
    data_path = os.path.join(DATA_DIR, "indian_companies_credit_data.csv")
    tm_path   = os.path.join(DATA_DIR, "transition_matrix.csv")
    if not os.path.exists(data_path):
        df = generate_synthetic_dataset()
        save_dataset(df)
    if not os.path.exists(tm_path):
        tm = generate_transition_matrix()
        save_transition_matrix(tm)
