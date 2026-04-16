# 🏦 Indian Corporate Credit Rating Prediction & Capital Allocation Optimization

> **MRP & MDP–based framework for predicting Indian credit ratings and optimising lending exposure**

---

## 📋 Overview

This is a production-style, modular **Python + Streamlit** project that:

1. **Predicts** an Indian company's credit rating using supervised ML (Logistic Regression, Random Forest, XGBoost).
2. **Models** how that rating evolves over time with a **Markov Reward Process (MRP)**.
3. **Optimises** exposure / capital allocation with a **Markov Decision Process (MDP)** solved via Value Iteration and Policy Iteration.
4. **Stress-tests** the model under four macro-economic scenarios.
5. **Exports** professional PDF reports and CSV data.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

On first launch the app will:
- Generate a synthetic dataset of ~1 000 Indian companies (if not already present)
- Generate a realistic transition matrix
- Train three ML classifiers and select the best one

Subsequent launches re-use the cached data and models.

---

## 🗂️ Project Structure

```
credit_rating_mdp_project/
│
├── app.py                          # Main Streamlit entry-point
├── requirements.txt                # Python dependencies
├── README.md
│
├── data/
│   ├── indian_companies_credit_data.csv   # Generated synthetic dataset
│   └── transition_matrix.csv              # 8×8 rating transition matrix
│
├── models/
│   ├── rating_model.pkl            # Best trained classifier
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoder.pkl           # Sector encoder
│   ├── label_to_int.pkl            # Rating → int mapping
│   ├── int_to_label.pkl            # int → Rating mapping
│   ├── feature_cols.pkl            # Feature column order
│   ├── all_models.pkl              # All trained models
│   └── metrics.json                # Evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Constants, mappings, helper functions
│   ├── data_preprocessing.py       # Dataset generation & data loading
│   ├── feature_engineering.py      # Encoding, scaling, feature prep
│   ├── train_rating_model.py       # ML training & model selection
│   ├── predict_rating.py           # Inference, SHAP, explanations
│   ├── mrp_model.py                # Markov Reward Process (from scratch)
│   ├── mdp_model.py                # Markov Decision Process (from scratch)
│   ├── stress_testing.py           # Macro-economic stress scenarios
│   └── report_generator.py         # PDF and CSV export
│
├── pages/
│   ├── 1_Home.py                   # Project overview & theory
│   ├── 2_Credit_Rating_Predictor.py # Company search, prediction, comparison
│   ├── 3_MRP_Transition_Model.py   # Transition heatmaps, state values
│   ├── 4_MDP_Optimizer.py          # Optimal policy, convergence
│   ├── 5_Stress_Testing.py         # Scenario comparison
│   └── 6_Reports.py               # PDF generation, CSV downloads
│
└── assets/                         # Generated report files
```

---

## 🇮🇳 Indian Credit Rating Scale

| Grade | Ratings | Risk Level |
|-------|---------|------------|
| **Investment Grade** | AAA, AA+, AA, AA−, A+, A, A−, BBB+, BBB, BBB− | Low – Moderate |
| **Speculative Grade** | BB+, BB, BB−, B+, B, B−, C | High |
| **Default** | D | Default |

For MRP/MDP, ratings are bucketed into **8 states**: AAA, AA, A, BBB, BB, B, C, D.  
**D is an absorbing state** — once in default, no recovery within this model.

---

## 📊 Module Details

### Module A — Credit Rating Prediction

| Feature | Detail |
|---------|--------|
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Selection** | Best macro-F1 via 5-fold cross-validation |
| **Inputs** | 19 financial features + sector |
| **Outputs** | Predicted rating, probability distribution, risk category, feature importance |
| **Explainability** | SHAP (if available) or built-in feature importance |

### Module B — Markov Reward Process

| Component | Detail |
|-----------|--------|
| **State space** | {AAA, AA, A, BBB, BB, B, C, D} |
| **Transition matrix** | 8×8, rows sum to 1, D absorbing |
| **Reward** | Economic coupon/yield net of default loss |
| **Bellman equation** | V = (I − γP)⁻¹ R |
| **Outputs** | State values, multi-step transitions, default probability curves |

### Module C — Markov Decision Process

| Component | Detail |
|-----------|--------|
| **State** | (credit_rating, exposure_level) — 24 states |
| **Actions** | Increase, Hold, Reduce, Exit |
| **Reward** | Coupon income − default loss × LGD − capital charge |
| **Solution** | Value Iteration & Policy Iteration |
| **Outputs** | Optimal policy table, policy heatmap, convergence plot |

---

## ⚡ Stress Testing Scenarios

| Scenario | Description |
|----------|-------------|
| **Normal Economy** | Baseline (unchanged transition matrix) |
| **Recession** | GDP contraction; higher downgrades & defaults |
| **High Inflation** | CPI > 7%; margin pressure; RBI rate hikes |
| **Credit Tightening** | NBFC stress; severe liquidity crunch |

---

## 🧪 Demo Walkthrough

1. Launch the app: `streamlit run app.py`
2. Go to **Credit Rating Predictor**
3. Search for **"Tata Motors"** → auto-loads financials → predicts rating
4. View MRP transition analysis and MDP recommendation inline
5. Navigate to **MRP Transition Model** for heatmaps and default curves
6. Navigate to **MDP Optimizer** for the full optimal policy
7. Run **Stress Testing** to compare scenarios
8. Go to **Reports** to generate a PDF and download CSVs

---

## 📐 Mathematical Foundation

### MRP — Bellman Equation

```
V = R + γ P V   →   V = (I − γP)⁻¹ R
```

### MDP — Bellman Optimality

```
V*(s) = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V*(s') ]
```

### Value Iteration

```
V_{k+1}(s) = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V_k(s') ]
repeat until |V_{k+1} − V_k| < ε
```

### Policy Iteration

```
1. Evaluate: V^π = (I − γ P^π)⁻¹ R^π
2. Improve: π'(s) = argmax_a [ R(s,a) + γ Σ_s' P(s'|s,a) V^π(s') ]
3. Repeat until π' = π
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, SHAP |
| RL | Custom MRP & MDP (no external RL libraries) |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Reports | FPDF2 |

---

## 📝 Dataset

The project uses a **realistic synthetic dataset** of ~1 000 Indian companies because
actual CRISIL / ICRA data is proprietary. The synthetic data:

- Covers 16 Indian sectors (Banking, IT, Steel, Auto, Pharma, etc.)
- Uses ~170 real Indian company names + synthetic names
- Financial ratios are calibrated per rating bucket
- Rating distribution mirrors real-world patterns (more A/BBB/AA than D)

---

## 📄 License

This project is for **educational / portfolio** purposes.  
Credit rating data is synthetic. No real financial advice is implied.
