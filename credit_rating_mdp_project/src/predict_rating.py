"""
predict_rating.py — Inference module for credit-rating prediction.

Loads a trained model + preprocessing artefacts and produces:
  • predicted rating
  • class-probability distribution
  • risk category
  • feature-importance / SHAP explanation
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib

from src.utils import (
    FINE_RATINGS, RATING_TO_BUCKET, MODELS_DIR,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    get_risk_category, get_color_for_rating,
)
from src.feature_engineering import load_preprocessing_artifacts

warnings.filterwarnings("ignore")

# Try to import SHAP — optional but preferred
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def _load_model_and_artifacts():
    """Load the persisted model, scaler, encoders."""
    model = joblib.load(os.path.join(MODELS_DIR, "rating_model.pkl"))
    le, scaler, l2i, i2l, feat_cols = load_preprocessing_artifacts()
    return model, le, scaler, l2i, i2l, feat_cols


def predict_single(company_data: dict) -> dict:
    """
    Predict the credit rating for a single company.

    Args:
        company_data: dict with keys matching ALL_FEATURES
                      (company_name is ignored for prediction).

    Returns:
        dict with keys:
          predicted_rating, predicted_bucket, risk_category,
          probabilities (dict rating→prob), confidence,
          feature_importance (dict feature→importance)
    """
    model, le, scaler, l2i, i2l, feat_cols = _load_model_and_artifacts()

    # Build a single-row DataFrame
    row = {}
    for col in feat_cols:
        if col == 'sector':
            sector_val = company_data.get('sector', 'Unknown')
            known = set(le.classes_)
            if sector_val not in known:
                sector_val = le.classes_[0]
            row['sector'] = le.transform([sector_val])[0]
        else:
            row[col] = float(company_data.get(col, 0))

    X_df = pd.DataFrame([row])[feat_cols]
    X_scaled = scaler.transform(X_df.values)

    # Predict class and probabilities
    pred_idx = model.predict(X_scaled)[0]
    pred_rating = i2l[pred_idx]
    pred_bucket = RATING_TO_BUCKET.get(pred_rating, pred_rating)

    proba = model.predict_proba(X_scaled)[0]
    # Map probabilities to rating labels
    classes_in_model = model.classes_
    prob_dict = {}
    for cls_idx, p in zip(classes_in_model, proba):
        prob_dict[i2l[cls_idx]] = round(float(p), 4)

    confidence = float(max(proba))

    # Feature importance
    fi = {}
    if hasattr(model, 'feature_importances_'):
        fi = dict(zip(feat_cols, model.feature_importances_.tolist()))
    elif hasattr(model, 'coef_'):
        avg_coef = np.mean(np.abs(model.coef_), axis=0)
        fi = dict(zip(feat_cols, avg_coef.tolist()))

    return {
        'predicted_rating': pred_rating,
        'predicted_bucket': pred_bucket,
        'risk_category': get_risk_category(pred_rating),
        'probabilities': prob_dict,
        'confidence': round(confidence, 4),
        'feature_importance': fi,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict ratings for multiple companies.
    Adds columns: predicted_rating, predicted_bucket, risk_category, confidence.
    """
    model, le, scaler, l2i, i2l, feat_cols = _load_model_and_artifacts()

    df_work = df.copy()

    # Encode sector
    if 'sector' in df_work.columns:
        known = set(le.classes_)
        df_work['sector'] = df_work['sector'].apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df_work['sector'] = le.transform(df_work['sector'].astype(str))
    else:
        df_work['sector'] = 0

    for col in feat_cols:
        if col not in df_work.columns:
            df_work[col] = 0
    X = df_work[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_scaled = scaler.transform(X.values)

    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    df['predicted_rating'] = [i2l[p] for p in preds]
    df['predicted_bucket'] = df['predicted_rating'].map(RATING_TO_BUCKET)
    df['risk_category'] = df['predicted_rating'].apply(get_risk_category)
    df['confidence'] = probas.max(axis=1).round(4)

    return df


def get_shap_explanation(company_data: dict, max_display: int = 15):
    """
    Compute SHAP values for a single prediction.
    Falls back to built-in feature importance if SHAP is unavailable.

    Returns:
        (shap_values or None, feature_importance_dict, feature_names)
    """
    model, le, scaler, l2i, i2l, feat_cols = _load_model_and_artifacts()

    row = {}
    for col in feat_cols:
        if col == 'sector':
            sector_val = company_data.get('sector', 'Unknown')
            known = set(le.classes_)
            if sector_val not in known:
                sector_val = le.classes_[0]
            row['sector'] = le.transform([sector_val])[0]
        else:
            row[col] = float(company_data.get(col, 0))

    X_df = pd.DataFrame([row])[feat_cols]
    X_scaled = scaler.transform(X_df.values)

    shap_values_out = None
    fi = {}

    if _HAS_SHAP:
        try:
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_scaled)
                shap_values_out = sv
            else:
                # Use KernelExplainer for linear models
                bg = shap.kmeans(X_scaled, 1)
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(X_scaled)
                shap_values_out = sv

            # Aggregate SHAP importance across classes
            if isinstance(sv, list):
                mean_abs = np.mean([np.abs(s) for s in sv], axis=0).flatten()
            else:
                mean_abs = np.mean(np.abs(sv), axis=0).flatten()
            fi = dict(zip(feat_cols, mean_abs.tolist()))
        except Exception:
            pass

    if not fi:
        if hasattr(model, 'feature_importances_'):
            fi = dict(zip(feat_cols, model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            avg = np.mean(np.abs(model.coef_), axis=0)
            fi = dict(zip(feat_cols, avg.tolist()))

    return shap_values_out, fi, feat_cols


def explain_rating(result: dict) -> list[str]:
    """
    Generate human-readable explanations for why a company got its rating.
    """
    explanations = []
    rating = result['predicted_rating']
    risk = result['risk_category']
    conf = result['confidence']

    explanations.append(
        f"The model predicts a **{rating}** rating with **{conf*100:.1f}%** confidence."
    )
    explanations.append(f"This falls in the **{risk}** category.")

    # Top drivers
    fi = result.get('feature_importance', {})
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_fi[:3]
        drivers = ", ".join([f"**{f}**" for f, _ in top3])
        explanations.append(f"Key drivers: {drivers}.")

    # Risk-specific commentary
    if risk == 'Low Risk':
        explanations.append(
            "The company demonstrates strong financial health with robust "
            "coverage ratios and low leverage — consistent with investment-grade "
            "quality in the Indian market."
        )
    elif risk == 'Moderate Risk':
        explanations.append(
            "The company has adequate financial metrics but may face sector-specific "
            "or macro headwinds. Monitor debt serviceability closely."
        )
    elif risk == 'High Risk':
        explanations.append(
            "Elevated leverage, weak coverage ratios, or deteriorating margins "
            "place this company in the speculative-grade territory. Consider "
            "reducing exposure."
        )
    else:
        explanations.append(
            "The company is in or near default. Immediate risk-mitigation "
            "measures are warranted."
        )

    return explanations
