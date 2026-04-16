"""
feature_engineering.py — Feature encoding, scaling, and selection utilities.

Transforms the raw company DataFrame into a numeric feature matrix suitable
for scikit-learn / XGBoost classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

from src.utils import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    FINE_RATINGS, RATING_ORDINAL, MODELS_DIR, ensure_dirs,
)


def encode_sector(df: pd.DataFrame, le: LabelEncoder = None,
                  fit: bool = True) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encode the 'sector' column.
    Returns the modified DataFrame and the fitted encoder.
    """
    df = df.copy()
    if 'sector' not in df.columns:
        df['sector'] = 0
        return df, le or LabelEncoder()
    if le is None:
        le = LabelEncoder()
    if fit:
        df['sector'] = le.fit_transform(df['sector'].astype(str))
    else:
        # Handle unseen sectors gracefully
        known = set(le.classes_)
        df['sector'] = df['sector'].apply(lambda x: x if x in known else le.classes_[0])
        df['sector'] = le.transform(df['sector'].astype(str))
    return df, le


def encode_target(y: pd.Series) -> tuple[np.ndarray, dict, dict]:
    """
    Encode fine-grained credit ratings as ordered integers.
    Returns (encoded_array, label_to_int, int_to_label).
    """
    label_to_int = {r: i for i, r in enumerate(FINE_RATINGS)}
    int_to_label = {i: r for r, i in label_to_int.items()}
    encoded = y.map(label_to_int).values.astype(int)
    return encoded, label_to_int, int_to_label


def scale_features(X: pd.DataFrame, scaler: StandardScaler = None,
                   fit: bool = True) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardise numeric features.
    Returns the scaled array and the fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def prepare_features(df: pd.DataFrame,
                     le: LabelEncoder = None,
                     scaler: StandardScaler = None,
                     fit: bool = True):
    """
    Full pipeline: encode categoricals → select features → scale.

    Returns:
        X_scaled (ndarray), y_encoded (ndarray), le, scaler,
        label_to_int, int_to_label, feature_names
    """
    df = df.copy()

    # Encode sector
    df, le = encode_sector(df, le=le, fit=fit)

    # Select features
    feature_cols = []
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale
    X_scaled, scaler = scale_features(X, scaler=scaler, fit=fit)

    # Target
    y_encoded, label_to_int, int_to_label = None, None, None
    if TARGET_COLUMN in df.columns:
        y_encoded, label_to_int, int_to_label = encode_target(df[TARGET_COLUMN])

    return X_scaled, y_encoded, le, scaler, label_to_int, int_to_label, feature_cols


def save_preprocessing_artifacts(le: LabelEncoder, scaler: StandardScaler,
                                  label_to_int: dict, int_to_label: dict,
                                  feature_cols: list):
    """Persist encoder, scaler, and label maps for inference."""
    ensure_dirs()
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_to_int, os.path.join(MODELS_DIR, "label_to_int.pkl"))
    joblib.dump(int_to_label, os.path.join(MODELS_DIR, "int_to_label.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))


def load_preprocessing_artifacts():
    """Load saved preprocessing artefacts."""
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_to_int = joblib.load(os.path.join(MODELS_DIR, "label_to_int.pkl"))
    int_to_label = joblib.load(os.path.join(MODELS_DIR, "int_to_label.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    return le, scaler, label_to_int, int_to_label, feature_cols
