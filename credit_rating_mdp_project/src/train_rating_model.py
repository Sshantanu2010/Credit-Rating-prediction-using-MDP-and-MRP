"""
train_rating_model.py — Train and evaluate credit-rating classifiers.

Trains Logistic Regression, Random Forest, and XGBoost on the synthetic
Indian corporate dataset, selects the best model via cross-validation,
and persists the winner along with evaluation metrics.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

from src.utils import MODELS_DIR, FINE_RATINGS, ensure_dirs
from src.data_preprocessing import load_dataset, clean_dataframe
from src.feature_engineering import (
    prepare_features, save_preprocessing_artifacts
)

warnings.filterwarnings("ignore")

# Try XGBoost; fall back to sklearn GradientBoosting
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _build_candidates() -> dict:
    """Return a dict of named model instances to evaluate."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, multi_class='multinomial', solver='lbfgs',
            class_weight='balanced', C=1.0, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        ),
    }
    if _HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
    else:
        candidates["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
    return candidates


def train_and_select(df: pd.DataFrame = None, test_size: float = 0.2,
                     cv_folds: int = 5):
    """
    Full training pipeline:
      1. Load & clean data
      2. Feature-engineer
      3. Cross-validate three classifiers
      4. Select best on macro-F1
      5. Retrain best on full training set
      6. Persist model + artefacts

    Returns:
        best_name, best_model, metrics (dict), X_test, y_test, int_to_label
    """
    # --- data ---
    if df is None:
        df = load_dataset()
    df = clean_dataframe(df)

    X_scaled, y_encoded, le, scaler, l2i, i2l, feat_cols = prepare_features(
        df, fit=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size,
        random_state=42, stratify=y_encoded,
    )

    # --- evaluate candidates ---
    candidates = _build_candidates()
    results = {}

    for name, model in candidates.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds,
            scoring='f1_macro', n_jobs=-1,
        )
        results[name] = {
            'cv_f1_mean': float(np.mean(scores)),
            'cv_f1_std':  float(np.std(scores)),
        }

    # --- pick winner ---
    best_name = max(results, key=lambda k: results[k]['cv_f1_mean'])
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    # --- test-set evaluation ---
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    cm  = confusion_matrix(y_test, y_pred)

    # Classification report as dict
    used_labels = sorted(set(y_test) | set(y_pred))
    target_names = [i2l[i] for i in used_labels]
    report = classification_report(
        y_test, y_pred, labels=used_labels,
        target_names=target_names, output_dict=True, zero_division=0,
    )

    metrics = {
        'best_model': best_name,
        'test_accuracy': float(acc),
        'test_f1_macro': float(f1),
        'cv_results': results,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_labels': target_names,
    }

    # --- feature importance ---
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        fi = np.mean(np.abs(best_model.coef_), axis=0)
    else:
        fi = np.zeros(len(feat_cols))
    metrics['feature_importance'] = dict(zip(feat_cols, fi.tolist()))

    # --- persist ---
    ensure_dirs()
    joblib.dump(best_model, os.path.join(MODELS_DIR, "rating_model.pkl"))
    save_preprocessing_artifacts(le, scaler, l2i, i2l, feat_cols)
    with open(os.path.join(MODELS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also train and save all models for comparison
    all_models = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        all_models[name] = model
    joblib.dump(all_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    return best_name, best_model, metrics, X_test, y_test, i2l


def load_trained_model():
    """Load the persisted best model and preprocessing artefacts."""
    model = joblib.load(os.path.join(MODELS_DIR, "rating_model.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, metrics


def is_model_trained() -> bool:
    """Check whether a trained model already exists."""
    return os.path.exists(os.path.join(MODELS_DIR, "rating_model.pkl"))


# ═══════════════════════════════════════════════════════════════
# CLI entry-point for standalone training
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.data_preprocessing import initialise_data
    initialise_data()
    best_name, _, metrics, *_ = train_and_select()
    print(f"\n✅  Best model: {best_name}")
    print(f"    Test accuracy : {metrics['test_accuracy']:.3f}")
    print(f"    Test F1-macro : {metrics['test_f1_macro']:.3f}")
    print(f"    CV results    : {metrics['cv_results']}")
