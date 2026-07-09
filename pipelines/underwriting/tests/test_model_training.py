import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_risk_model import preprocess_features  # noqa: E402

MIN_TIER_GAP = 0.05  # Enforces a minimum 5% gap floor between consecutive pricing tiers to absorb sampling noise


def test_artifacts_exist():
    models_dir = os.path.join("pipelines", "underwriting", "models")
    model_path = os.path.join(models_dir, "model.joblib")
    explainer_path = os.path.join(models_dir, "explainer.joblib")
    encoder_path = os.path.join(models_dir, "encoder_config.json")

    assert os.path.exists(model_path), f"model.joblib not found in {models_dir}"
    assert os.path.exists(explainer_path), f"explainer.joblib not found in {models_dir}"
    assert os.path.exists(encoder_path), f"encoder_config.json not found in {models_dir}"


def test_saved_model_auc_range():
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    models_dir = os.path.join("pipelines", "underwriting", "models")

    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns=["defaulted_or_claimed"])
    y = df["defaulted_or_claimed"]

    # Load encoder config and preprocess
    with open(os.path.join(models_dir, "encoder_config.json"), "r") as f:
        encoder_config = json.load(f)

    X_encoded = preprocess_features(X_raw, encoder_config)

    # Perform the exact same train/test split as training to isolate test set
    _, X_test, _, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the trained model
    model = joblib.load(os.path.join(models_dir, "model.joblib"))

    # Evaluate
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    assert 0.65 <= auc <= 0.97, f"Trained model test ROC-AUC score {auc:.4f} is outside [0.65, 0.97]"


def test_pricing_tier_risk_sorted():
    """
    Enforces that actual default rates within the three risk pricing tiers
    scale monotonically (Low < Medium < High) in the test set.

    NOTE ON VARIANCE:
    The High Risk tier has a relatively small sample size (n=56 on 600 test rows).
    This small n results in higher sampling variance. If a new seed or data split
    causes a transient overlap in rates, re-run with a larger sample size or review
    the logit weights rather than assuming model degradation.
    """
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    models_dir = os.path.join("pipelines", "underwriting", "models")

    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns=["defaulted_or_claimed"])
    y = df["defaulted_or_claimed"]

    with open(os.path.join(models_dir, "encoder_config.json"), "r") as f:
        encoder_config = json.load(f)

    X_encoded = preprocess_features(X_raw, encoder_config)

    _, X_test, _, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(os.path.join(models_dir, "model.joblib"))
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    from src.train_risk_model import map_default_probability_to_tier
    test_results = pd.DataFrame({"default": y_test, "prob": y_pred_prob})

    tiers = [map_default_probability_to_tier(p)[0] for p in y_pred_prob]
    test_results["tier"] = tiers

    tier_rates = {}
    for tier_name in ["Low", "Medium", "High"]:
        subset = test_results[test_results["tier"] == tier_name]
        tier_rates[tier_name] = subset["default"].mean() if len(subset) > 0 else 0.0

    med_low = tier_rates["Medium"] - tier_rates["Low"]
    high_med = tier_rates["High"] - tier_rates["Medium"]
    assert med_low >= MIN_TIER_GAP, \
        f"Low-to-Medium pricing gap ({med_low:.2%}) is below floor {MIN_TIER_GAP:.0%}"
    assert high_med >= MIN_TIER_GAP, \
        f"Medium-to-High pricing gap ({high_med:.2%}) is below floor {MIN_TIER_GAP:.0%}"


def test_single_batch_overfit():
    # Make sure we use a stratified, deterministic split to get 20 rows containing both classes
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    df = pd.read_csv(csv_path)

    X_raw = df.drop(columns=["defaulted_or_claimed"])
    y = df["defaulted_or_claimed"]

    X_encoded, _ = preprocess_features(X_raw)

    # Start with a larger stratified sample
    X_train_full, _, y_train_full, _ = train_test_split(
        X_encoded, y, test_size=0.8, random_state=42, stratify=y
    )

    # Extract exactly 10 rows from class 0 and 10 rows from class 1 for a stratified subset
    class_0_indices = np.where(y_train_full.values == 0)[0][:10]
    class_1_indices = np.where(y_train_full.values == 1)[0][:10]

    subset_indices = np.concatenate([class_0_indices, class_1_indices])

    X_sub = X_train_full.iloc[subset_indices]
    y_sub = y_train_full.iloc[subset_indices]

    # Train on these 20 rows for many iterations to confirm the model can learn and memorize perfectly
    overfit_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.3,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )

    overfit_model.fit(X_sub, y_sub)

    train_acc = overfit_model.score(X_sub, y_sub)
    assert train_acc >= 0.99, f"Overfit test failed: accuracy was only {train_acc:.4f}, expected >= 0.99"
