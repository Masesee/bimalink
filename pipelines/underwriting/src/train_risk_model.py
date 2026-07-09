# Chosen: XGBoost - standard, robust default parameters, and has excellent native SHAP support for tree-based models.
import sys
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import shap

# Add the parent directory to the python path to allow importing schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess_features(df, encoder_config=None):
    """
    Preprocess features for training and scoring.
    One-hot encodes categorical columns and aligns columns with training space.
    """
    df = df.copy()

    categorical_cols = ["occupation", "avg_daily_income_band"]

    # Remove metadata and redundant columns if present
    for col in ["phone_number_hash", "data_provenance", "years_active"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Standardize sacco contribution flag
    if "sacco_contribution_flag" in df.columns:
        df["sacco_contribution_flag"] = df["sacco_contribution_flag"].astype(int)

    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    if encoder_config is None:
        # Training mode: capture the exact feature names and ordering
        feature_names = list(df_encoded.columns)
        if "defaulted_or_claimed" in feature_names:
            feature_names.remove("defaulted_or_claimed")
        return df_encoded, {"feature_names": feature_names}
    else:
        # Scoring mode: reindex columns to match saved training columns
        feature_names = encoder_config["feature_names"]
        if "defaulted_or_claimed" in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=["defaulted_or_claimed"])
        df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
        return df_aligned


MIN_TIER_GAP = 0.05  # Enforces a minimum 5% gap floor between consecutive pricing tiers to absorb sampling noise


def map_default_probability_to_tier(prob: float) -> tuple[str, int]:
    """
    Maps default probability to risk tier and premium quote in KES.
    Actuarial pricing is out of scope for this demo; these are illustrative placeholder numbers.
    """
    # Low risk -> Bronze tier
    if prob < 0.15:
        return "Low", 150
    # Medium risk -> Silver tier
    elif prob < 0.35:
        return "Medium", 300
    # High risk -> Gold tier
    else:
        return "High", 500


def train_pipeline(data_path, models_dir):
    print(f"[LOG] Loading synthetic profiles from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate features and target
    X_raw = df.drop(columns=["defaulted_or_claimed"])
    y = df["defaulted_or_claimed"]

    # Preprocess features
    X_encoded, encoder_config = preprocess_features(X_raw)

    # Drop target column if pd.get_dummies kept it (it shouldn't, but let's be safe)
    if "defaulted_or_claimed" in X_encoded.columns:
        X_encoded = X_encoded.drop(columns=["defaulted_or_claimed"])

    # Split into train and test (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Carve validation set from train for early stopping (10% of train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"[LOG] Train shape: {X_tr.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Initialize XGBoost classifier with early stopping in construction
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        early_stopping_rounds=15,
        eval_metric="logloss"
    )

    # Train model
    print("[LOG] Fitting XGBoost model with early stopping on validation set...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluate on test set
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate default rate per risk tier in test set
    test_results = pd.DataFrame({
        "default": y_test,
        "prob": y_pred_prob
    })

    tiers = []
    for p in y_pred_prob:
        tier, _ = map_default_probability_to_tier(p)
        tiers.append(tier)
    test_results["tier"] = tiers

    print("\n--- TEST SET DEFAULT RATES BY PRICING TIER ---")
    tier_rates = {}
    for tier_name in ["Low", "Medium", "High"]:
        subset = test_results[test_results["tier"] == tier_name]
        total = len(subset)
        defaults = subset["default"].sum()
        def_rate = (defaults / total) if total > 0 else 0.0
        tier_rates[tier_name] = def_rate
        print(f"Tier: {tier_name:<6} | Total: {total:<4} | Defaults: {defaults:<3} | Default Rate: {def_rate:.2%}")
    print("----------------------------------------------\n")

    # Business logical validation: Risk pricing tiers must be monotonically sorted and separate risk
    med_low_gap = tier_rates["Medium"] - tier_rates["Low"]
    high_med_gap = tier_rates["High"] - tier_rates["Medium"]
    if med_low_gap < MIN_TIER_GAP or high_med_gap < MIN_TIER_GAP:
        raise ValueError(
            "CRITICAL ERROR: Pricing tiers are not risk-sorted with the required gap of "
            f"{MIN_TIER_GAP:.0%}! Low={tier_rates['Low']:.2%}, "
            f"Medium={tier_rates['Medium']:.2%}, High={tier_rates['High']:.2%}"
        )

    print("\n--- SYNTHETIC RISK MODEL EVALUATION METRICS ---")
    print(f"Test ROC-AUC Score   : {auc:.4f}")
    print(f"Test Precision       : {precision:.4f}")
    print(f"Test Recall          : {recall:.4f}")
    print(f"Test F1 Score        : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("------------------------------------------------\n")

    # Fail loudly on out-of-band ROC-AUC
    if auc < 0.65:
        raise ValueError(
            f"CRITICAL ERROR: Test ROC-AUC of {auc:.4f} is below minimum threshold of 0.65. "
            "Model did not learn."
        )
    if auc > 0.97:
        raise ValueError(
            f"CRITICAL ERROR: Test ROC-AUC of {auc:.4f} is above maximum threshold of 0.97. "
            "Target leakage detected."
        )

    # Fit SHAP Explainer
    print("[LOG] Fitting SHAP TreeExplainer on trained model...")
    explainer = shap.TreeExplainer(model)

    # Create models directory
    os.makedirs(models_dir, exist_ok=True)

    # Save artifacts
    model_path = os.path.join(models_dir, "model.joblib")
    explainer_path = os.path.join(models_dir, "explainer.joblib")
    encoder_path = os.path.join(models_dir, "encoder_config.json")

    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    with open(encoder_path, "w") as f:
        json.dump(encoder_config, f, indent=2)

    print(f"[LOG] Saved trained model to: {model_path}")
    print(f"[LOG] Saved SHAP explainer to: {explainer_path}")
    print(f"[LOG] Saved encoder config to: {encoder_path}")
    print("[LOG] Model training pipeline run completed successfully.")


if __name__ == "__main__":
    data_file = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    models_folder = os.path.join("pipelines", "underwriting", "models")
    train_pipeline(data_file, models_folder)
