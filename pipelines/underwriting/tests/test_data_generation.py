import os
import pandas as pd


def test_synthetic_data_integrity():
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")

    # Assert file exists
    assert os.path.exists(csv_path), f"Synthetic profiles CSV not found at {csv_path}"

    # Load data
    df = pd.read_csv(csv_path)

    # Assert row count
    assert len(df) == 3000, f"Expected 3000 rows, got {len(df)}"

    # Assert no nulls
    assert df.isnull().sum().sum() == 0, "Synthetic dataset contains null values"


def test_synthetic_class_balance():
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    df = pd.read_csv(csv_path)

    default_rate = df["defaulted_or_claimed"].mean()

    # Assert default rate is in the 10% to 40% band
    assert 0.10 <= default_rate <= 0.40, f"Default rate {default_rate:.2%} is outside the [10%, 40%] band"


def test_directional_feature_relationships():
    csv_path = os.path.join("pipelines", "underwriting", "data", "synthetic_profiles.csv")
    df = pd.read_csv(csv_path)

    # Group by default class
    class_means = df.groupby("defaulted_or_claimed").mean(numeric_only=True)

    # 1. Higher momo_txn_regularity_score -> lower default (mean for class 0 > class 1)
    assert class_means.loc[0, "momo_txn_regularity_score"] > class_means.loc[1, "momo_txn_regularity_score"], \
        "Directional check failed: Higher momo regularity should correspond to lower default risk"

    # 2. Higher momo_txn_frequency -> lower default (mean for class 0 > class 1)
    assert class_means.loc[0, "momo_txn_frequency"] > class_means.loc[1, "momo_txn_frequency"], \
        "Directional check failed: Higher momo frequency should correspond to lower default risk"

    # 3. Longer airtime top-up cadence -> higher default (mean for class 1 > class 0)
    assert class_means.loc[1, "airtime_topup_cadence"] > class_means.loc[0, "airtime_topup_cadence"], \
        "Directional check failed: Longer airtime cadence should correspond to higher default risk"

    # 4. Sacco contribution flag -> lower default (mean for class 0 > class 1)
    assert class_means.loc[0, "sacco_contribution_flag"] > class_means.loc[1, "sacco_contribution_flag"], \
        "Directional check failed: SACCO membership should correspond to lower default risk"
