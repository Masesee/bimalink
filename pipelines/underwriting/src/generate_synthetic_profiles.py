import sys
import os
import hashlib
import numpy as np
import pandas as pd

# Add the parent directory to the python path to allow importing schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.contracts import UserProfile, SelfReported, SyntheticHistorical, LiveBehavioral

def generate_synthetic_data(num_rows=3000, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    profiles = []
    labels = []

    # Documented weights for risk probability logit calculation
    # logit = intercept (0.8)
    #       - 3.0 * regularity_score (higher regularity -> lower default)
    #       - 1.5 * frequency_ratio (higher frequency -> lower default)
    #       + 2.5 * cadence_ratio (longer cadence -> higher default)
    #       - 1.2 * sacco_flag (sacco contribution -> lower default)
    #       - 1.0 * years_over_3 (active over 3 years -> lower default)
    #       - 1.0 * income_over_1500 (income over 1500 -> lower default)
    #       + gaussian_noise (std = 1.2)
    
    for i in range(num_rows):
        # 1. Generate phone number hash
        raw_phone = f"+25471234{i:04d}"
        phone_hash = hashlib.sha256(raw_phone.encode()).hexdigest()

        # 2. Self reported fields
        # Occupation distribution: boda_rider (40%), market_trader (40%), other (20%)
        occupation = rng.choice(["boda_rider", "market_trader", "other"], p=[0.4, 0.4, 0.2])
        
        # Avg daily income band (KES): under_500, 500_to_1500, over_1500
        # Income correlates slightly with occupation
        if occupation == "boda_rider":
            avg_daily_income_band = rng.choice(["under_500", "500_to_1500", "over_1500"], p=[0.25, 0.60, 0.15])
        elif occupation == "market_trader":
            avg_daily_income_band = rng.choice(["under_500", "500_to_1500", "over_1500"], p=[0.20, 0.50, 0.30])
        else:
            avg_daily_income_band = rng.choice(["under_500", "500_to_1500", "over_1500"], p=[0.33, 0.34, 0.33])

        # Years active: under_1, 1_to_3, over_3
        years_active = rng.choice(["under_1", "1_to_3", "over_3"], p=[0.20, 0.50, 0.30])

        # 3. Synthetic historical fields
        # momo_txn_frequency: transactions per week, exponential-like
        momo_txn_frequency = float(rng.exponential(scale=12.0) + 1.0)
        
        # momo_txn_regularity_score: 0-1, higher = more regular intervals
        momo_txn_regularity_score = float(rng.uniform(0.2, 0.95))

        # airtime_topup_cadence: days between top-ups, average
        airtime_topup_cadence = float(rng.gamma(shape=3.0, scale=1.5) + 0.5)

        # sacco_contribution_flag
        sacco_contribution_flag = bool(rng.choice([True, False], p=[0.3, 0.7]))

        # 4. Live behavioral fields (session data, independent of target except menu_completion_rate)
        ussd_session_duration_sec = float(rng.exponential(scale=30.0) + 5.0)
        
        # menu_completion_rate correlates weakly with regularity score
        raw_completion = momo_txn_regularity_score * 0.3 + rng.normal(loc=0.5, scale=0.15)
        menu_completion_rate = float(np.clip(raw_completion, 0.0, 1.0))

        session_hour_of_day = int(np.clip(int(rng.normal(loc=13.0, scale=4.0)), 0, 23))
        retry_count = int(rng.poisson(lam=0.8))

        # Build objects to validate contract schema
        self_reported = SelfReported(
            occupation=occupation,
            avg_daily_income_band=avg_daily_income_band,
            years_active=years_active
        )
        
        synthetic_historical = SyntheticHistorical(
            momo_txn_frequency=momo_txn_frequency,
            momo_txn_regularity_score=momo_txn_regularity_score,
            airtime_topup_cadence=airtime_topup_cadence,
            sacco_contribution_flag=sacco_contribution_flag
        )
        
        live_behavioral = LiveBehavioral(
            ussd_session_duration_sec=ussd_session_duration_sec,
            menu_completion_rate=menu_completion_rate,
            session_hour_of_day=session_hour_of_day,
            retry_count=retry_count
        )

        user_profile = UserProfile(
            phone_number_hash=phone_hash,
            self_reported=self_reported,
            synthetic_historical=synthetic_historical,
            live_behavioral=live_behavioral
        )

        # 5. Compute Risk target label defaulted_or_claimed (0 or 1)
        # Scale inputs for logit calculations
        freq_ratio = min(momo_txn_frequency / 50.0, 1.0)
        cadence_ratio = min(airtime_topup_cadence / 15.0, 1.0)
        sacco_val = 1.0 if sacco_contribution_flag else 0.0
        years_val = 1.0 if years_active == "over_3" else 0.0
        income_val = 1.0 if avg_daily_income_band == "over_1500" else 0.0

        logit = (
            1.2
            - 5.0 * momo_txn_regularity_score
            - 3.0 * freq_ratio
            + 4.0 * cadence_ratio
            - 2.0 * sacco_val
            - 1.5 * years_val
            - 1.5 * income_val
        )

        # Add Gaussian noise
        noise = rng.normal(loc=0.0, scale=0.3)
        full_logit = logit + noise
        prob_default = 1.0 / (1.0 + np.exp(-full_logit))

        defaulted = int(rng.binomial(1, prob_default))

        profiles.append(user_profile)
        labels.append(defaulted)

    # Flatten profiles to create DataFrame
    flat_data = []
    for profile, label in zip(profiles, labels):
        flat_data.append({
            "phone_number_hash": profile.phone_number_hash,
            "occupation": profile.self_reported.occupation,
            "avg_daily_income_band": profile.self_reported.avg_daily_income_band,
            "years_active": profile.self_reported.years_active,
            "momo_txn_frequency": profile.synthetic_historical.momo_txn_frequency,
            "momo_txn_regularity_score": profile.synthetic_historical.momo_txn_regularity_score,
            "airtime_topup_cadence": profile.synthetic_historical.airtime_topup_cadence,
            "sacco_contribution_flag": int(profile.synthetic_historical.sacco_contribution_flag),
            "ussd_session_duration_sec": profile.live_behavioral.ussd_session_duration_sec,
            "menu_completion_rate": profile.live_behavioral.menu_completion_rate,
            "session_hour_of_day": profile.live_behavioral.session_hour_of_day,
            "retry_count": profile.live_behavioral.retry_count,
            "defaulted_or_claimed": label
        })

    df = pd.DataFrame(flat_data)
    return df

def main():
    print("[LOG] Starting generation of synthetic user profiles...")
    
    # Create target directory if it doesn't exist
    data_dir = os.path.join("pipelines", "underwriting", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    df = generate_synthetic_data(num_rows=3000, seed=42)

    # Label synthetic explicitly in file paths and logs
    full_path = os.path.join(data_dir, "synthetic_profiles.csv")
    sample_path = os.path.join(data_dir, "sample_10_rows.csv")

    df.to_csv(full_path, index=False)
    df.head(10).to_csv(sample_path, index=False)

    print(f"[LOG] Successfully generated and saved 3000 synthetic profiles to: {full_path}")
    print(f"[LOG] Saved a sample of 10 synthetic rows for reference to: {sample_path}")

    # Summary stats
    default_rate = df["defaulted_or_claimed"].mean()
    class_counts = df["defaulted_or_claimed"].value_counts().to_dict()
    
    print("\n--- SYNTHETIC DATA GENERATION SUMMARY STATS ---")
    print(f"Total Synthetic Rows: {len(df)}")
    print(f"Synthetic Target Class Balance: 0={class_counts.get(0, 0)}, 1={class_counts.get(1, 0)} (Default Rate: {default_rate:.2%})")
    
    print("\nFeature Means by Synthetic Class (defaulted_or_claimed):")
    numeric_cols = [
        "momo_txn_frequency", 
        "momo_txn_regularity_score", 
        "airtime_topup_cadence", 
        "sacco_contribution_flag",
        "menu_completion_rate"
    ]
    means_df = df.groupby("defaulted_or_claimed")[numeric_cols].mean()
    print(means_df.to_string())
    
    print("\nCategorical Distributions by Synthetic Class:")
    for cat_col in ["occupation", "avg_daily_income_band", "years_active"]:
        print(f"\nDistribution for {cat_col}:")
        dist = df.groupby(["defaulted_or_claimed", cat_col]).size().unstack(fill_value=0)
        # Normalize by class totals
        dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
        print(dist_pct.round(1).astype(str) + "%")
    print("-----------------------------------------------\n")

if __name__ == "__main__":
    main()
