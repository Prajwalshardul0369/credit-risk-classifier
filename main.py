# main.py
# ─────────────────────────────────────────────────────
#  Credit Risk ML  |  Full Pipeline
#  Run:  python main.py
# ─────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import run_preprocessing
from src.feature_selection   import run_feature_selection
from src.encoding            import run_encoding
from src.model_training      import run_training
from src.model_tuning        import run_tuning


def main():

    print("\n")
    print("*" * 50)
    print("     CREDIT RISK ML  —  PIPELINE START")
    print("*" * 50)

    # ── Step 1: Load & Clean Data ──────────────────
    df = run_preprocessing()

    # ── Step 2: Feature Selection ──────────────────
    df, columns_to_be_kept_numerical = run_feature_selection(df)

    # ── Step 3: Encode Features ────────────────────
    df_encoded = run_encoding(df)

    # ── Step 4: Train Models ───────────────────────
    xgb_model, label_encoder = run_training(df_encoded)

    # ── Step 5: Tune Best Model ────────────────────
    best_model = run_tuning(df_encoded)

    print("\n")
    print("*" * 50)
    print("     PIPELINE COMPLETE  ✓")
    print("*" * 50)
    print("\n")


if __name__ == '__main__':
    main()
