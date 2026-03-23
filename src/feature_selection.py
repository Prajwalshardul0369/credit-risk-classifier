# src/feature_selection.py

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_feature_selection(df):

    print("=" * 50)
    print("  STEP 2: Feature Selection")
    print("=" * 50)

    # ── Chi-square test ──────────────────────────────
    print("\n[Chi-square] Categorical features:")
    for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
        chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
        print(i, '---', pval)

    # Since all the categorical features have pval <=0.05, we will accept all

    # ── VIF for numerical columns ────────────────────
    print("\n[VIF] Numerical features:")
    numeric_columns = []
    for i in df.columns:
        if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
            numeric_columns.append(i)

    vif_data = df[numeric_columns]
    total_columns = vif_data.shape[1]
    columns_to_be_kept = []
    column_index = 0

    for i in range(0, total_columns):
        vif_value = variance_inflation_factor(vif_data, column_index)
        print(column_index, '---', vif_value)

        if vif_value <= 6:
            columns_to_be_kept.append(numeric_columns[i])
            column_index = column_index + 1
        else:
            vif_data = vif_data.drop([numeric_columns[i]], axis=1)

    # ── ANOVA for columns_to_be_kept ─────────────────
    print("\n[ANOVA] Numerical features:")
    columns_to_be_kept_numerical = []

    for i in columns_to_be_kept:
        a = list(df[i])
        b = list(df['Approved_Flag'])

        group_P1 = [value for value, group in zip(a, b) if group == 'P1']
        group_P2 = [value for value, group in zip(a, b) if group == 'P2']
        group_P3 = [value for value, group in zip(a, b) if group == 'P3']
        group_P4 = [value for value, group in zip(a, b) if group == 'P4']

        f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

        if p_value <= 0.05:
            columns_to_be_kept_numerical.append(i)

    # feature selection is done for cat and num features

    # listing all the final features
    features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    df = df[features + ['Approved_Flag']]

    print(f"\n  Done. Selected {len(columns_to_be_kept_numerical)} numerical + 5 categorical features.")

    return df, columns_to_be_kept_numerical
