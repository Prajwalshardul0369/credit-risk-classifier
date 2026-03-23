# src/data_preprocessing.py

import pandas as pd


def run_preprocessing():

    print("=" * 50)
    print("  STEP 1: Data Preprocessing")
    print("=" * 50)

    # Load the dataset
    a1 = pd.read_excel("credit_risk_applicant_profile.xlsx")
    a2 = pd.read_excel("credit_risk_credit_history.xlsx")

    df1 = a1.copy()
    df2 = a2.copy()

    # Remove nulls
    df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

    columns_to_be_removed = []

    for i in df2.columns:
        if df2.loc[df2[i] == -99999].shape[0] > 10000:
            columns_to_be_removed.append(i)

    df2 = df2.drop(columns_to_be_removed, axis=1)

    for i in df2.columns:
        df2 = df2.loc[ df2[i] != -99999 ]

    # Checking common column names
    print("\nCommon columns between df1 and df2:")
    for i in list(df1.columns):
        if i in list(df2.columns):
            print(i)

    # Merge the two dataframes, inner join so that no nulls are present
    df = pd.merge(df1, df2, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

    # check how many columns are categorical
    print("\nCategorical columns:")
    for i in df.columns:
        if df[i].dtype == 'object':
            print(i)

    print(f"\n  Done. Merged dataframe shape: {df.shape}")

    return df
