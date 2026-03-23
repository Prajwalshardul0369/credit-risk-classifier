# src/encoding.py

import pandas as pd


def run_encoding(df):

    print("=" * 50)
    print("  STEP 3: Encoding")
    print("=" * 50)

    # Label encoding for the categorical features
    df['MARITALSTATUS'].unique()
    df['EDUCATION'].unique()
    df['GENDER'].unique()
    df['last_prod_enq2'].unique()
    df['first_prod_enq2'].unique()

    # Ordinal feature -- EDUCATION
    # SSC            : 1
    # 12TH           : 2
    # GRADUATE       : 3
    # UNDER GRADUATE : 3
    # POST-GRADUATE  : 4
    # OTHERS         : 1
    # PROFESSIONAL   : 3

    # Others has to be verified by the business end user

    df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
    df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
    df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
    df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
    df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
    df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
    df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3

    df['EDUCATION'].value_counts()
    df['EDUCATION'] = df['EDUCATION'].astype(int)
    df.info()

    df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

    df_encoded.info()
    k = df_encoded.describe()

    print(f"\n  Done. Encoded dataframe shape: {df_encoded.shape}")

    return df_encoded
