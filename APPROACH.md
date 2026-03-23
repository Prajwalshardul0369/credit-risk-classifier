# Project Approach & Decision Log

## Problem Statement
Predict the **credit risk tier** (P1–P4) of a loan applicant based on their demographic profile and credit history.
- **P1** → Lowest risk (most creditworthy)
- **P4** → Highest risk (least creditworthy)

---

## Pipeline Overview

```
Raw Excel Files
      │
      ▼
[Step 1] Data Preprocessing    → Merge applicant profile + credit history
      │
      ▼
[Step 2] Feature Selection      → Chi-square · VIF · ANOVA
      │
      ▼
[Step 3] Encoding               → Ordinal + One-Hot Encoding
      │
      ▼
[Step 4] Model Training         → RF · XGBoost · Decision Tree
      │
      ▼
Best Model: XGBoost ✅
```

---

## Step 1 — Data Preprocessing

- Sentinel value `-99999` represents missing data (common in bureau datasets)
- Columns in credit_history with **>10,000 rows** of `-99999` were **dropped**
- Remaining `-99999` rows were **removed**
- **Inner join** on `PROSPECTID` ensures no null-filled rows in final dataset

---

## Step 2 — Feature Selection

| Method       | Target          | Threshold    | Outcome                        |
|--------------|-----------------|--------------|--------------------------------|
| Chi-Square   | Categorical     | p ≤ 0.05     | All 5 categorical kept         |
| VIF          | Numerical       | VIF ≤ 6      | Multicollinear features removed|
| ANOVA        | Numerical       | p ≤ 0.05     | Low-signal features removed    |

VIF features are dropped **iteratively** (one at a time), not all at once.

---

## Step 3 — Encoding

**EDUCATION** → Ordinal (order matters):

| Label          | Value |
|----------------|-------|
| SSC            | 1     |
| 12TH           | 2     |
| GRADUATE       | 3     |
| UNDER GRADUATE | 3     |
| PROFESSIONAL   | 3     |
| POST-GRADUATE  | 4     |
| OTHERS         | 1 ⚠️  |

> `OTHERS = 1` is a conservative assumption — needs business confirmation.

**MARITALSTATUS, GENDER, last_prod_enq2, first_prod_enq2** → One-Hot Encoding

---

## Step 4 — Model Training

80/20 train-test split · `random_state=42` for reproducibility

| Model          | Notes                                           |
|----------------|-------------------------------------------------|
| Random Forest  | 200 estimators · good ensemble baseline         |
| XGBoost ✅     | Gradient boosting · best accuracy & F1          |
| Decision Tree  | max_depth=20 · interpretable but overfits       |

---

## Planned Improvements

1. Hyperparameter tuning (Optuna / GridSearchCV)
2. SMOTE for class imbalance handling
3. SHAP values for model explainability
4. Stratified K-Fold cross-validation
5. Business validation of `EDUCATION = OTHERS` mapping
