# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def run_training(df_encoded):

    print("=" * 50)
    print("  STEP 4: Model Training")
    print("=" * 50)

    # ── 1. Random Forest ─────────────────────────────
    print("\n[1] Random Forest:")

    y = df_encoded['Approved_Flag']
    x = df_encoded.drop(['Approved_Flag'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print()
    print(f'Accuracy: {accuracy}')
    print()
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

    for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
        print(f"Class {v}:")
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}")
        print(f"F1 Score: {f1_score[i]}")
        print()

    # ── 2. XGBoost ───────────────────────────────────
    print("\n[2] XGBoost:")

    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    y = df_encoded['Approved_Flag']
    x = df_encoded.drop(['Approved_Flag'], axis=1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    xgb_classifier.fit(x_train, y_train)
    y_pred = xgb_classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print()
    print(f'Accuracy: {accuracy:.2f}')
    print()

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

    for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
        print(f"Class {v}:")
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}")
        print(f"F1 Score: {f1_score[i]}")
        print()

    # ── 3. Decision Tree ─────────────────────────────
    print("\n[3] Decision Tree:")

    y = df_encoded['Approved_Flag']
    x = df_encoded.drop(['Approved_Flag'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
    dt_model.fit(x_train, y_train)
    y_pred = dt_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print()
    print(f"Accuracy: {accuracy:.2f}")
    print()

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

    for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
        print(f"Class {v}:")
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}")
        print(f"F1 Score: {f1_score[i]}")
        print()

    print("  Done. XGBoost gave best results → proceeding to tuning.")

    return xgb_classifier, label_encoder
