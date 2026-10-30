"""
predictive_model.py
Advanced Machine Learning Module for Proactive Logistics Exception Management.
Trains two production models:
1. TIT Delay Regressor (RandomForestRegressor): Predicts continuous transit delay (hours).
2. Demurrage Risk Classifier (RandomForestClassifier): Predicts probability of demurrage breach (>6h TAT).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_feature_dataset():
    shipments_file = os.path.join(DATA_DIR, "inbound_shipments.csv")
    tat_file = os.path.join(OUTPUT_DIR, "tat_simulation_results.csv")

    df_shipments = pd.read_csv(shipments_file)
    df_tat = pd.read_csv(tat_file)

    # Filter to baseline FCFS to model true raw incoming operational risk
    df_fcfs = df_tat[df_tat["policy"] == "BASELINE_FCFS"][["shipment_id", "total_tat_hours", "demurrage_flag"]]

    # Merge shipment features with resulting TAT & demurrage outcome
    df = pd.merge(df_shipments, df_fcfs, on="shipment_id")

    # Feature Engineering
    df["dispatch_dt"] = pd.to_datetime(df["dispatch_timestamp"])
    df["dispatch_hour"] = df["dispatch_dt"].dt.hour
    df["dispatch_dayofweek"] = df["dispatch_dt"].dt.dayofweek
    df["is_month_end"] = (df["dispatch_dt"].dt.day >= 25).astype(int)

    return df

def train_predictive_models():
    print("[ML Engine] Preparing features from historical distribution logs...")
    df = prepare_feature_dataset()

    features_num = [
        "distance_km", "planned_tit_hours", "payload_tons", 
        "pallet_count", "dispatch_hour", "dispatch_dayofweek", 
        "is_month_end", "stockout_urgency_score"
    ]
    features_cat = ["origin_cdc", "transporter", "fleet_type", "category"]

    X = df[features_num + features_cat]
    y_reg = df["tit_variance_hours"] # Continuous delay in hours
    y_clf = df["demurrage_flag"]       # Binary demurrage breach flag

    # Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat)
        ]
    )

    # 1. Regressor Pipeline: Predict Delay Hours
    print("[ML Engine] Training TIT Delay Regressor (Random Forest)...")
    reg_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1))
    ])
    reg_pipe.fit(X_train, y_reg_train)

    y_reg_pred = reg_pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)
    print(f"  -> Regressor Performance: RMSE = {rmse:.2f} hrs | R² Score = {r2:.3f}")

    # 2. Classifier Pipeline: Predict Demurrage Risk Probability
    print("[ML Engine] Training Demurrage Risk Classifier (Random Forest)...")
    clf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    clf_pipe.fit(X_train, y_clf_train)

    y_clf_prob = clf_pipe.predict_proba(X_test)[:, 1]
    y_clf_pred = clf_pipe.predict(X_test)
    roc_auc = roc_auc_score(y_clf_test, y_clf_prob)
    print(f"  -> Classifier Performance: ROC-AUC = {roc_auc:.3f}")

    # Save artifacts
    reg_path = os.path.join(MODEL_DIR, "tit_delay_regressor.joblib")
    clf_path = os.path.join(MODEL_DIR, "demurrage_classifier.joblib")
    joblib.dump(reg_pipe, reg_path)
    joblib.dump(clf_pipe, clf_path)

    print(f"[ML Engine] Models successfully serialized to:\n  - {reg_path}\n  - {clf_path}")

    # Feature Importance Extraction
    cat_encoder = reg_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(features_cat))
    all_feature_names = features_num + cat_feature_names

    importances = reg_pipe.named_steps["regressor"].feature_importances_
    feat_imp = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    feat_imp.sort_values(by="importance", ascending=False, inplace=True)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "ml_feature_importances.csv"), index=False)
    print(f"[ML Engine] Feature importances saved -> {os.path.join(OUTPUT_DIR, 'ml_feature_importances.csv')}")

    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "roc_auc": round(roc_auc, 3)
    }

if __name__ == "__main__":
    train_predictive_models()
