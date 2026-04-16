import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — skipping XGB models.")

from preprocess import load_and_preprocess, get_feature_cols


def train_models():
    df, _ = load_and_preprocess()
    feature_cols = get_feature_cols()

    X = df[feature_cols].fillna(0)
    y_cls = df['Good_Investment']
    y_reg = df['Future_Price_5Y']

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    mlflow.set_experiment("RealEstate_Investment_Advisor")

    best_cls = {'name': None, 'model': None, 'auc': 0}
    best_reg = {'name': None, 'model': None, 'rmse': 1e9}

    # ──────────────────────────────────────────────────────────────
    # CLASSIFICATION MODELS
    # ──────────────────────────────────────────────────────────────
    cls_models = [
        ('LogisticRegression', LogisticRegression(max_iter=500, random_state=42)),
        ('RandomForest_Classifier', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
    ]
    if XGB_AVAILABLE:
        cls_models.append(('XGBoost_Classifier',
                            XGBClassifier(n_estimators=150, use_label_encoder=False,
                                          eval_metric='logloss', random_state=42)))

    for name, model in cls_models:
        with mlflow.start_run(run_name=name):
            X_tr = X_train_sc if name == 'LogisticRegression' else X_train
            X_te = X_test_sc if name == 'LogisticRegression' else X_test

            model.fit(X_tr, yc_train)
            preds = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]

            acc = accuracy_score(yc_test, preds)
            prec = precision_score(yc_test, preds)
            rec = recall_score(yc_test, preds)
            f1 = f1_score(yc_test, preds)
            auc = roc_auc_score(yc_test, proba)
            cm = confusion_matrix(yc_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metrics({
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
            })
            mlflow.sklearn.log_model(model, name)

            print(f"\n[CLS] {name}")
            print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

            if auc > best_cls['auc']:
                best_cls = {'name': name, 'model': model, 'auc': auc,
                            'scaled': name == 'LogisticRegression'}

    # ──────────────────────────────────────────────────────────────
    # REGRESSION MODELS
    # ──────────────────────────────────────────────────────────────
    reg_models = [
        ('LinearRegression', LinearRegression()),
        ('RandomForest_Regressor', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ]
    if XGB_AVAILABLE:
        reg_models.append(('XGBoost_Regressor',
                            XGBRegressor(n_estimators=150, random_state=42)))

    for name, model in reg_models:
        with mlflow.start_run(run_name=name):
            X_tr = X_train_sc if name == 'LinearRegression' else X_train
            X_te = X_test_sc if name == 'LinearRegression' else X_test

            model.fit(X_tr, yr_train)
            preds = model.predict(X_te)

            rmse = np.sqrt(mean_squared_error(yr_test, preds))
            mae = mean_absolute_error(yr_test, preds)
            r2 = r2_score(yr_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_metrics({
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
            })
            mlflow.sklearn.log_model(model, name)

            print(f"\n[REG] {name}")
            print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

            if rmse < best_reg['rmse']:
                best_reg = {'name': name, 'model': model, 'rmse': rmse,
                            'scaled': name == 'LinearRegression'}

    # ── Save best models ────────────────────────────────────────
    with open('models/best_classifier.pkl', 'wb') as f:
        pickle.dump(best_cls, f)
    with open('models/best_regressor.pkl', 'wb') as f:
        pickle.dump(best_reg, f)

    print(f"\n✅ Best Classifier : {best_cls['name']}  (AUC={best_cls['auc']:.4f})")
    print(f"✅ Best Regressor  : {best_reg['name']}  (RMSE={best_reg['rmse']:.4f})")
    print("Models saved to models/")


if __name__ == '__main__':
    train_models()
