import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import MODEL_DATASET_FILE, MODEL_FILE, SCALER_FILE, REPORTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_aqi_category(aqi: float) -> str:
    """
    Returns CPCB AQI category name for a given AQI value.
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def train_aqi_model():
    logger.info("Starting model training stage...")
    
    if not os.path.exists(MODEL_DATASET_FILE):
        raise FileNotFoundError(f"Model dataset not found at {MODEL_DATASET_FILE}. Run preprocessing first.")
        
    df = pd.read_csv(MODEL_DATASET_FILE)
    logger.info(f"Loaded modeling dataset with {len(df)} rows.")
    
    # Define features and target
    features = ["HCHO", "NO2", "CO", "aerosol_index", "temp", "RH", "wind_speed", "fire_count"]
    target = "AQI"
    
    # 1. Time-based train/test split (last 20% of days held out)
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    logger.info(f"Split details: {len(train_dates)} training days, {len(test_dates)} testing days.")
    
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 2. Scale features (fit on train split only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"StandardScaler saved to {SCALER_FILE}")
    
    # 3. Primary Model: Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    logger.info("Training Primary Model: RandomForestRegressor...")
    rf_model.fit(X_train_scaled, y_train)
    logger.info("RandomForestRegressor training complete.")
    
    # Save Random Forest as the primary model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(rf_model, f)
    logger.info(f"Primary model (Random Forest) saved to {MODEL_FILE}")
    
    # 4. Optional Comparison Model: XGBoost
    xgb_success = False
    xgb_model = None
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        logger.info("Training Comparison Model: XGBoost Regressor...")
        xgb_model.fit(X_train_scaled, y_train)
        logger.info("XGBoost training complete.")
        xgb_success = True
    except ImportError:
        logger.info("XGBoost module not found. Skipping XGBoost comparison model.")

    # 5. Predict and evaluate both
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_rf = np.clip(y_pred_rf, 0, 500)
    
    rf_rmse = float(root_mean_squared_error(y_test, y_pred_rf))
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    
    y_test_cat = [get_aqi_category(v) for v in y_test]
    y_pred_rf_cat = [get_aqi_category(v) for v in y_pred_rf]
    rf_acc = float(accuracy_score(y_test_cat, y_pred_rf_cat))
    
    logger.info(f"RF Test Set Metrics - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}, Accuracy: {rf_acc*100:.1f}%")
    
    comparison_metrics = {
        "Random Forest": {
            "rmse": rf_rmse,
            "mae": rf_mae,
            "r2": rf_r2,
            "category_accuracy": rf_acc
        }
    }
    
    if xgb_success and xgb_model is not None:
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        y_pred_xgb = np.clip(y_pred_xgb, 0, 500)
        
        xgb_rmse = float(root_mean_squared_error(y_test, y_pred_xgb))
        xgb_mae = float(mean_absolute_error(y_test, y_pred_xgb))
        xgb_r2 = float(r2_score(y_test, y_pred_xgb))
        
        y_pred_xgb_cat = [get_aqi_category(v) for v in y_pred_xgb]
        xgb_acc = float(accuracy_score(y_test_cat, y_pred_xgb_cat))
        
        logger.info(f"XGB Test Set Metrics - RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}, R²: {xgb_r2:.2f}, Accuracy: {xgb_acc*100:.1f}%")
        
        comparison_metrics["XGBoost"] = {
            "rmse": xgb_rmse,
            "mae": xgb_mae,
            "r2": xgb_r2,
            "category_accuracy": xgb_acc
        }
        
    # Save comparison metrics JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    comparison_file = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison_metrics, f, indent=4)
    logger.info(f"Model comparison report saved to {comparison_file}")
    
    # 6. Plot and save feature importance for Random Forest (Primary)
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    
    plt.figure(figsize=(8, 5))
    feat_imp.plot(kind="barh", color="teal")
    plt.title("Random Forest (Primary Model) Feature Importance")
    plt.xlabel("Importance score")
    plt.tight_layout()
    
    feat_imp_file = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(feat_imp_file, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved to {feat_imp_file}")

if __name__ == "__main__":
    train_aqi_model()
