import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import MODEL_FILE, SCALER_FILE, PREDICTED_AQI_GRID_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Re-use category mapping helper
from src.model.train_model import get_aqi_category

def predict_full_grid():
    logger.info("Applying AQI model to predict surface AQI for full India grid...")
    
    # 1. Check model files and source file
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError("Model or scaler files not found in models/. Run train_model.py first.")
        
    full_grid_features_file = os.path.join(
        os.path.dirname(PREDICTED_AQI_GRID_FILE), "full_grid_features.csv"
    )
    if not os.path.exists(full_grid_features_file):
        raise FileNotFoundError(f"Full grid features not found at {full_grid_features_file}. Run preprocessing first.")
        
    # Load model and scaler
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
        
    # Load features
    grid_df = pd.read_csv(full_grid_features_file)
    logger.info(f"Loaded full grid features with {len(grid_df)} cell-days.")
    
    features = ["HCHO", "NO2", "CO", "aerosol_index", "temp", "RH", "wind_speed", "fire_count"]
    
    # Scale grid features
    X_grid = grid_df[features]
    
    # Handle NaNs in grid just in case (optional backup imputer)
    if X_grid.isna().any().any():
        logger.warning("Found NaN values in grid features. Filling with feature medians...")
        X_grid = X_grid.fillna(X_grid.median())
        
    X_grid_scaled = scaler.transform(X_grid)
    
    # 2. Predict AQI
    y_pred = model.predict(X_grid_scaled)
    # Clip predictions to standard CPCB range [0, 500]
    y_pred = np.clip(y_pred, 0, 500)
    
    # Add predictions back to dataframe
    grid_df["AQI"] = np.round(y_pred, 1)
    grid_df["AQI_Category"] = grid_df["AQI"].apply(get_aqi_category)
    
    # Save the output file
    os.makedirs(os.path.dirname(PREDICTED_AQI_GRID_FILE), exist_ok=True)
    grid_df.to_csv(PREDICTED_AQI_GRID_FILE, index=False)
    
    logger.info(f"Predicted AQI grid saved to {PREDICTED_AQI_GRID_FILE} (Shape: {grid_df.shape})")
    
    # Print high level summary stats
    mean_aqi = grid_df["AQI"].mean()
    category_counts = grid_df["AQI_Category"].value_counts(normalize=True) * 100
    logger.info(f"Mean predicted AQI across India grid: {mean_aqi:.2f}")
    logger.info("Predicted AQI Category distribution:")
    for cat, pct in category_counts.items():
        logger.info(f"  - {cat}: {pct:.1f}%")

if __name__ == "__main__":
    predict_full_grid()
