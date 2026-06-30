import sys
import os
import time

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DEFAULT_START_DATE, DEFAULT_END_DATE
from src.ingest.openaq_fetch import fetch_openaq_data
from src.ingest.gee_fetch import fetch_gee_data
from src.ingest.firms_fetch import fetch_firms_data
from src.preprocess.preprocess_data import preprocess_pipeline
from src.model.train_model import train_aqi_model
from src.model.predict_grid import predict_full_grid
from src.hotspot.detect_hotspots import detect_hcho_hotspots

def run_pipeline():
    print("=" * 60)
    print("   INDIA AIR QUALITY & HCHO HOTSPOT PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Target analysis period: {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Stage 1: Data Ingestion
    print("\n--- STAGE 1: Data Ingestion (GEE, OpenAQ, NASA FIRMS) ---")
    try:
        print("[1/3] Running OpenAQ station measurement fetcher...")
        fetch_openaq_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        
        print("[2/3] Running Google Earth Engine satellite grid fetcher...")
        fetch_gee_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        
        print("[3/3] Running NASA FIRMS active fire fetcher...")
        fetch_firms_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        print("=> Stage 1 Ingestion completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error in Stage 1 (Ingestion): {e}")
        sys.exit(1)
        
    # Stage 2: Spatial-Temporal Preprocessing
    print("\n--- STAGE 2: Spatial-Temporal Preprocessing & Join ---")
    try:
        print("Running spatial KDTree queries, CPCB calculations, and imputation...")
        preprocess_pipeline()
        print("=> Stage 2 Preprocessing completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error in Stage 2 (Preprocessing): {e}")
        sys.exit(1)
        
    # Stage 3: Random Forest Model Training
    print("\n--- STAGE 3: Random Forest Model Training & Comparison ---")
    try:
        print("Training RandomForestRegressor and comparing performance with XGBoost...")
        train_aqi_model()
        print("=> Stage 3 Model Training completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error in Stage 3 (Model Training): {e}")
        sys.exit(1)
        
    # Stage 4: Full Grid Prediction
    print("\n--- STAGE 4: Full Grid predicted Surface AQI ---")
    try:
        print("Applying trained Random Forest model to predict AQI for India grid...")
        predict_full_grid()
        print("=> Stage 4 Grid Prediction completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error in Stage 4 (Grid Prediction): {e}")
        sys.exit(1)
        
    # Stage 5: DBSCAN Hotspot Detection
    print("\n--- STAGE 5: DBSCAN HCHO Hotspot Clustering & Fire Joins ---")
    try:
        print("Running DBSCAN spatial clustering and cross-referencing fire events...")
        detect_hcho_hotspots()
        print("=> Stage 5 Hotspot Detection completed successfully.")
    except Exception as e:
        print(f"[ERROR] Error in Stage 5 (Hotspot Detection): {e}")
        sys.exit(1)
        
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY IN {duration:.1f} SECONDS")
    print("=" * 60)
    print("All processed datasets are saved in /data/processed/ and model in /models/")
    print("You can now launch the React + FastAPI web application dashboard.")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
