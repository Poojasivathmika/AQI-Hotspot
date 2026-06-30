import os
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    FIRMS_RAW_FILE, DEFAULT_START_DATE, DEFAULT_END_DATE, INDIA_BBOX,
    FIRMS_DATA_SOURCE, FIRMS_LOCAL_CSV_PATH, FALLBACK_TO_OFFLINE, FIRMS_MAP_KEY
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_mock_firms_data(start_date, end_date):
    """
    Generates realistic synthetic active fire points (VIIRS/MODIS-like format) for India.
    Spikes fire counts significantly in Punjab, Haryana, and Western UP from mid-October to mid-November.
    """
    logger.info("Generating mock active fire points (NASA FIRMS format) over India...")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
    
    records = []
    np.random.seed(99) # Fixed seed
    
    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_year = date_obj.timetuple().tm_yday
        
        # Crop burning seasonal factor (peaks Nov 5, Day 309)
        peak_day = 309
        burning_width = 10
        burning_factor = np.exp(-((day_of_year - peak_day) ** 2) / (2 * (burning_width ** 2)))
        
        # 1. Background fires (low density all over India BBox)
        num_bg_fires = np.random.randint(15, 45)
        for _ in range(num_bg_fires):
            # Sample coordinate within India bbox, avoiding Himalayas and oceans if possible
            lat = np.random.uniform(10.0, 26.0)
            lon = np.random.uniform(73.0, 86.0)
            confidence = np.random.uniform(50, 95)
            frp = np.random.uniform(2.0, 45.0)
            
            records.append({
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "acq_date": date_str,
                "confidence": round(confidence, 1),
                "frp": round(frp, 1)
            })
            
        # 2. Crop burning fires in Punjab/Haryana/West UP
        # Punjab/Haryana bounding box: Lat [29.2, 32.5], Lon [74.2, 77.0]
        # Number of agricultural fires: ranges from 0-10 per day off-season, to 150-600 per day at peak!
        num_crop_fires = int(np.random.uniform(5, 20) + burning_factor * np.random.uniform(250, 600))
        
        # Define multiple cluster centers representing Punjab/Haryana districts (Amritsar, Sangrur, Bhatinda, Kaithal)
        centers = [
            (31.4, 75.0), # Amritsar/Tarn Taran
            (30.2, 75.8), # Sangrur/Patiala
            (30.1, 75.0), # Bathinda/Mansa
            (29.8, 76.3)  # Kaithal/Karnal (Haryana)
        ]
        
        for _ in range(num_crop_fires):
            # Pick a center
            center_lat, center_lon = centers[np.random.choice(len(centers))]
            # Spread fires in a Gaussian cluster around center
            lat = np.random.normal(center_lat, 0.4)
            lon = np.random.normal(center_lon, 0.4)
            
            # Clamp to bbox
            lat = np.clip(lat, 29.0, 32.8)
            lon = np.clip(lon, 74.0, 77.5)
            
            # Fires in Punjab are intensive, high FRP
            confidence = np.random.uniform(70, 100)
            frp = np.random.uniform(10.0, 180.0) * (1.0 + burning_factor * 0.5)
            
            records.append({
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "acq_date": date_str,
                "confidence": round(confidence, 1),
                "frp": round(frp, 1)
            })
            
    df = pd.DataFrame(records)
    return df

def load_local_csv_firms(path):
    """
    Copies a pre-exported NASA FIRMS CSV file to the raw data location.
    """
    import shutil
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Local FIRMS CSV path '{path}' does not exist.")
    os.makedirs(os.path.dirname(FIRMS_RAW_FILE), exist_ok=True)
    shutil.copy2(path, FIRMS_RAW_FILE)
    logger.info(f"Copied local FIRMS data from {path} to {FIRMS_RAW_FILE}")
    return pd.read_csv(FIRMS_RAW_FILE)

def fetch_firms_api(start_date, end_date):
    """
    Retrieves data from NASA FIRMS API.
    """
    map_key = FIRMS_MAP_KEY
    if not map_key:
        raise ValueError("FIRMS_MAP_KEY is not set in environment or config.")
        
    logger.info("FIRMS API map key found. Attempting to fetch real active fire points...")
    # NASA FIRMS API endpoint
    # Format: https://firms.modaps.eosdis.nasa.gov/api/country/csv/[MAP_KEY]/[SOURCE]/[COUNTRY]/[DAY_RANGE]/[DATE]
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{map_key}/VIIRS_SNPP_NRT/IND/10/{start_date}"
    
    response = requests.get(url, timeout=20)
    if response.status_code == 200:
        # Check if response text is valid CSV
        if "latitude" in response.text and "acq_date" in response.text:
            os.makedirs(os.path.dirname(FIRMS_RAW_FILE), exist_ok=True)
            with open(FIRMS_RAW_FILE, 'w') as f:
                f.write(response.text)
            logger.info("Successfully downloaded and saved FIRMS data.")
            return pd.read_csv(FIRMS_RAW_FILE)
        else:
            raise ValueError(f"Downloaded FIRMS file format is invalid: {response.text[:100]}")
    else:
        raise ConnectionError(f"FIRMS API failed with code: {response.status_code}")

def fetch_firms_data(start_date, end_date):
    """
    Fetches FIRMS active fire data based on FIRMS_DATA_SOURCE mode.
    """
    if os.path.exists(FIRMS_RAW_FILE):
        logger.info(f"Raw FIRMS file already exists at {FIRMS_RAW_FILE}. Skipping fetch.")
        return pd.read_csv(FIRMS_RAW_FILE)

    logger.info(f"Fetching FIRMS active fires in '{FIRMS_DATA_SOURCE}' mode...")
    
    if FIRMS_DATA_SOURCE == "offline":
        df = generate_mock_firms_data(start_date, end_date)
        
    elif FIRMS_DATA_SOURCE == "local_csv":
        try:
            df = load_local_csv_firms(FIRMS_LOCAL_CSV_PATH)
        except Exception as e:
            logger.error(f"Failed to load FIRMS from local CSV path '{FIRMS_LOCAL_CSV_PATH}': {e}")
            if FALLBACK_TO_OFFLINE:
                logger.warning("FALLBACK_TO_OFFLINE is True. Falling back to offline generator.")
                df = generate_mock_firms_data(start_date, end_date)
            else:
                raise
                
    elif FIRMS_DATA_SOURCE == "api":
        try:
            df = fetch_firms_api(start_date, end_date)
        except Exception as e:
            logger.warning(f"FIRMS API fetch failed: {e}")
            if FALLBACK_TO_OFFLINE:
                logger.warning("FALLBACK_TO_OFFLINE is True. Falling back to offline generator.")
                df = generate_mock_firms_data(start_date, end_date)
            else:
                raise e
    else:
        logger.error(f"Invalid FIRMS_DATA_SOURCE mode: '{FIRMS_DATA_SOURCE}'. Defaulting to offline.")
        df = generate_mock_firms_data(start_date, end_date)
        
    os.makedirs(os.path.dirname(FIRMS_RAW_FILE), exist_ok=True)
    df.to_csv(FIRMS_RAW_FILE, index=False)
    logger.info(f"Saved {len(df)} active fire records to {FIRMS_RAW_FILE}")
    return df

if __name__ == "__main__":
    fetch_firms_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
