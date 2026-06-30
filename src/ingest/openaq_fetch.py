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
    OPENAQ_RAW_FILE, DEFAULT_START_DATE, DEFAULT_END_DATE, INDIA_BBOX,
    OPENAQ_DATA_SOURCE, OPENAQ_LOCAL_CSV_PATH, FALLBACK_TO_OFFLINE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_mock_openaq_data(start_date, end_date):
    """
    Generates realistic, spatio-temporally consistent mock air quality data for India.
    Includes the October-November biomass burning seasonal spike in Northern India.
    """
    logger.info("Generating mock OpenAQ station data for India...")
    
    # Define key monitoring stations with coordinates
    stations = [
        {"station_id": "IN_Delhi_ITO", "city": "Delhi", "lat": 28.63, "lon": 77.24, "region": "North"},
        {"station_id": "IN_Amritsar_Civil", "city": "Amritsar", "lat": 31.63, "lon": 74.87, "region": "North"},
        {"station_id": "IN_Lucknow_Lalbagh", "city": "Lucknow", "lat": 26.85, "lon": 80.94, "region": "North"},
        {"station_id": "IN_Patna_EcoPark", "city": "Patna", "lat": 25.60, "lon": 85.12, "region": "North"},
        {"station_id": "IN_Mumbai_Bandra", "city": "Mumbai", "lat": 19.06, "lon": 72.82, "region": "West"},
        {"station_id": "IN_Bengaluru_City", "city": "Bengaluru", "lat": 12.97, "lon": 77.59, "region": "South"},
        {"station_id": "IN_Chennai_Alandur", "city": "Chennai", "lat": 13.00, "lon": 80.20, "region": "South"},
        {"station_id": "IN_Kolkata_Victoria", "city": "Kolkata", "lat": 22.54, "lon": 88.34, "region": "East"},
        {"station_id": "IN_Hyderabad_Sanathnagar", "city": "Hyderabad", "lat": 17.45, "lon": 78.43, "region": "South"}
    ]
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    
    dates = [ (start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1) ]
    
    pollutants = {
        "pm25": {"unit": "ug/m³", "base_low": 20, "base_high": 60},
        "pm10": {"unit": "ug/m³", "base_low": 40, "base_high": 110},
        "no2": {"unit": "ug/m³", "base_low": 15, "base_high": 35},
        "so2": {"unit": "ug/m³", "base_low": 5, "base_high": 15},
        "co": {"unit": "mg/m³", "base_low": 0.3, "base_high": 0.8},
        "o3": {"unit": "ug/m³", "base_low": 20, "base_high": 55}
    }
    
    records = []
    np.random.seed(42) # Consistent random generation
    
    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_year = date_obj.timetuple().tm_yday
        
        # Calculate burning season intensity (peaks around Oct 25 - Nov 15)
        # We model this using a Gaussian curve centered on day 310 (approx Nov 6)
        peak_day = 310
        width = 12
        burning_factor = np.exp(-((day_of_year - peak_day) ** 2) / (2 * (width ** 2)))
        
        for st_info in stations:
            is_northern = st_info["region"] == "North"
            
            # Weekly patterns (slightly lower on Sundays)
            weekday_mult = 0.9 if date_obj.weekday() == 6 else 1.0
            
            for param, p_info in pollutants.items():
                base = np.random.uniform(p_info["base_low"], p_info["base_high"])
                
                # Apply region-specific modifiers
                if is_northern:
                    # Northern cities generally have higher base levels
                    base *= 1.8
                    if param in ["pm25", "pm10", "co", "no2"]:
                        # Crop burning spike in North India
                        spike = burning_factor * 250.0 if param == "pm25" else burning_factor * 300.0
                        if param == "co":
                            spike = burning_factor * 3.5
                        if param == "no2":
                            spike = burning_factor * 45.0
                        base += spike
                elif st_info["city"] == "Mumbai" or st_info["city"] == "Kolkata":
                    # Coastal cities have moderate levels
                    base *= 1.2
                else:
                    # South/other cities have cleaner air
                    base *= 0.8
                
                # Add random walk noise
                noise = np.random.normal(0, base * 0.1)
                value = max(0.1, base * weekday_mult + noise)
                
                # Ensure CO is formatted reasonably (typically decimals in mg/m3)
                if param == "co":
                    value = round(value, 2)
                else:
                    value = round(value, 1)
                
                records.append({
                    "station_id": st_info["station_id"],
                    "lat": st_info["lat"],
                    "lon": st_info["lon"],
                    "date": date_str,
                    "parameter": param,
                    "value": value,
                    "unit": p_info["unit"]
                })
                
    df = pd.DataFrame(records)
    return df

def load_local_csv_openaq(path):
    """
    Copies a pre-exported OpenAQ CSV file to the raw data location.
    """
    import shutil
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Local OpenAQ CSV path '{path}' does not exist.")
    os.makedirs(os.path.dirname(OPENAQ_RAW_FILE), exist_ok=True)
    shutil.copy2(path, OPENAQ_RAW_FILE)
    logger.info(f"Copied local OpenAQ data from {path} to {OPENAQ_RAW_FILE}")
    return pd.read_csv(OPENAQ_RAW_FILE)

def fetch_openaq_api(start_date, end_date):
    """
    Retrieves data from OpenAQ API directly.
    """
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "country": "IN",
        "date_from": f"{start_date}T00:00:00Z",
        "date_to": f"{end_date}T23:59:59Z",
        "limit": 10000,
        "page": 1
    }
    
    headers = {
        "Accept": "application/json"
    }
    
    api_key = os.environ.get("OPENAQ_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
        
    all_data = []
    logger.info(f"Attempting to query OpenAQ API from {start_date} to {end_date}...")
    
    response = requests.get(url, params=params, headers=headers, timeout=15)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        
        if results:
            logger.info(f"Successfully retrieved {len(results)} records from OpenAQ API.")
            for r in results:
                lat = r.get("coordinates", {}).get("latitude")
                lon = r.get("coordinates", {}).get("longitude")
                if lat is None or lon is None:
                    continue
                    
                date_str = r.get("date", {}).get("utc", "").split("T")[0]
                
                all_data.append({
                    "station_id": r.get("locationId", r.get("location", "unknown")),
                    "lat": lat,
                    "lon": lon,
                    "date": date_str,
                    "parameter": r.get("parameter"),
                    "value": r.get("value"),
                    "unit": r.get("unit")
                })
            
            df = pd.DataFrame(all_data)
            if len(df) > 100:
                return df
            else:
                raise ValueError("OpenAQ API returned insufficient number of records.")
        else:
            raise ValueError("OpenAQ API returned empty results.")
    else:
        raise ConnectionError(f"OpenAQ API request failed with status code: {response.status_code}")

def fetch_openaq_data(start_date, end_date):
    """
    Fetches OpenAQ data based on OPENAQ_DATA_SOURCE mode.
    """
    if os.path.exists(OPENAQ_RAW_FILE):
        logger.info(f"Raw OpenAQ file already exists at {OPENAQ_RAW_FILE}. Skipping fetch.")
        return pd.read_csv(OPENAQ_RAW_FILE)

    logger.info(f"Fetching OpenAQ data in '{OPENAQ_DATA_SOURCE}' mode...")
    
    if OPENAQ_DATA_SOURCE == "offline":
        df = generate_mock_openaq_data(start_date, end_date)
        
    elif OPENAQ_DATA_SOURCE == "local_csv":
        try:
            df = load_local_csv_openaq(OPENAQ_LOCAL_CSV_PATH)
        except Exception as e:
            logger.error(f"Failed to load OpenAQ from local CSV path '{OPENAQ_LOCAL_CSV_PATH}': {e}")
            if FALLBACK_TO_OFFLINE:
                logger.warning("FALLBACK_TO_OFFLINE is True. Falling back to offline generator.")
                df = generate_mock_openaq_data(start_date, end_date)
            else:
                raise
                
    elif OPENAQ_DATA_SOURCE == "api":
        try:
            df = fetch_openaq_api(start_date, end_date)
        except Exception as e:
            logger.warning(f"OpenAQ API fetch failed: {e}")
            if FALLBACK_TO_OFFLINE:
                logger.warning("FALLBACK_TO_OFFLINE is True. Falling back to offline generator.")
                df = generate_mock_openaq_data(start_date, end_date)
            else:
                raise e
    else:
        logger.error(f"Invalid OPENAQ_DATA_SOURCE mode: '{OPENAQ_DATA_SOURCE}'. Defaulting to offline.")
        df = generate_mock_openaq_data(start_date, end_date)
        
    os.makedirs(os.path.dirname(OPENAQ_RAW_FILE), exist_ok=True)
    df.to_csv(OPENAQ_RAW_FILE, index=False)
    logger.info(f"Saved {len(df)} station measurement records to {OPENAQ_RAW_FILE}")
    return df

if __name__ == "__main__":
    fetch_openaq_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
