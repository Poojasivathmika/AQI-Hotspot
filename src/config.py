import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Geographic bounds for India
# [minLon, minLat, maxLon, maxLat]
INDIA_BBOX = [68.0, 6.0, 98.0, 38.0]

# Focus time window (Biomass burning season + baseline months)
DEFAULT_START_DATE = "2023-10-01"
DEFAULT_END_DATE = "2023-11-30"

# Paths
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Data files
OPENAQ_RAW_FILE = DATA_RAW_DIR / "openaq_india.csv"
SATELLITE_GRID_DIR = DATA_RAW_DIR / "satellite"
FIRMS_RAW_FILE = DATA_RAW_DIR / "firms_india.csv"

# Preprocessed & output files
MODEL_DATASET_FILE = DATA_PROCESSED_DIR / "model_dataset.csv"
PREDICTED_AQI_GRID_FILE = DATA_PROCESSED_DIR / "predicted_aqi_grid.csv"
HOTSPOTS_FILE = DATA_PROCESSED_DIR / "hotspots.csv"
HOTSPOT_TRENDS_FILE = DATA_PROCESSED_DIR / "hotspot_trends.csv"

# Model files
MODEL_FILE = MODELS_DIR / "aqi_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# API Keys and Environment settings
# Anthropic API key for AI summary page
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Ingestion configuration
GEE_DATA_SOURCE = os.environ.get("GEE_DATA_SOURCE", "offline").lower()
OPENAQ_DATA_SOURCE = os.environ.get("OPENAQ_DATA_SOURCE", "offline").lower()
FIRMS_DATA_SOURCE = os.environ.get("FIRMS_DATA_SOURCE", "offline").lower()

GEE_LOCAL_CSV_PATH = os.environ.get("GEE_LOCAL_CSV_PATH", "")
OPENAQ_LOCAL_CSV_PATH = os.environ.get("OPENAQ_LOCAL_CSV_PATH", "")
FIRMS_LOCAL_CSV_PATH = os.environ.get("FIRMS_LOCAL_CSV_PATH", "")

FALLBACK_TO_OFFLINE = os.environ.get("FALLBACK_TO_OFFLINE", "True").lower() == "true"
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")

# CPCB Sub-Index Breakpoint Tables
# Breakpoint concentration ranges and corresponding AQI sub-index ranges
# PM2.5, PM10, NO2, SO2 (24h average in ug/m3)
# CO (8h average in mg/m3)
# O3 (8h average in ug/m3)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        {"low": 0, "high": 30, "i_low": 0, "i_high": 50},
        {"low": 31, "high": 60, "i_low": 51, "i_high": 100},
        {"low": 61, "high": 90, "i_low": 101, "i_high": 200},
        {"low": 91, "high": 120, "i_low": 201, "i_high": 300},
        {"low": 121, "high": 250, "i_low": 301, "i_high": 400},
        {"low": 251, "high": 99999, "i_low": 401, "i_high": 500}
    ],
    "PM10": [
        {"low": 0, "high": 50, "i_low": 0, "i_high": 50},
        {"low": 51, "high": 100, "i_low": 51, "i_high": 100},
        {"low": 101, "high": 250, "i_low": 101, "i_high": 200},
        {"low": 251, "high": 350, "i_low": 201, "i_high": 300},
        {"low": 351, "high": 430, "i_low": 301, "i_high": 400},
        {"low": 431, "high": 99999, "i_low": 401, "i_high": 500}
    ],
    "NO2": [
        {"low": 0, "high": 40, "i_low": 0, "i_high": 50},
        {"low": 41, "high": 80, "i_low": 51, "i_high": 100},
        {"low": 81, "high": 180, "i_low": 101, "i_high": 200},
        {"low": 181, "high": 280, "i_low": 201, "i_high": 300},
        {"low": 281, "high": 400, "i_low": 301, "i_high": 400},
        {"low": 401, "high": 99999, "i_low": 401, "i_high": 500}
    ],
    "SO2": [
        {"low": 0, "high": 40, "i_low": 0, "i_high": 50},
        {"low": 41, "high": 80, "i_low": 51, "i_high": 100},
        {"low": 81, "high": 380, "i_low": 101, "i_high": 200},
        {"low": 381, "high": 800, "i_low": 201, "i_high": 300},
        {"low": 801, "high": 1600, "i_low": 301, "i_high": 400},
        {"low": 1601, "high": 99999, "i_low": 401, "i_high": 500}
    ],
    "CO": [
        {"low": 0.0, "high": 1.0, "i_low": 0, "i_high": 50},
        {"low": 1.1, "high": 2.0, "i_low": 51, "i_high": 100},
        {"low": 2.1, "high": 10.0, "i_low": 101, "i_high": 200},
        {"low": 10.1, "high": 17.0, "i_low": 201, "i_high": 300},
        {"low": 17.1, "high": 34.0, "i_low": 301, "i_high": 400},
        {"low": 34.1, "high": 99999.0, "i_low": 401, "i_high": 500}
    ],
    "O3": [
        {"low": 0, "high": 50, "i_low": 0, "i_high": 50},
        {"low": 51, "high": 100, "i_low": 51, "i_high": 100},
        {"low": 101, "high": 168, "i_low": 101, "i_high": 200},
        {"low": 169, "high": 208, "i_low": 201, "i_high": 300},
        {"low": 209, "high": 748, "i_low": 301, "i_high": 400},
        {"low": 749, "high": 99999, "i_low": 401, "i_high": 500}
    ]
}
