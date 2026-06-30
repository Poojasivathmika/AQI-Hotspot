import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    CPCB_BREAKPOINTS, OPENAQ_RAW_FILE, SATELLITE_GRID_DIR, FIRMS_RAW_FILE,
    MODEL_DATASET_FILE, SCALER_FILE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calculate_sub_index(param: str, val: float) -> float:
    """
    Computes CPCB sub-index for a given pollutant and concentration value
    using the linear interpolation formula.
    """
    if pd.isna(val) or val < 0:
        return np.nan
        
    # Map parameter names from OpenAQ format to config format
    mapping = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "co": "CO",
        "o3": "O3"
    }
    
    cfg_param = mapping.get(param.lower())
    if not cfg_param or cfg_param not in CPCB_BREAKPOINTS:
        return np.nan
        
    bp_list = CPCB_BREAKPOINTS[cfg_param]
    
    for bp in bp_list:
        if bp["low"] <= val <= bp["high"]:
            # CPCB sub-index linear interpolation formula:
            # I = [(I_hi - I_lo)/(B_hi - B_lo)] * (Cp - B_lo) + I_lo
            sub = ((bp["i_high"] - bp["i_low"]) / (bp["high"] - bp["low"])) * (val - bp["low"]) + bp["i_low"]
            return round(sub)
            
    # Cap at 500 for extremely high values exceeding the severe band
    return 500.0

def calculate_cpcb_aqi(row) -> float:
    """
    Calculates overall CPCB AQI from sub-indices.
    Requires at least 3 pollutants, with at least one being PM2.5 or PM10.
    """
    sub_indices = {}
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    
    for p in pollutants:
        val = row.get(p)
        if not pd.isna(val):
            sub = calculate_sub_index(p, val)
            if not pd.isna(sub):
                sub_indices[p] = sub
                
    # Check CPCB validity conditions
    if len(sub_indices) < 3:
        return np.nan
        
    if "pm25" not in sub_indices and "pm10" not in sub_indices:
        return np.nan
        
    return float(max(sub_indices.values()))

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates Haversine distance in kilometers.
    Can accept arrays.
    """
    R = 6371.0 # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def lonlat_to_xyz(lon, lat):
    """
    Converts lon/lat degrees to 3D Cartesian coordinates (x, y, z) on a sphere.
    Used for exact spatial range queries with KDTree.
    """
    R = 6371.0 # Earth radius in km
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))

def load_and_concat_satellite_grids():
    """
    Loads all daily satellite grids and concatenates them.
    """
    grid_files = glob.glob(os.path.join(SATELLITE_GRID_DIR, "satellite_grid_*.csv"))
    if not grid_files:
        raise FileNotFoundError(f"No satellite grid files found in {SATELLITE_GRID_DIR}")
        
    logger.info(f"Loading and concatenating {len(grid_files)} daily satellite grid files...")
    dfs = [pd.read_csv(f) for f in grid_files]
    grid_df = pd.concat(dfs, ignore_index=True)
    return grid_df

def preprocess_pipeline():
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Load and process OpenAQ station measurements
    if not os.path.exists(OPENAQ_RAW_FILE):
        raise FileNotFoundError(f"OpenAQ raw data not found at {OPENAQ_RAW_FILE}. Run ingestion first.")
        
    openaq_df = pd.read_csv(OPENAQ_RAW_FILE)
    logger.info(f"Loaded {len(openaq_df)} raw OpenAQ records.")
    
    # Pivot to get pollutants as columns per station-date
    station_daily = openaq_df.pivot_table(
        index=["station_id", "lat", "lon", "date"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()
    
    # Calculate AQI
    station_daily["AQI"] = station_daily.apply(calculate_cpcb_aqi, axis=1)
    
    # Drop rows without valid AQI
    valid_aqi_df = station_daily.dropna(subset=["AQI"]).copy()
    logger.info(f"Calculated CPCB AQI. Station-days with valid AQI: {len(valid_aqi_df)} (Dropped {len(station_daily) - len(valid_aqi_df)} invalid).")
    
    # 2. Load Satellite Grid data
    grid_df = load_and_concat_satellite_grids()
    logger.info(f"Concatenated satellite grid: {len(grid_df)} total grid cells across days.")
    
    # Get unique grid points to map stations
    unique_grids = grid_df[["lat", "lon"]].drop_duplicates().values # array of [lat, lon]
    unique_stations = valid_aqi_df[["station_id", "lat", "lon"]].drop_duplicates()
    
    # Map stations to nearest grid cells (Haversine distance <= 25km)
    station_to_grid = {}
    rejected_stations = 0
    
    for _, st in unique_stations.iterrows():
        # Compute distances to all grid cells
        dists = haversine_distance(st["lat"], st["lon"], unique_grids[:, 0], unique_grids[:, 1])
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist <= 25.0:
            station_to_grid[st["station_id"]] = {
                "grid_lat": unique_grids[min_idx, 0],
                "grid_lon": unique_grids[min_idx, 1],
                "distance_km": round(min_dist, 2)
            }
        else:
            rejected_stations += 1
            logger.warning(f"Station {st['station_id']} rejected: nearest grid cell is {min_dist:.1f}km away (> 25km limit).")
            
    logger.info(f"Mapped {len(station_to_grid)} stations to nearest grid cells. (Rejected: {rejected_stations}).")
    
    # Apply mapping back to AQI table
    valid_aqi_df["grid_lat"] = valid_aqi_df["station_id"].map(lambda sid: station_to_grid[sid]["grid_lat"] if sid in station_to_grid else np.nan)
    valid_aqi_df["grid_lon"] = valid_aqi_df["station_id"].map(lambda sid: station_to_grid[sid]["grid_lon"] if sid in station_to_grid else np.nan)
    valid_aqi_df = valid_aqi_df.dropna(subset=["grid_lat", "grid_lon"])
    
    # 3. Load FIRMS fires and calculate daily grid cell fire counts
    if not os.path.exists(FIRMS_RAW_FILE):
        raise FileNotFoundError(f"FIRMS raw file not found at {FIRMS_RAW_FILE}. Run ingestion first.")
        
    firms_df = pd.read_csv(FIRMS_RAW_FILE)
    logger.info(f"Loaded {len(firms_df)} active fire records from FIRMS.")
    
    # Group fires by date
    fires_by_date = {date: group for date, group in firms_df.groupby("acq_date")}
    
    # For each cell-day, count fires within 10km radius using KDTree
    logger.info("Running spatial range query for fire counts using KDTree (10km search radius)...")
    grid_df["fire_count"] = 0
    
    # Group grid by date to process efficiently
    grid_groups = []
    for date, group in grid_df.groupby("date"):
        if date in fires_by_date:
            fire_day = fires_by_date[date]
            # Convert fire points to Cartesian 3D
            fire_xyz = lonlat_to_xyz(fire_day["longitude"].values, fire_day["latitude"].values)
            # Convert grid points to Cartesian 3D
            grid_xyz = lonlat_to_xyz(group["lon"].values, group["lat"].values)
            
            # Build KDTree on fire points
            tree = cKDTree(fire_xyz)
            # Query count of fires within r = 10km (Cartesian distance matches geodesic distance closely at 10km)
            counts = [len(idx_list) for idx_list in tree.query_ball_point(grid_xyz, r=10.0)]
            group = group.copy()
            group["fire_count"] = counts
        else:
            group = group.copy()
            group["fire_count"] = 0
        grid_groups.append(group)
        
    grid_df = pd.concat(grid_groups, ignore_index=True)
    logger.info(f"Fire counts aggregated. Total fires linked to grid-days: {grid_df['fire_count'].sum()}")
    
    # Save the expanded grid DataFrame temporarily, it's used for predictions in Stage 4
    # Make sure target folder exists
    os.makedirs(os.path.dirname(MODEL_DATASET_FILE), exist_ok=True)
    
    # 4. Temporal and spatial merge
    # Join station AQI observations to grid cell observations on (date, lat, lon)
    # Merge using left_on and right_on to avoid renaming conflicts
    merged_df = pd.merge(
        valid_aqi_df[["date", "grid_lat", "grid_lon", "AQI"]],
        grid_df,
        left_on=["date", "grid_lat", "grid_lon"],
        right_on=["date", "lat", "lon"],
        how="inner"
    )
    # Drop the redundant grid_lat/grid_lon columns
    merged_df = merged_df.drop(columns=["grid_lat", "grid_lon"])
    logger.info(f"Merged modeling dataset size: {len(merged_df)} samples.")
    
    if len(merged_df) == 0:
        raise ValueError("Merged modeling dataset is empty! Please check coordinates overlapping between stations and GEE grid.")
        
    # 5. Handle missing values
    features = ["HCHO", "NO2", "CO", "aerosol_index", "temp", "RH", "wind_speed", "fire_count"]
    
    # Drop rows where > 50% of features are missing
    threshold = len(features) / 2
    missing_counts = merged_df[features].isna().sum(axis=1)
    rows_to_drop = missing_counts > threshold
    dropped_count = rows_to_drop.sum()
    merged_df = merged_df[~rows_to_drop].copy()
    
    logger.info(f"Dropped {dropped_count} rows with >50% missing features.")
    
    # Median impute remaining gaps
    imputed_count = 0
    for col in features:
        nas = merged_df[col].isna()
        if nas.any():
            # Median per coordinate to preserve spatial structure
            medians = merged_df.groupby(["lat", "lon"])[col].transform("median")
            # If some coords have all NaNs, fallback to global median
            global_median = merged_df[col].median()
            medians = medians.fillna(global_median)
            
            merged_df.loc[nas, col] = medians[nas]
            imputed_count += nas.sum()
            
    logger.info(f"Median imputed {imputed_count} individual feature values.")
    
    # Save the processed dataset
    merged_df.to_csv(MODEL_DATASET_FILE, index=False)
    logger.info(f"Preprocessed dataset saved to {MODEL_DATASET_FILE} (Shape: {merged_df.shape})")
    
    # Save grid features too (we need this to apply prediction to full grid later)
    # We save this as data/processed/full_grid_features.csv
    full_grid_features_file = os.path.dirname(MODEL_DATASET_FILE) + "/full_grid_features.csv"
    grid_df.to_csv(full_grid_features_file, index=False)
    logger.info(f"Full grid features saved to {full_grid_features_file} (Shape: {grid_df.shape})")

if __name__ == "__main__":
    preprocess_pipeline()
