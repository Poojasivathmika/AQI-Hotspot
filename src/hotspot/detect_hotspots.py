import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import PREDICTED_AQI_GRID_FILE, FIRMS_RAW_FILE, HOTSPOTS_FILE, HOTSPOT_TRENDS_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Haversine distance calculation for manual distance checking
from src.preprocess.preprocess_data import haversine_distance

def detect_hcho_hotspots():
    logger.info("Starting HCHO Hotspot Detection pipeline...")
    
    # 1. Load data
    if not os.path.exists(PREDICTED_AQI_GRID_FILE):
        raise FileNotFoundError(f"Predicted AQI grid not found at {PREDICTED_AQI_GRID_FILE}. Run predict_grid.py first.")
    if not os.path.exists(FIRMS_RAW_FILE):
        raise FileNotFoundError(f"FIRMS active fires file not found at {FIRMS_RAW_FILE}. Run ingestion first.")
        
    grid_df = pd.read_csv(PREDICTED_AQI_GRID_FILE)
    firms_df = pd.read_csv(FIRMS_RAW_FILE)
    
    logger.info(f"Loaded predicted grid ({len(grid_df)} rows) and FIRMS active fires ({len(firms_df)} rows).")
    
    # Group fires by date for efficient lookup
    fires_by_date = {date: group for date, group in firms_df.groupby("acq_date")}
    
    hotspots_list = []
    daily_trends = []
    
    # Group grid by date
    grouped_grid = grid_df.groupby("date")
    
    for date, day_grid in grouped_grid:
        # 2. Daily thresholding: HCHO > 95th percentile
        hcho_vals = day_grid["HCHO"].dropna()
        if len(hcho_vals) == 0:
            continue
            
        threshold = hcho_vals.quantile(0.95)
        flagged_cells = day_grid[day_grid["HCHO"] > threshold].copy()
        
        if len(flagged_cells) < 3:
            # Not enough points for clustering
            daily_trends.append({
                "date": date,
                "total_hotspots": 0,
                "fire_associated_fraction": 0.0
            })
            continue
            
        # 3. DBSCAN clustering on lat/lon of flagged cells
        # eps = 0.25 degrees is ~27km, min_samples = 3
        coords = flagged_cells[["lat", "lon"]].values
        db = DBSCAN(eps=0.25, min_samples=3).fit(coords)
        labels = db.labels_
        
        flagged_cells["cluster_id"] = labels
        
        # Filter out noise points (label = -1)
        clustered_cells = flagged_cells[flagged_cells["cluster_id"] != -1]
        
        day_hotspots = []
        fire_associated_count = 0
        total_clusters = len(clustered_cells["cluster_id"].unique()) if len(clustered_cells) > 0 else 0
        
        if total_clusters > 0:
            day_fires = fires_by_date.get(date, pd.DataFrame())
            
            for cluster_id, group in clustered_cells.groupby("cluster_id"):
                centroid_lat = group["lat"].mean()
                centroid_lon = group["lon"].mean()
                mean_hcho = group["HCHO"].mean()
                n_pixels = len(group)
                
                # 4. Cross-reference with FIRMS active fires (distance <= 15km)
                is_fire_associated = 0
                if len(day_fires) > 0:
                    # Calculate distances from centroid to all fires on this day
                    dists = haversine_distance(
                        centroid_lat, centroid_lon,
                        day_fires["latitude"].values, day_fires["longitude"].values
                    )
                    if len(dists) > 0 and np.min(dists) <= 15.0:
                        is_fire_associated = 1
                        fire_associated_count += 1
                
                day_hotspots.append({
                    "date": date,
                    "cluster_id": f"{date}_C{cluster_id}",
                    "centroid_lat": round(centroid_lat, 4),
                    "centroid_lon": round(centroid_lon, 4),
                    "mean_hcho": float(mean_hcho),
                    "n_pixels": int(n_pixels),
                    "fire_associated": int(is_fire_associated)
                })
                
            hotspots_list.extend(day_hotspots)
            
        # Record daily trend statistics
        fire_fraction = fire_associated_count / total_clusters if total_clusters > 0 else 0.0
        daily_trends.append({
            "date": date,
            "total_hotspots": total_clusters,
            "fire_associated_fraction": round(fire_fraction, 3)
        })
        
    # 5. Save results to CSVs
    os.makedirs(os.path.dirname(HOTSPOTS_FILE), exist_ok=True)
    
    hotspots_df = pd.DataFrame(hotspots_list)
    trends_df = pd.DataFrame(daily_trends)
    
    hotspots_df.to_csv(HOTSPOTS_FILE, index=False)
    trends_df.to_csv(HOTSPOT_TRENDS_FILE, index=False)
    
    logger.info(f"Hotspot detection complete. Saved {len(hotspots_df)} clusters to {HOTSPOTS_FILE}")
    logger.info(f"Saved daily hotspot trends to {HOTSPOT_TRENDS_FILE}")
    
    # Log summary stats
    total_detected = len(hotspots_df)
    fire_linked = hotspots_df["fire_associated"].sum() if total_detected > 0 else 0
    logger.info(f"Total HCHO Hotspots detected: {total_detected}")
    if total_detected > 0:
        logger.info(f"Fire-associated Hotspots: {fire_linked} ({fire_linked/total_detected*100:.1f}% of total)")
    else:
        logger.info("Fire-associated Hotspots: 0 (0.0% of total)")

if __name__ == "__main__":
    detect_hcho_hotspots()
