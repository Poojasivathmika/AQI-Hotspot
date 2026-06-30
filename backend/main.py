import os
import sys
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    PREDICTED_AQI_GRID_FILE, HOTSPOTS_FILE, HOTSPOT_TRENDS_FILE,
    FIRMS_RAW_FILE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="India Air Quality & HCHO Hotspot Intelligence API",
    description="FastAPI backend serving surface AQI predictions, TROPOMI HCHO hotspots, and NASA FIRMS active fires overlay.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev/hackathon purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_predicted_grid():
    if os.path.exists(PREDICTED_AQI_GRID_FILE):
        try:
            return pd.read_csv(PREDICTED_AQI_GRID_FILE)
        except Exception as e:
            logger.error(f"Error reading grid file: {e}")
    return None

def load_hotspots():
    if os.path.exists(HOTSPOTS_FILE):
        try:
            return pd.read_csv(HOTSPOTS_FILE)
        except Exception as e:
            logger.error(f"Error reading hotspots file: {e}")
    return None

def load_trends():
    if os.path.exists(HOTSPOT_TRENDS_FILE):
        try:
            return pd.read_csv(HOTSPOT_TRENDS_FILE)
        except Exception as e:
            logger.error(f"Error reading trends file: {e}")
    return None

def load_fires():
    if os.path.exists(FIRMS_RAW_FILE):
        try:
            return pd.read_csv(FIRMS_RAW_FILE)
        except Exception as e:
            logger.error(f"Error reading fires file: {e}")
    return None

@app.get("/api/health")
def health_check():
    """Simple API health check endpoint."""
    grid_exists = os.path.exists(PREDICTED_AQI_GRID_FILE)
    hotspots_exists = os.path.exists(HOTSPOTS_FILE)
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_status": {
            "predicted_grid_file": "available" if grid_exists else "missing",
            "hotspots_file": "available" if hotspots_exists else "missing"
        }
    }

@app.get("/api/dates")
def get_dates():
    """Retrieve all unique dates available in the predicted dataset."""
    grid_df = load_predicted_grid()
    if grid_df is None:
        raise HTTPException(status_code=404, detail="Predicted grid data file is missing. Run the pipeline first.")
    
    dates = sorted(grid_df["date"].dropna().unique().tolist())
    return {
        "success": True,
        "count": len(dates),
        "dates": dates
    }

@app.get("/api/dashboard")
def get_dashboard(date: str = Query(..., description="Date formatted as YYYY-MM-DD")):
    """
    Get metrics summary, spatial grid data, active fires, and hotspot centroids for a given date.
    """
    grid_df = load_predicted_grid()
    hotspots_df = load_hotspots()
    fires_df = load_fires()
    
    if grid_df is None:
        raise HTTPException(status_code=404, detail="Predicted grid dataset missing. Run pipeline first.")
        
    day_grid = grid_df[grid_df["date"] == date]
    if len(day_grid) == 0:
        raise HTTPException(status_code=404, detail=f"No grid prediction data found for date {date}")
        
    day_hotspots = hotspots_df[hotspots_df["date"] == date] if hotspots_df is not None and len(hotspots_df) > 0 else pd.DataFrame()
    day_fires = fires_df[fires_df["acq_date"] == date] if fires_df is not None and len(fires_df) > 0 else pd.DataFrame()
    
    # Calculate summary metrics
    mean_aqi = float(day_grid["AQI"].mean())
    max_aqi = float(day_grid["AQI"].max())
    hotspot_count = len(day_hotspots)
    fire_count = len(day_fires)
    
    # Clean NaN values for JSON serialization
    day_grid = day_grid.fillna(0)
    day_hotspots = day_hotspots.fillna(0)
    day_fires = day_fires.fillna(0)
    
    grid_records = day_grid.to_dict(orient="records")
    hotspot_records = day_hotspots.to_dict(orient="records")
    fire_records = day_fires.to_dict(orient="records")
    
    return {
        "success": True,
        "date": date,
        "metrics": {
            "mean_aqi": round(mean_aqi, 1),
            "max_aqi": round(max_aqi, 1),
            "hotspot_count": hotspot_count,
            "fire_count": fire_count
        },
        "grid": grid_records,
        "hotspots": hotspot_records,
        "fires": fire_records
    }

@app.get("/api/trends")
def get_trends():
    """
    Retrieve historical daily hotspot counts, fire correlations, and city timelines.
    """
    trends_df = load_trends()
    grid_df = load_predicted_grid()
    
    if grid_df is None:
        raise HTTPException(status_code=404, detail="Predicted grid dataset missing. Run pipeline first.")
        
    trends_records = []
    if trends_df is not None:
        trends_df = trends_df.fillna(0)
        trends_records = trends_df.to_dict(orient="records")
        
    # Regional timelines for comparison
    cities_coords = {
        "Delhi (NCR)": (28.6, 77.2),
        "Mumbai": (19.1, 72.8),
        "Bengaluru": (13.0, 77.6)
    }
    
    city_timelines = {}
    grid_uniq = grid_df[["lat", "lon"]].drop_duplicates()
    
    for name, (lat_c, lon_c) in cities_coords.items():
        dists = (grid_uniq["lat"] - lat_c)**2 + (grid_uniq["lon"] - lon_c)**2
        closest = grid_uniq.iloc[dists.idxmin()]
        
        city_df = grid_df[(grid_df["lat"] == closest["lat"]) & (grid_df["lon"] == closest["lon"])].sort_values("date")
        city_df = city_df.fillna(0)
        city_timelines[name] = city_df[["date", "AQI", "HCHO"]].to_dict(orient="records")
        
    return {
        "success": True,
        "trends": trends_records,
        "city_timelines": city_timelines
    }

@app.get("/api/insights")
def get_insights(date: str = Query(..., description="Date formatted as YYYY-MM-DD")):
    """
    Queries Google Gemini API or falls back to rule-based insights if key is missing/fails.
    """
    grid_df = load_predicted_grid()
    hotspots_df = load_hotspots()
    fires_df = load_fires()
    
    if grid_df is None:
        raise HTTPException(status_code=404, detail="Data not found.")
        
    day_grid = grid_df[grid_df["date"] == date]
    if len(day_grid) == 0:
        return {"summary": "No data available for the requested date.", "advisories": []}
        
    day_hotspots = hotspots_df[hotspots_df["date"] == date] if hotspots_df is not None and len(hotspots_df) > 0 else pd.DataFrame()
    day_fires = fires_df[fires_df["acq_date"] == date] if fires_df is not None and len(fires_df) > 0 else pd.DataFrame()
    
    mean_aqi = float(day_grid["AQI"].mean())
    max_aqi = float(day_grid["AQI"].max())
    hotspot_count = len(day_hotspots)
    fire_count = len(day_fires)
    
    # 1. Rule-based fallback generator (Premium local atmospheric intelligence)
    nat_cat = "Good"
    if mean_aqi > 50: nat_cat = "Satisfactory"
    if mean_aqi > 100: nat_cat = "Moderate"
    if mean_aqi > 200: nat_cat = "Poor"
    if mean_aqi > 300: nat_cat = "Very Poor"
    if mean_aqi > 400: nat_cat = "Severe"
    
    is_burning_peak = False
    try:
        dt_obj = datetime.strptime(date, "%Y-%m-%d")
        month_name = dt_obj.strftime("%B")
        day_num = dt_obj.day
        if month_name == "November" or (month_name == "October" and day_num >= 15):
            is_burning_peak = True
    except Exception as e:
        logger.error(f"Error parsing date: {e}")
        
    if is_burning_peak:
        fallback_summary = (
            f"On {date}, atmospheric monitoring shows highly elevated pollution levels across Northern India, "
            f"closely linked to seasonal agricultural stubble burning. The national average predicted surface AQI stands at "
            f"{mean_aqi:.1f} ({nat_cat}). High-density formaldehyde (HCHO) columns from Sentinel-5P TROPOMI were "
            f"strongly concentrated over the Punjab and Haryana agricultural belt, matching {fire_count} active thermal anomalies detected by NASA FIRMS. "
            f"Prevailing wind vectors are dispersing this dense smoke plume southeastward along the Indo-Gangetic Plain, triggering severe AQI spikes in downstream regions like Delhi NCR."
        )
        fallback_advisories = [
            "High Health Risk Warning: Residents of Punjab, Haryana, and Delhi-NCR are advised to avoid outdoor sports and strenuous activities due to hazardous particulate loading.",
            "Medical Action Required: Asthmatics, children, and elderly individuals in the Indo-Gangetic plain should wear N95 masks when outdoors and run indoor air purifiers.",
            "Regulatory Advisory: Agricultural authorities should deploy enforcement teams to crop residue zones, and NCR industries should execute stage-3 GRAP reduction protocols."
        ]
    else:
        fallback_summary = (
            f"On {date}, air quality metrics across India remained within a baseline envelope, reflecting lower "
            f"atmospheric loading. The national average predicted AQI is {mean_aqi:.1f} ({nat_cat}), with high AQI levels "
            f"confined primarily to dense urban centers. Formaldehyde (HCHO) column values are uniform with no signs of regional crop burning. "
            f"Only {fire_count} scattered thermal anomalies were detected by FIRMS, representing no major threat to regional air columns."
        )
        fallback_advisories = [
            "Satisfactory Air Quality: Most regions show satisfactory or moderate conditions; no special outdoor restrictions are required for the general population.",
            "Sensitive Group Advisory: Individuals with extreme respiratory sensitivities in heavy metropolitan hotspots (e.g. Mumbai, Kolkata) should monitor local station metrics.",
            "Routine Surveillance: Regulatory agencies can maintain normal monitoring intervals; crop fields show minimal anomaly activity."
        ]

    # 2. Try Gemini API
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        logger.info("GEMINI_API_KEY is not set. Using rule-based local fallback insights.")
        return {
            "summary": fallback_summary,
            "advisories": fallback_advisories,
            "is_fallback": True
        }
        
    logger.info("Attempting to fetch insights from Google Gemini API...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    prompt = f"""
    You are an atmospheric scientist and environmental policy advisor presenting to the ISRO (Indian Space Research Organisation) clean air committee.
    Write a plain-language summary (4-6 sentences) and 3 bullet-point health advisories based on these atmospheric telemetry statistics for date {date}:
    - National average AQI: {mean_aqi:.1f}
    - Max predicted AQI: {max_aqi:.1f}
    - NASA FIRMS Active fire spots: {fire_count}
    - DBSCAN HCHO Hotspot clusters: {hotspot_count}
    
    Format the output strictly as a JSON object with two keys:
    - 'summary': string (the paragraph summary)
    - 'advisories': array of strings (exactly 3 advisories, each 1-2 sentences)
    Do not include markdown code block tags like ```json or any other text before/after the JSON. Just return the raw JSON object.
    """
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
        if response.status_code == 200:
            res_data = response.json()
            text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            parsed = json.loads(text)
            return {
                "summary": parsed.get("summary", fallback_summary),
                "advisories": parsed.get("advisories", fallback_advisories),
                "is_fallback": False
            }
        else:
            logger.warning(f"Gemini API returned status code: {response.status_code}. Using fallback.")
    except Exception as e:
        logger.warning(f"Error querying Gemini API: {e}. Using fallback.")
        
    return {
        "summary": fallback_summary,
        "advisories": fallback_advisories,
        "is_fallback": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
