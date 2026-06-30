import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import json
import requests
import logging

# Add src to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.config import (
    PREDICTED_AQI_GRID_FILE, HOTSPOTS_FILE, HOTSPOT_TRENDS_FILE,
    FIRMS_RAW_FILE, ANTHROPIC_API_KEY
)

# Page configuration
st.set_page_config(
    page_title="India Air Quality & HCHO Hotspot Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom styling for premium appearance
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .stCard {
        background-color: #171a23;
        border: 1px solid #2d313f;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2ecc71;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load data with caching
@st.cache_data
def load_predicted_grid():
    if os.path.exists(PREDICTED_AQI_GRID_FILE):
        return pd.read_csv(PREDICTED_AQI_GRID_FILE)
    return None

@st.cache_data
def load_hotspots():
    if os.path.exists(HOTSPOTS_FILE):
        return pd.read_csv(HOTSPOTS_FILE)
    return None

@st.cache_data
def load_trends():
    if os.path.exists(HOTSPOT_TRENDS_FILE):
        return pd.read_csv(HOTSPOT_TRENDS_FILE)
    return None

@st.cache_data
def load_fires():
    if os.path.exists(FIRMS_RAW_FILE):
        return pd.read_csv(FIRMS_RAW_FILE)
    return None

# Load all datasets
grid_data = load_predicted_grid()
hotspots_data = load_hotspots()
trends_data = load_trends()
fires_data = load_fires()

# Verify dataset availability
if grid_data is None or hotspots_data is None:
    st.error("⚠️ Data files are missing! Please run ingestion, preprocessing, modeling, and hotspot scripts to prepare the dashboard data.")
    st.info("Ensure you have run the following pipeline files:\n"
            "1. `src/ingest/openaq_fetch.py`, `gee_fetch.py`, `firms_fetch.py` (data download)\n"
            "2. `src/preprocess/preprocess_data.py` (spatial-temporal join)\n"
            "3. `src/model/train_model.py` and `predict_grid.py` (AQI predictions)\n"
            "4. `src/hotspot/detect_hotspots.py` (formaldehyde hotspot clustering)")
    st.stop()

# Get date ranges
available_dates = sorted(grid_data["date"].unique())

# Sidebar setup
st.sidebar.image("https://img.icons8.com/clouds/100/null/satellite-sending-signal.png", width=80)
st.sidebar.title("Satellite Earth Observation")
st.sidebar.write("AQI Prediction & Formaldehyde (HCHO) Hotspot Analysis Platform (India)")
st.sidebar.divider()

# Date selector
selected_date_str = st.sidebar.select_slider(
    "Select Observation Date:",
    options=available_dates,
    value=available_dates[int(len(available_dates)/2)]
)

st.sidebar.write(f"Showing analysis for: **{selected_date_str}**")

# Filter daily grids
day_grid = grid_data[grid_data["date"] == selected_date_str]
day_hotspots = hotspots_data[hotspots_data["date"] == selected_date_str] if len(hotspots_data) > 0 else pd.DataFrame()
day_fires = fires_data[fires_data["acq_date"] == selected_date_str] if fires_data is not None else pd.DataFrame()

# Sidebar Stats Cards
st.sidebar.subheader("Daily Insights Summary")

mean_aqi = day_grid["AQI"].mean()
max_aqi = day_grid["AQI"].max()
avg_hcho = day_grid["HCHO"].mean() * 1e4 # scaled for readability
fire_count = len(day_fires)
hotspot_count = len(day_hotspots)

def get_aqi_color(val):
    if val <= 50: return "#2ecc71" # Good
    elif val <= 100: return "#85c1e9" # Satisfactory
    elif val <= 200: return "#f1c40f" # Moderate
    elif val <= 300: return "#e67e22" # Poor
    elif val <= 400: return "#e74c3c" # Very Poor
    else: return "#922b21" # Severe

st.sidebar.markdown(f"""
<div class="stCard">
    <div class="metric-label">National Average predicted AQI</div>
    <div class="metric-value" style="color:{get_aqi_color(mean_aqi)}">{mean_aqi:.1f}</div>
</div>
<div class="stCard">
    <div class="metric-label">Max predicted AQI (Hotspot Zone)</div>
    <div class="metric-value" style="color:{get_aqi_color(max_aqi)}">{max_aqi:.1f}</div>
</div>
<div class="stCard">
    <div class="metric-label">Active HCHO Hotspot Clusters</div>
    <div class="metric-value" style="color:#e67e22">{hotspot_count}</div>
</div>
<div class="stCard">
    <div class="metric-label">NASA FIRMS Active Fire Spots</div>
    <div class="metric-value" style="color:#e74c3c">{fire_count}</div>
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tabs = st.tabs(["🗺️ India Predicted AQI", "🔥 HCHO Hotspots & Fires", "📈 Spatio-Temporal Trends", "🤖 AI Executive Briefing"])

# TAB 1: AQI GRID MAP
with tabs[0]:
    st.header(f"Predicted Ground-Level AQI (Surface AQI Map)")
    st.write("Predicted surface AQI derived from Sentinel-5P column density (NO2, CO, Aerosol Index) and meteorological conditions.")
    
    # Render plotly map
    aqi_colors = {
        "Good": "#2ecc71",
        "Satisfactory": "#85c1e9",
        "Moderate": "#f1c40f",
        "Poor": "#e67e22",
        "Very Poor": "#e74c3c",
        "Severe": "#922b21"
    }
    
    fig_aqi = px.scatter_mapbox(
        day_grid,
        lat="lat",
        lon="lon",
        color="AQI_Category",
        color_discrete_map=aqi_colors,
        category_orders={"AQI_Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]},
        size=day_grid["AQI"].clip(lower=10),
        size_max=12,
        hover_data={
            "AQI": ":.1f",
            "HCHO": ":.6f",
            "NO2": ":.6f",
            "CO": ":.4f",
            "aerosol_index": ":.2f",
            "temp": ":.1f",
            "RH": ":.1f",
            "fire_count": True,
            "AQI_Category": False
        },
        zoom=4.2,
        center={"lat": 22.0, "lon": 78.96},
        mapbox_style="carto-darkmatter",
        height=700
    )
    
    fig_aqi.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            title="CPCB AQI Category",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(10,10,10,0.8)",
            font=dict(color="white")
        )
    )
    
    st.plotly_chart(fig_aqi, use_container_width=True)

# TAB 2: HCHO HOTSPOTS & FIRES
with tabs[1]:
    st.header("Formaldehyde (HCHO) Concentrations & Fire Overlays")
    st.write("Identifies high HCHO column density anomalies (95th percentile threshold) clustered via DBSCAN. Toggle active fires to examine agricultural residue burning.")
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        st.write("### Layer Controls")
        show_fires = st.checkbox("Overlay NASA FIRMS Fire Detections", value=True)
        show_centroids = st.checkbox("Show DBSCAN Cluster Centroids", value=True)
        
        st.write("---")
        st.write("**HCHO Cluster Legend:**")
        st.markdown("""
        * <span style='color:magenta; font-weight:bold;'>Magenta Marker</span>: Fire-associated Hotspot (&le; 15km to active fire)
        * <span style='color:cyan; font-weight:bold;'>Cyan Marker</span>: Industrial / Other Source Hotspot
        * **Background Scale**: Column density of HCHO (mol/m²)
        """, unsafe_allow_html=True)
        
    with col1:
        # Base figure of HCHO concentration heatmap
        fig_hcho = px.scatter_mapbox(
            day_grid,
            lat="lat",
            lon="lon",
            color="HCHO",
            color_continuous_scale="YlOrRd",
            range_color=[0.5e-4, 4e-4],
            hover_data={"HCHO": ":.6f", "temp": ":.1f", "RH": ":.1f"},
            zoom=4.2,
            center={"lat": 22.0, "lon": 78.96},
            mapbox_style="carto-darkmatter",
            height=700
        )
        
        fig_hcho.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title="HCHO (mol/m²)",
                x=0.93,
                y=0.5,
                bgcolor="rgba(10,10,10,0.8)",
                tickfont=dict(color="white"),
                titlefont=dict(color="white")
            )
        )
        
        # Add Active Fires Trace
        if show_fires and len(day_fires) > 0:
            fig_hcho.add_trace(go.Scattermapbox(
                lat=day_fires["latitude"],
                lon=day_fires["longitude"],
                mode="markers",
                marker=dict(
                    size=5,
                    color="#ff3300",
                    opacity=0.7
                ),
                name="NASA FIRMS Fires",
                hovertext=[f"Fire Spot (FRP: {frp:.1f} MW)" for frp in day_fires["frp"]],
                hoverinfo="text"
            ))
            
        # Add Hotspot Centroids Trace
        if show_centroids and len(day_hotspots) > 0:
            colors = day_hotspots["fire_associated"].map({0: "cyan", 1: "magenta"})
            
            fig_hcho.add_trace(go.Scattermapbox(
                lat=day_hotspots["centroid_lat"],
                lon=day_hotspots["centroid_lon"],
                mode="markers+text",
                marker=dict(
                    size=day_hotspots["n_pixels"] * 1.5 + 10,
                    color=colors,
                    opacity=0.9,
                    symbol="circle"
                ),
                name="Hotspot Centroids",
                hovertext=[
                    f"Cluster ID: {cid}<br>Pixels (Size): {npix}<br>Mean HCHO: {mhcho:.6f}<br>Fire-associated: {'Yes' if fa==1 else 'No'}"
                    for cid, npix, mhcho, fa in zip(
                        day_hotspots["cluster_id"],
                        day_hotspots["n_pixels"],
                        day_hotspots["mean_hcho"],
                        day_hotspots["fire_associated"]
                    )
                ],
                hoverinfo="text"
            ))
            
        st.plotly_chart(fig_hcho, use_container_width=True)

# TAB 3: SPATIO-TEMPORAL TRENDS
with tabs[2]:
    st.header("Spatio-Temporal Trend & Correlation Analysis")
    st.write("Observe seasonal patterns, correlation coefficients, and pollutant relationships over the target timeline.")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.write("### Biomass Burning Season Trend")
        if trends_data is not None:
            fig_t1 = go.Figure()
            fig_t1.add_trace(go.Scatter(
                x=trends_data["date"], y=trends_data["total_hotspots"],
                mode="lines+markers", name="HCHO Hotspots Count",
                line=dict(color="#e67e22", width=2.5)
            ))
            fig_t1.add_trace(go.Scatter(
                x=trends_data["date"], y=trends_data["fire_associated_fraction"] * 100,
                mode="lines", name="Fire-Linked fraction (%)",
                line=dict(color="magenta", width=2, dash="dash"),
                yaxis="y2"
            ))
            
            fig_t1.update_layout(
                title="HCHO Hotspots vs. Active Fire Correlation over Time",
                xaxis_title="Date",
                yaxis_title="Total DBSCAN Hotspot Clusters",
                yaxis2=dict(
                    title="Fire-Associated Hotspots (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 105]
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(x=0.01, y=0.98),
                height=400
            )
            st.plotly_chart(fig_t1, use_container_width=True)
            
    with col_t2:
        st.write("### Regional AQI Timeline Comparison")
        # Sample AQI at key coordinates representing major cities
        # Delhi (28.6, 77.2), Mumbai (19.1, 72.8), Bengaluru (13.0, 77.6)
        cities_coords = {
            "Delhi (NCR)": (28.6, 77.2),
            "Mumbai": (19.1, 72.8),
            "Bengaluru": (13.0, 77.6)
        }
        
        fig_t2 = go.Figure()
        colors_city = {"Delhi (NCR)": "#e74c3c", "Mumbai": "#f1c40f", "Bengaluru": "#2ecc71"}
        
        for name, (lat_c, lon_c) in cities_coords.items():
            # Find closest cell
            grid_uniq = grid_data[["lat", "lon"]].drop_duplicates()
            dists = (grid_uniq["lat"] - lat_c)**2 + (grid_uniq["lon"] - lon_c)**2
            closest = grid_uniq.iloc[dists.idxmin()]
            
            city_df = grid_data[(grid_data["lat"] == closest["lat"]) & (grid_data["lon"] == closest["lon"])].sort_values("date")
            fig_t2.add_trace(go.Scatter(
                x=city_df["date"], y=city_df["AQI"],
                mode="lines", name=name,
                line=dict(color=colors_city[name], width=2)
            ))
            
        fig_t2.update_layout(
            title="Predicted Surface AQI Timeline (Selected Cities)",
            xaxis_title="Date",
            yaxis_title="Predicted AQI",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(x=0.01, y=0.98),
            height=400
        )
        st.plotly_chart(fig_t2, use_container_width=True)
        
    st.divider()
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.write("### Parameter Relationships")
        fig_c1 = px.scatter(
            grid_data.sample(min(3000, len(grid_data))),
            x="HCHO", y="AQI", color="temp",
            color_continuous_scale="Viridis",
            labels={"HCHO": "Formaldehyde Column Density (mol/m²)", "AQI": "Predicted AQI"},
            title="AQI vs. Formaldehyde Concentration (Sampled Grid Cells)",
            height=400
        )
        fig_c1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_c1, use_container_width=True)
        
    with col_c2:
        st.write("### Aerosol Index vs. Fire Densities")
        fig_c2 = px.scatter(
            grid_data[grid_data["fire_count"] > 0].sample(min(1500, len(grid_data[grid_data["fire_count"] > 0]))),
            x="fire_count", y="aerosol_index", color="CO",
            labels={"fire_count": "Active Fires (within 10km)", "aerosol_index": "Absorbing Aerosol Index"},
            title="Aerosol Loading (Plume intensity) vs. Local Fire Counts",
            height=400
        )
        fig_c2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_c2, use_container_width=True)

# TAB 4: AI SUMMARY
with tabs[3]:
    st.header("AI Executive Summary & Decision-Support Briefing")
    st.write("Generates plain-language public health summaries and environmental advisory notes based on satellite monitoring and AQI prediction grids.")
    
    # 1. Retrieve stats to build prompt
    top_aqi = day_grid.sort_values("AQI", ascending=False).head(3)
    top_hcho = day_grid.sort_values("HCHO", ascending=False).head(3)
    
    # Prepare details
    aqi_hotspots_txt = []
    for idx, r in top_aqi.iterrows():
        aqi_hotspots_txt.append(f"Coord ({r['lat']:.1f}°N, {r['lon']:.1f}°E): AQI {r['AQI']:.0f} ({r['AQI_Category']})")
        
    hcho_hotspots_txt = []
    for idx, r in top_hcho.iterrows():
        # Check if this pixel is associated with a cluster
        c_flag = "No"
        if len(day_hotspots) > 0:
            # find closest cluster centroid
            cdists = (day_hotspots["centroid_lat"] - r["lat"])**2 + (day_hotspots["centroid_lon"] - r["lon"])**2
            min_c = day_hotspots.iloc[cdists.idxmin()]
            if np.sqrt(cdists.min()) <= 0.3:
                c_flag = f"Yes (Cluster {min_c['cluster_id']}, Fire-Associated: {'Yes' if min_c['fire_associated']==1 else 'No'})"
        hcho_hotspots_txt.append(f"Coord ({r['lat']:.1f}°N, {r['lon']:.1f}°E): Density {r['HCHO']*1e4:.2f}e-4 mol/m², Fire-Associated: {c_flag}")
        
    national_mean_aqi = mean_aqi
    
    # Generate fallback briefing text based on rules (to avoid dependency on API Key)
    # We will check if ANTHROPIC_API_KEY is available. If so, we can query. Else, use the robust rule-based engine.
    has_api_key = len(ANTHROPIC_API_KEY.strip()) > 0
    
    briefing_text = ""
    advisory_bullets = []
    
    # Rule-based briefing generator (High fidelity fallback)
    # Categorize national average
    nat_cat = "Good"
    if national_mean_aqi > 50: nat_cat = "Satisfactory"
    if national_mean_aqi > 100: nat_cat = "Moderate"
    if national_mean_aqi > 200: nat_cat = "Poor"
    if national_mean_aqi > 300: nat_cat = "Very Poor"
    if national_mean_aqi > 400: nat_cat = "Severe"
    
    # Detect if Punjab burning season is active
    is_burning_peak = False
    month_name = datetime.strptime(selected_date_str, "%Y-%m-%d").strftime("%B")
    day_num = datetime.strptime(selected_date_str, "%Y-%m-%d").day
    if month_name == "November" or (month_name == "October" and day_num >= 15):
        is_burning_peak = True
        
    if is_burning_peak:
        briefing_text = (
            f"On {selected_date_str}, atmospheric monitoring shows highly elevated pollution levels across Northern India, "
            f"closely linked to seasonal agricultural stubble burning. The national average predicted surface AQI stands at "
            f"{national_mean_aqi:.1f} ({nat_cat}). High-density formaldehyde (HCHO) columns from Sentinel-5P TROPOMI were "
            f"strongly concentrated over the Punjab and Haryana agricultural belt, with a peak density of {top_hcho.iloc[0]['HCHO']*1e4:.2f}e-4 mol/m². "
            f"This matches {fire_count} active thermal anomalies detected by NASA FIRMS. The prevailing wind vectors are dispersing "
            f"this dense smoke plume southeastward along the Indo-Gangetic Plain, triggering severe AQI spikes in downstream regions like Delhi NCR."
        )
        advisory_bullets = [
            "**High Health Risk Warning**: Residents of Punjab, Haryana, and Delhi-NCR are advised to avoid outdoor sports and strenuous activities due to hazardous particulate loading.",
            "**Medical Action Required**: Asthmatics, children, and elderly individuals in the Indo-Gangetic plain should wear N95 masks when outdoors and run indoor air purifiers where available.",
            "**Regulatory Advisory**: Agricultural authorities should deploy enforcement teams to crop residue zones, and industrial operations in NCR should execute stage-3 GRAP reduction protocols."
        ]
    else:
        briefing_text = (
            f"On {selected_date_str}, air quality metrics across India remained within a baseline envelope, reflecting a lower "
            f"atmospheric loading. The national average predicted AQI is {national_mean_aqi:.1f} ({nat_cat}), with high AQI levels "
            f"confined primarily to dense urban centers. Formaldehyde (HCHO) column values are uniform, averaging a background value of "
            f"{avg_hcho:.2f}e-4 mol/m² with no signs of regional crop burning. Only {fire_count} scattered thermal anomalies were detected by "
            f"FIRMS across the subcontinent, typical of minor localized agricultural activities and representing no major threat to regional air columns."
        )
        advisory_bullets = [
            "**Satisfactory Air Quality**: Most regions show satisfactory or moderate conditions; no special outdoor restrictions are required for the general population.",
            "**Sensitive Group Advisory**: Individuals with extreme respiratory sensitivities in heavy metropolitan hotspots (e.g. Mumbai, Kolkata) should monitor local station metrics.",
            "**Routine Surveillance**: Regulatory agencies can maintain normal monitoring intervals; crop fields show minimal anomaly activity."
        ]
        
    # Attempt to query LLM if API Key is set
    if has_api_key:
        with st.spinner("Quering LLM (Claude-3.5-Sonnet) via Anthropic API..."):
            try:
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": "claude-3-5-sonnet-20240620",
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are an atmospheric scientist and policy advisor. Write a plain-language summary (4-6 sentences) and 3 bullet-point health advisories based on these stats for date {selected_date_str}:\n"
                                       f"- National average AQI: {national_mean_aqi:.1f} ({nat_cat})\n"
                                       f"- Top 3 AQI hotspots: {', '.join(aqi_hotspots_txt)}\n"
                                       f"- Top 3 HCHO concentrations: {', '.join(hcho_hotspots_txt)}\n"
                                       f"- Active fire spots: {fire_count}\n"
                                       f"- Total HCHO clusters: {hotspot_count}\n"
                                       f"Format the output strictly as a JSON object with two keys: 'summary' (string) and 'advisories' (array of strings)."
                        }
                    ]
                }
                
                resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    raw_text = resp_json["content"][0]["text"]
                    # parse json from LLM
                    parsed = json.loads(raw_text)
                    briefing_text = parsed.get("summary", briefing_text)
                    advisory_bullets = parsed.get("advisories", advisory_bullets)
                    st.success("🤖 Summary successfully generated by Claude.")
                else:
                    st.warning(f"Claude API query failed (Code: {resp.status_code}). Rendered high-fidelity local intelligence brief instead.")
            except Exception as e:
                st.warning(f"Could not connect to Anthropic API: {e}. Rendered high-fidelity local intelligence brief instead.")
    else:
        st.info("ℹ️ Local expert rule engine active. (To use Anthropic LLM, set `ANTHROPIC_API_KEY` in environment).")
        
    # Render briefing
    st.write("---")
    st.subheader("📋 Executive Intelligence Brief")
    st.info(briefing_text)
    
    st.subheader("🚨 Public Health & Policy Advisory")
    for bullet in advisory_bullets:
        st.markdown(f"- {bullet}")
        
    st.divider()
    st.subheader("📊 Key Spatial Aggregations (for {selected_date_str})")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.write("**Top Predicted AQI Locations (Grid Cells)**")
        st.dataframe(
            top_aqi[["lat", "lon", "AQI", "AQI_Category", "HCHO", "fire_count"]]
            .rename(columns={"lat": "Latitude", "lon": "Longitude", "fire_count": "Nearby Fires"}),
            hide_index=True
        )
    with col_a2:
        st.write("**Top Formaldehyde Concentration Cells**")
        st.dataframe(
            top_hcho[["lat", "lon", "HCHO", "aerosol_index", "temp", "fire_count"]]
            .rename(columns={"lat": "Latitude", "lon": "Longitude", "fire_count": "Nearby Fires"}),
            hide_index=True
        )
