# ISRO Clean Air Link: Air Quality & HCHO Hotspot Intelligence Platform

A high-fidelity spatial air quality modeling and formaldehyde (HCHO) hotspot intelligence platform. Developed for environmental surveillance and policy action during biomass burning seasons.

This application is built on a **FastAPI backend** and a **React SPA frontend** using Vite and HTML5 Canvas mapping.

---

## Repository Structure

```
/backend
  main.py                     - FastAPI REST backend serving predictions and AI summaries
/data
  /raw                        - Raw OpenAQ, Earth Engine, and FIRMS files (simulated/cached)
  /processed                  - Tabular model training sets, predicted grid maps, and clusters
/frontend
  /src
    /components
      InteractiveMap.jsx      - Canvas-based interactive GIS mapping component
    App.jsx                   - Core dashboard layout, date selectors, Recharts trends
    index.css                 - Premium dark-theme custom design system
/models                       - Trained RandomForest model & StandardScaler serialization
/reports                      - Model comparison metrics, evaluations, and feature plots
/src                          - Pipeline modular scripts (ingest, preprocess, model, hotspot)
.env                          - Local configuration parameters and API keys
run_pipeline.py               - End-to-end Python pipeline orchestrator
run_dev.py                    - Concurrent launcher for backend + frontend dev servers
requirements.txt              - Python package dependencies
```

---

## Core Technologies & Algorithm

1. **Modular Ingestion Layer**: Fetches OpenAQ station datasets, NASA FIRMS fire logs, and Sentinel-5P columns. Each module supports `api`, `offline` fallback, and `local_csv` loading modes via `.env`.
2. **Machine Learning Model**: Uses **Random Forest Regressor** as the primary predictive engine, trained on ERA5 meteorology and TROPOMI column densities to predict ground-level CPCB AQI. Keeps **XGBoost** as a secondary performance benchmark.
3. **Hotspot Detection**: Performs **DBSCAN spatial clustering** on the 95th percentile highest HCHO cells, then cross-references NASA FIRMS thermal anomalies (within 15km) to identify fire-associated plumes vs. industrial emissions.
4. **AI Summary Generation**: Integrates the **Google Gemini API** (using `requests` directly) to generate public health summaries and policy advisories, falling back to a rule-based expert engine if API keys are absent.

---

## Setup & Running Guide

### 1. Prerequisite Installations
Ensure you have Python 3.10+ and Node.js (with npm) installed.

### 2. Install Backend Dependencies
From the repository root, install the Python requirements:
```bash
pip install -r requirements.txt fastapi uvicorn
```

### 3. Install Frontend Dependencies
Navigate to the `frontend` folder and install NPM packages:
```bash
cd frontend
npm install
cd ..
```

### 4. Configuration (`.env`)
Create or edit `.env` in the root folder:
```ini
# Data modes: 'offline', 'api', or 'local_csv'
GEE_DATA_SOURCE=offline
OPENAQ_DATA_SOURCE=offline
FIRMS_DATA_SOURCE=offline

# API Keys (Provide your Gemini key for AI insights)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Execute Data Pipeline
Run the script to execute ingestion, preprocessing, Random Forest training, grid prediction, and hotspot clustering:
```bash
python run_pipeline.py
```

### 6. Run the Dashboard Application
Start the FastAPI backend and React frontend concurrently in development mode:
```bash
python run_dev.py
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser to explore the dashboard.

---

## API Endpoints (FastAPI)

- `GET /api/dates`: Returns a list of dates available in the predictions.
- `GET /api/dashboard?date=YYYY-MM-DD`: Returns metrics, predicted grid, hotspots, and fire points.
- `GET /api/trends`: Returns historical daily hotspot counts and city comparisons.
- `GET /api/insights?date=YYYY-MM-DD`: Fetches Gemini AI briefings.
