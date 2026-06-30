import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, 
  Legend as ChartLegend, ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { 
  Activity, Flame, AlertTriangle, ShieldAlert, Cpu, 
  Map as MapIcon, BarChart2, FileText, CheckSquare, Sparkles, RefreshCw
} from 'lucide-react';
import InteractiveMap from './components/InteractiveMap';

const AQI_RANGES = [
  { name: 'Good', min: 0, max: 50, color: '#2ecc71', text: '#0b0c10' },
  { name: 'Satisfactory', min: 51, max: 100, color: '#3498db', text: '#0b0c10' },
  { name: 'Moderate', min: 101, max: 200, color: '#f1c40f', text: '#0b0c10' },
  { name: 'Poor', min: 201, max: 300, color: '#e67e22', text: '#fff' },
  { name: 'Very Poor', min: 301, max: 400, color: '#e74c3c', text: '#fff' },
  { name: 'Severe', min: 401, max: 500, color: '#962d2d', text: '#fff' }
];

export default function App() {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [activeTab, setActiveTab] = useState('aqi'); // 'aqi', 'hotspots', 'trends'
  
  // Dashboard data for active date
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Trends data loaded once
  const [trendsData, setTrendsData] = useState(null);
  const [trendsLoading, setTrendsLoading] = useState(true);

  // AI insights data
  const [insights, setInsights] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);

  // Map layer controls
  const [showFires, setShowFires] = useState(true);
  const [showCentroids, setShowCentroids] = useState(true);

  const API_BASE = 'http://localhost:8000/api';

  // 1. Fetch available dates and trends on load
  useEffect(() => {
    fetch(`${API_BASE}/dates`)
      .then(res => res.json())
      .then(data => {
        if (data.success && data.dates.length > 0) {
          setDates(data.dates);
          // Set middle or first date as default
          const defaultDate = data.dates[Math.floor(data.dates.length / 2)] || data.dates[0];
          setSelectedDate(defaultDate);
        }
      })
      .catch(err => console.error("Error fetching dates:", err));

    fetch(`${API_BASE}/trends`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setTrendsData(data);
        }
        setTrendsLoading(false);
      })
      .catch(err => {
        console.error("Error fetching trends:", err);
        setTrendsLoading(false);
      });
  }, []);

  // 2. Fetch dashboard data when date changes
  useEffect(() => {
    if (!selectedDate) return;
    setLoading(true);
    
    fetch(`${API_BASE}/dashboard?date=${selectedDate}`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setDashboardData(data);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching dashboard data:", err);
        setLoading(false);
      });

    // Fetch AI insights
    setInsightsLoading(true);
    fetch(`${API_BASE}/insights?date=${selectedDate}`)
      .then(res => res.json())
      .then(data => {
        setInsights(data);
        setInsightsLoading(false);
      })
      .catch(err => {
        console.error("Error fetching insights:", err);
        setInsightsLoading(false);
      });
  }, [selectedDate]);

  // Handle Date slider change
  const handleSliderChange = (e) => {
    const idx = parseInt(e.target.value);
    if (dates[idx]) {
      setSelectedDate(dates[idx]);
    }
  };

  const getAqiColor = (aqi) => {
    const range = AQI_RANGES.find(r => aqi >= r.min && aqi <= r.max);
    return range ? range.color : '#962d2d';
  };

  const getAqiName = (aqi) => {
    const range = AQI_RANGES.find(r => aqi >= r.min && aqi <= r.max);
    return range ? range.name : 'Severe';
  };

  // Prepare top tables
  const getTopAqiCells = () => {
    if (!dashboardData || !dashboardData.grid) return [];
    return [...dashboardData.grid]
      .sort((a, b) => b.AQI - a.AQI)
      .slice(0, 5);
  };

  const getTopHchoCells = () => {
    if (!dashboardData || !dashboardData.grid) return [];
    return [...dashboardData.grid]
      .sort((a, b) => b.HCHO - a.HCHO)
      .slice(0, 5);
  };

  // Recharts Trends prep
  const getHotspotTrendsChart = () => {
    if (!trendsData || !trendsData.trends) return [];
    return trendsData.trends.map(t => ({
      date: t.date,
      'Hotspot Count': t.total_hotspots,
      'Fire-Linked Fraction (%)': Math.round(t.fire_associated_fraction * 100)
    }));
  };

  const getCityTimelinesChart = () => {
    if (!trendsData || !trendsData.city_timelines) return [];
    const delhi = trendsData.city_timelines['Delhi (NCR)'] || [];
    const mumbai = trendsData.city_timelines['Mumbai'] || [];
    const bengaluru = trendsData.city_timelines['Bengaluru'] || [];

    // align by date
    const chartData = delhi.map((d, i) => {
      const mVal = mumbai[i] ? mumbai[i].AQI : 0;
      const bVal = bengaluru[i] ? bengaluru[i].AQI : 0;
      return {
        date: d.date,
        'Delhi (NCR)': d.AQI,
        'Mumbai': mVal,
        'Bengaluru': bVal
      };
    });
    return chartData;
  };

  const getScatterAqiHcho = () => {
    if (!dashboardData || !dashboardData.grid) return [];
    // sample 200 grid points to avoid lagging Recharts scatter render
    const sampled = [];
    const step = Math.max(1, Math.floor(dashboardData.grid.length / 200));
    for (let i = 0; i < dashboardData.grid.length; i += step) {
      sampled.push({
        aqi: dashboardData.grid[i].AQI,
        hcho: parseFloat((dashboardData.grid[i].HCHO * 1e4).toFixed(3)),
        temp: dashboardData.grid[i].temp
      });
    }
    return sampled;
  };

  const selectedDateIndex = dates.indexOf(selectedDate);

  return (
    <div className="app-container">
      {/* 1. SIDEBAR PANELS */}
      <aside className="sidebar">
        <div className="logo-section">
          <img src="https://img.icons8.com/clouds/100/null/satellite-sending-signal.png" alt="Satellite" className="logo-img" />
          <div className="logo-text">
            <h1>ISRO Clean Air Link</h1>
            <p>AQI & Hotspot Telemetry Suite</p>
          </div>
        </div>

        <div className="sidebar-divider" />

        {/* Date Selector Slider */}
        <div className="control-label">Observation Epoch</div>
        <div className="date-display">{selectedDate || 'Loading...'}</div>
        
        {dates.length > 0 && (
          <div className="slider-container">
            <input 
              type="range" 
              min={0} 
              max={dates.length - 1} 
              value={selectedDateIndex >= 0 ? selectedDateIndex : 0} 
              onChange={handleSliderChange}
              className="slider-input" 
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              <span>{dates[0]}</span>
              <span>{dates[dates.length - 1]}</span>
            </div>
          </div>
        )}

        <div className="sidebar-divider" style={{ marginTop: 0 }} />

        {/* Daily summary metric widgets */}
        <div className="control-label">Daily Telemetry Summary</div>
        
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', margin: '20px 0' }}>
            <div className="spinner" />
          </div>
        ) : (
          <div className="sidebar-metrics">
            <div className="metric-card">
              <span className="metric-title">National Avg Predicted AQI</span>
              <span className="metric-value" style={{ color: getAqiColor(dashboardData?.metrics.mean_aqi) }}>
                {dashboardData?.metrics.mean_aqi.toFixed(1)}
              </span>
              <span style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                CPCB Scale: {getAqiName(dashboardData?.metrics.mean_aqi)}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-title">Peak predicted AQI</span>
              <span className="metric-value" style={{ color: getAqiColor(dashboardData?.metrics.max_aqi) }}>
                {dashboardData?.metrics.max_aqi.toFixed(1)}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-title">DBSCAN HCHO Hotspots</span>
              <span className="metric-value" style={{ color: 'var(--accent-purple)' }}>
                {dashboardData?.metrics.hotspot_count}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-title">NASA FIRMS Fires</span>
              <span className="metric-value" style={{ color: 'var(--accent-red)' }}>
                {dashboardData?.metrics.fire_count}
              </span>
            </div>
          </div>
        )}
      </aside>

      {/* 2. MAIN DASHBOARD CONTENT AREA */}
      <main className="main-content">
        <header className="header-panel">
          <div className="header-title">
            <h2>Spatial Air Quality Modeling & TROPOMI Hotspot Analytics</h2>
            <p>Powered by Random Forest Regressor & DBSCAN Clustering Fallback Pipeline</p>
          </div>

          <div className="tabs-container">
            <button 
              className={`tab-btn ${activeTab === 'aqi' ? 'active' : ''}`}
              onClick={() => setActiveTab('aqi')}
            >
              <MapIcon size={16} /> Predicted Surface AQI
            </button>
            <button 
              className={`tab-btn ${activeTab === 'hotspots' ? 'active' : ''}`}
              onClick={() => setActiveTab('hotspots')}
            >
              <Flame size={16} /> HCHO Hotspots & Fires
            </button>
            <button 
              className={`tab-btn ${activeTab === 'trends' ? 'active' : ''}`}
              onClick={() => setActiveTab('trends')}
            >
              <BarChart2 size={16} /> Multi-Temporal Trends
            </button>
          </div>
        </header>

        {/* LOADING SHIELD */}
        {loading && activeTab !== 'trends' ? (
          <div className="loading-container panel-card" style={{ minHeight: '520px' }}>
            <div className="spinner" />
            <span>Synchronizing satellite grids and fire points...</span>
          </div>
        ) : (
          <>
            {/* VIEW TAB 1: AQI GRID MAP */}
            {activeTab === 'aqi' && (
              <div className="grid-two-cols">
                <div className="panel-card">
                  <div className="panel-header">
                    <h3>🗺️ Surface AQI Heatmap Grid</h3>
                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>0.5° Resolution India Grid. Scroll to Zoom, Drag to Pan.</span>
                  </div>
                  
                  <InteractiveMap 
                    gridData={dashboardData?.grid} 
                    hotspots={dashboardData?.hotspots} 
                    fires={dashboardData?.fires}
                    showFires={false}
                    showCentroids={false}
                  />
                </div>

                {/* AQI scale and guidelines sidebar */}
                <div className="panel-card" style={{ justifyContent: 'space-between' }}>
                  <div>
                    <h3 style={{ fontSize: '14px', marginBottom: '12px', borderBottom: '1px solid var(--border-muted)', paddingBottom: '6px' }}>
                      CPCB AQI Scale
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {AQI_RANGES.map(range => (
                        <div 
                          key={range.name}
                          style={{
                            background: range.color,
                            color: range.text,
                            padding: '10px 14px',
                            borderRadius: '8px',
                            fontSize: '12px',
                            fontWeight: '700',
                            display: 'flex',
                            justifyContent: 'space-between'
                          }}
                        >
                          <span>{range.name}</span>
                          <span>{range.min} - {range.max}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '10px', fontSize: '12px', color: 'var(--text-secondary)', border: '1px dashed var(--border-muted)' }}>
                    <p style={{ fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>Model Interpretation</p>
                    Ground-level AQI predictions are output from a RandomForest model trained on OpenAQ ground sensors, Sentinel-5P tropospheric column densities (NO2, CO, Aerosol Index), and ERA5-Land meteorological aggregates (Temperature, RH, Wind Speed).
                  </div>
                </div>
              </div>
            )}

            {/* VIEW TAB 2: HCHO HOTSPOTS & OVERLAYS */}
            {activeTab === 'hotspots' && (
              <div className="grid-two-cols">
                <div className="panel-card" style={{ position: 'relative' }}>
                  <div className="panel-header">
                    <h3>🔥 HCHO Hotspots & Fires Overlay</h3>
                    <div style={{ display: 'flex', gap: '10px' }}>
                      <label className="checkbox-label">
                        <input 
                          type="checkbox" 
                          checked={showFires} 
                          onChange={(e) => setShowFires(e.target.checked)} 
                        />
                        Fires (NASA FIRMS)
                      </label>
                      <label className="checkbox-label">
                        <input 
                          type="checkbox" 
                          checked={showCentroids} 
                          onChange={(e) => setShowCentroids(e.target.checked)} 
                        />
                        Hotspots (DBSCAN)
                      </label>
                    </div>
                  </div>

                  <InteractiveMap 
                    gridData={dashboardData?.grid} 
                    hotspots={dashboardData?.hotspots} 
                    fires={dashboardData?.fires}
                    showFires={showFires}
                    showCentroids={showCentroids}
                  />

                  {/* Map Legend Overlay */}
                  <div className="map-legend">
                    <div className="legend-item">
                      <div className="legend-dot" style={{ background: '#ff3838' }} />
                      <span>Fires (NASA FIRMS)</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-dot" style={{ background: '#ff00ff' }} />
                      <span>Fire-Associated HCHO Centroid (&le;15km)</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-dot" style={{ background: '#00ffff' }} />
                      <span>Other HCHO Centroid (Industrial/Chemical)</span>
                    </div>
                  </div>
                </div>

                {/* HotspotCentroids Table */}
                <div className="panel-card">
                  <div className="panel-header">
                    <h3>Active DBSCAN HCHO Clusters</h3>
                    <span className="badge" style={{ background: 'var(--accent-purple)', color: '#000' }}>
                      {dashboardData?.hotspots.length} Clusters
                    </span>
                  </div>

                  <div className="table-container">
                    {dashboardData?.hotspots.length === 0 ? (
                      <div style={{ color: 'var(--text-secondary)', fontSize: '13px', textAlign: 'center', marginTop: '40px' }}>
                        No HCHO anomaly hotspots detected for this observation day.
                      </div>
                    ) : (
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Cluster ID</th>
                            <th>Centroid Location</th>
                            <th>Pixels</th>
                            <th>Avg HCHO</th>
                            <th>Source Type</th>
                          </tr>
                        </thead>
                        <tbody>
                          {dashboardData?.hotspots.map((hs, i) => (
                            <tr key={i}>
                              <td style={{ fontFamily: 'var(--mono)', fontWeight: '600' }}>{hs.cluster_id.split('_').slice(-1)[0]}</td>
                              <td>{hs.centroid_lat.toFixed(2)}°N, {hs.centroid_lon.toFixed(2)}°E</td>
                              <td style={{ fontFamily: 'var(--mono)' }}>{hs.n_pixels}</td>
                              <td style={{ fontFamily: 'var(--mono)' }}>{(hs.mean_hcho * 1e4).toFixed(2)}e-4</td>
                              <td>
                                <span 
                                  className="badge" 
                                  style={{ 
                                    background: hs.fire_associated ? 'rgba(255, 56, 56, 0.15)' : 'rgba(0, 242, 254, 0.15)',
                                    color: hs.fire_associated ? 'var(--accent-red)' : 'var(--accent-cyan)',
                                    border: `1px solid ${hs.fire_associated ? 'rgba(255, 56, 56, 0.3)' : 'rgba(0, 242, 254, 0.3)'}`
                                  }}
                                >
                                  {hs.fire_associated ? 'Biomass Burn' : 'Industrial'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* VIEW TAB 3: TEMPORAL TREND CHARTS */}
            {activeTab === 'trends' && (
              <>
                {trendsLoading ? (
                  <div className="loading-container panel-card" style={{ minHeight: '400px' }}>
                    <div className="spinner" />
                    <span>Analyzing historical trends...</span>
                  </div>
                ) : (
                  <div className="grid-trends">
                    {/* Trend 1: Hotspots and Fire Correlation */}
                    <div className="panel-card">
                      <div className="panel-header">
                        <h3>📈 DBSCAN Hotspots vs. Active Fire Correlation</h3>
                      </div>
                      <div style={{ width: '100%', height: 320 }}>
                        <ResponsiveContainer>
                          <LineChart data={getHotspotTrendsChart()} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="date" stroke="var(--text-secondary)" tick={{ fontSize: 10 }} />
                            <YAxis yAxisId="left" stroke="var(--accent-purple)" tick={{ fontSize: 11 }} label={{ value: 'Hotspot Clusters', angle: -90, position: 'insideLeft', style: { fill: 'var(--accent-purple)' } }} />
                            <YAxis yAxisId="right" orientation="right" stroke="var(--accent-orange)" tick={{ fontSize: 11 }} label={{ value: 'Fire-Linked Fraction (%)', angle: 90, position: 'insideRight', style: { fill: 'var(--accent-orange)' } }} />
                            <ChartTooltip contentStyle={{ background: 'var(--bg-surface)', borderColor: 'var(--border-muted)', color: '#fff' }} />
                            <ChartLegend />
                            <Line yAxisId="left" type="monotone" dataKey="Hotspot Count" stroke="var(--accent-purple)" strokeWidth={2} activeDot={{ r: 6 }} />
                            <Line yAxisId="right" type="monotone" dataKey="Fire-Linked Fraction (%)" stroke="var(--accent-orange)" strokeWidth={2} strokeDasharray="3 3" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Trend 2: City Timelines comparison */}
                    <div className="panel-card">
                      <div className="panel-header">
                        <h3>🏙️ Predicted Surface AQI Timeline (Selected Cities)</h3>
                      </div>
                      <div style={{ width: '100%', height: 320 }}>
                        <ResponsiveContainer>
                          <LineChart data={getCityTimelinesChart()} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="date" stroke="var(--text-secondary)" tick={{ fontSize: 10 }} />
                            <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 11 }} />
                            <ChartTooltip contentStyle={{ background: 'var(--bg-surface)', borderColor: 'var(--border-muted)', color: '#fff' }} />
                            <ChartLegend />
                            <Line type="monotone" dataKey="Delhi (NCR)" stroke="var(--accent-red)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="Mumbai" stroke="var(--accent-orange)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="Bengaluru" stroke="var(--aqi-good)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                )}

                <div className="panel-card" style={{ minHeight: '350px' }}>
                  <div className="panel-header">
                    <h3>📊 Parameter Correlation: AQI vs. HCHO Column Density (Current Epoch Grid Cells)</h3>
                  </div>
                  <div style={{ width: '100%', height: 280 }}>
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" dataKey="hcho" name="HCHO density" unit=" e-4 mol/m²" stroke="var(--text-secondary)" />
                        <YAxis type="number" dataKey="aqi" name="Predicted AQI" stroke="var(--text-secondary)" />
                        <ChartTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: 'var(--bg-surface)', borderColor: 'var(--border-muted)', color: '#fff' }} />
                        <Scatter name="Grid Points" data={getScatterAqiHcho()} fill="var(--accent-cyan)" opacity={0.6} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}
          </>
        )}

        {/* 3. AI EXECUTIVE SUMMARY BRIEFING BOX (Google Gemini API) */}
        <section className="ai-briefing-box">
          <div className="briefing-header">
            <Sparkles size={20} style={{ color: 'var(--accent-cyan)' }} />
            <h3>AI Executive Intelligence Briefing (ISRO Decision Support)</h3>
            {insights?.is_fallback && (
              <span className="badge" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-secondary)', border: '1px solid var(--border-muted)', marginLeft: '10px' }}>
                Local Rules Engine
              </span>
            )}
          </div>
          
          {insightsLoading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '14px', minHeight: '120px' }}>
              <div className="spinner" />
              <span>Consulting Gemini LLM and synthesizing atmospheric telemetry...</span>
            </div>
          ) : (
            <>
              <p className="briefing-text">{insights?.summary || 'No AI summary generated.'}</p>
              
              <div className="advisories-grid">
                {insights?.advisories && insights.advisories.map((adv, idx) => {
                  const badges = [
                    { type: 'danger', label: 'Particulate Hazard' },
                    { type: 'warning', label: 'Exposure Alert' },
                    { type: 'info', label: 'Policy Directive' }
                  ];
                  const badge = badges[idx] || { type: 'info', label: 'Advisory' };
                  
                  return (
                    <div key={idx} className="advisory-card">
                      <span className={`advisory-badge ${badge.type}`}>{badge.label}</span>
                      <span className="advisory-desc">{adv}</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </section>

        {/* 4. KEY SPATIAL TABULAR AGGREGATIONS */}
        <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          {/* Top AQI Cells */}
          <div className="panel-card">
            <div className="panel-header">
              <h3>🚨 Top 5 predicted AQI locations</h3>
            </div>
            <div className="table-container" style={{ maxHeight: '180px' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Coordinates</th>
                    <th>Predicted AQI</th>
                    <th>CPCB Category</th>
                    <th>HCHO</th>
                    <th>Nearby Fires</th>
                  </tr>
                </thead>
                <tbody>
                  {getTopAqiCells().map((cell, i) => (
                    <tr key={i}>
                      <td>{cell.lat.toFixed(2)}°N, {cell.lon.toFixed(2)}°E</td>
                      <td style={{ fontFamily: 'var(--mono)', fontWeight: '700', color: getAqiColor(cell.AQI) }}>{cell.AQI.toFixed(1)}</td>
                      <td>
                        <span className="badge" style={{ background: getAqiColor(cell.AQI), color: AQI_RANGES.find(r => cell.AQI >= r.min && cell.AQI <= r.max)?.text || '#fff' }}>
                          {cell.AQI_Category}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{(cell.HCHO * 1e4).toFixed(2)}e-4</td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{cell.fire_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Top HCHO Cells */}
          <div className="panel-card">
            <div className="panel-header">
              <h3>🌾 Top 5 TROPOMI HCHO Concentration Cells</h3>
            </div>
            <div className="table-container" style={{ maxHeight: '180px' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Coordinates</th>
                    <th>HCHO Density</th>
                    <th>Aerosol Index</th>
                    <th>Temp</th>
                    <th>Nearby Fires</th>
                  </tr>
                </thead>
                <tbody>
                  {getTopHchoCells().map((cell, i) => (
                    <tr key={i}>
                      <td>{cell.lat.toFixed(2)}°N, {cell.lon.toFixed(2)}°E</td>
                      <td style={{ fontFamily: 'var(--mono)', fontWeight: '700', color: 'var(--accent-cyan)' }}>{(cell.HCHO * 1e4).toFixed(3)}e-4</td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{cell.aerosol_index.toFixed(2)}</td>
                      <td>{cell.temp}°C</td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{cell.fire_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
