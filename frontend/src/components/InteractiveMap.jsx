import React, { useRef, useEffect, useState } from 'react';

// Bounding box for India analysis grid
const MIN_LON = 68.0;
const MAX_LON = 98.0;
const MIN_LAT = 6.0;
const MAX_LAT = 38.0;

const AQI_COLORS = {
  'Good': '#2ecc71',
  'Satisfactory': '#3498db',
  'Moderate': '#f1c40f',
  'Poor': '#e67e22',
  'Very Poor': '#e74c3c',
  'Severe': '#962d2d'
};

export default function InteractiveMap({ gridData, hotspots, fires, showFires, showCentroids }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Viewport states for Zoom and Pan
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Hover Tooltip state
  const [hoveredCell, setHoveredCell] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Handle Resize and Dimensions
  const [dims, setDims] = useState({ width: 600, height: 600 });

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        setDims({
          width: clientWidth || 600,
          height: clientHeight || 550
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Projection utilities
  const toCanvasX = (lon) => {
    return ((lon - MIN_LON) / (MAX_LON - MIN_LON)) * dims.width;
  };

  const toCanvasY = (lat) => {
    return dims.height - ((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * dims.height;
  };

  const fromCanvasX = (x) => {
    return ((x - pan.x) / (dims.width * zoom)) * (MAX_LON - MIN_LON) + MIN_LON;
  };

  const fromCanvasY = (y) => {
    return ((dims.height - (y - pan.y)) / (dims.height * zoom)) * (MAX_LAT - MIN_LAT) + MIN_LAT;
  };

  // Re-draw Canvas on any state or data change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear and fill background
    ctx.clearRect(0, 0, dims.width, dims.height);
    ctx.fillStyle = '#07080c';
    ctx.fillRect(0, 0, dims.width, dims.height);

    ctx.save();
    // Apply Pan & Zoom transformations
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // 1. Draw Lat/Lon Grids (Reference Grid lines)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 0.5;
    ctx.fillStyle = 'rgba(143, 148, 168, 0.4)';
    ctx.font = '8px monospace';

    // Longitude lines
    for (let lon = 70; lon <= 95; lon += 5) {
      const x = toCanvasX(lon);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, dims.height);
      ctx.stroke();
      ctx.fillText(`${lon}°E`, x + 2, dims.height - 5);
    }
    // Latitude lines
    for (let lat = 10; lat <= 35; lat += 5) {
      const y = toCanvasY(lat);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(dims.width, y);
      ctx.stroke();
      ctx.fillText(`${lat}°N`, 5, y - 2);
    }

    // 2. Draw AQI grid cells
    if (gridData && gridData.length > 0) {
      // Calculate cell sizes on canvas based on 0.5-degree resolution
      const cellW = (0.5 / (MAX_LON - MIN_LON)) * dims.width;
      const cellH = (0.5 / (MAX_LAT - MIN_LAT)) * dims.height;

      gridData.forEach((cell) => {
        const x = toCanvasX(cell.lon) - cellW / 2;
        const y = toCanvasY(cell.lat) - cellH / 2;
        const color = AQI_COLORS[cell.AQI_Category] || '#888';

        ctx.fillStyle = color;
        // Make grid cells slightly semi-transparent to allow grids/fires overlay transparency
        ctx.globalAlpha = 0.65;
        ctx.fillRect(x, y, cellW + 0.3, cellH + 0.3); // add 0.3 to prevent seams
      });
      ctx.globalAlpha = 1.0;
    }

    // 3. Draw Active Fires (NASA FIRMS)
    if (showFires && fires && fires.length > 0) {
      fires.forEach((fire) => {
        const x = toCanvasX(fire.longitude);
        const y = toCanvasY(fire.latitude);

        // Draw small warning halo for fires
        ctx.beginPath();
        ctx.arc(x, y, 4 + (fire.frp ? Math.min(6, fire.frp / 40) : 0), 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 56, 56, 0.4)';
        ctx.fill();

        // Draw core hot dot
        ctx.beginPath();
        ctx.arc(x, y, 1.8, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff7f50';
        ctx.fill();
      });
    }

    // 4. Draw DBSCAN Hotspots Centroids
    if (showCentroids && hotspots && hotspots.length > 0) {
      const timeMs = Date.now();
      const pulseRadius = (timeMs % 1500) / 1500; // Pulsing multiplier

      hotspots.forEach((hs) => {
        const x = toCanvasX(hs.centroid_lon);
        const y = toCanvasY(hs.centroid_lat);
        const color = hs.fire_associated ? '#ff00ff' : '#00ffff'; // Magenta for fire-linked, Cyan for industrial

        // Draw pulsating concentric circle
        ctx.beginPath();
        ctx.arc(x, y, (hs.n_pixels * 1.5 + 8) * (1 + pulseRadius * 0.4), 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 1 - pulseRadius;
        ctx.stroke();

        // Solid centroid indicator
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        // Small crosshair lines
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y);
        ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8);
        ctx.stroke();
      });
      ctx.globalAlpha = 1.0;
    }

    ctx.restore();
  }, [dims, gridData, hotspots, fires, showFires, showCentroids, zoom, pan]);

  // Handle drag to pan map
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - pan.x,
      y: e.clientY - pan.y
    });
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
      setHoveredCell(null);
    } else {
      // Handle Hover/Tooltip logic
      // Translate canvas coordinate to lat/lon
      const lon = fromCanvasX(x);
      const lat = fromCanvasY(y);

      // Search for nearest cell in grid data
      let nearest = null;
      let minDist = 0.6; // search within ~60km (grid size is 0.5 degrees)

      if (gridData && gridData.length > 0) {
        gridData.forEach((cell) => {
          const dist = Math.sqrt(Math.pow(cell.lon - lon, 2) + Math.pow(cell.lat - lat, 2));
          if (dist < minDist) {
            minDist = dist;
            nearest = cell;
          }
        });
      }

      // Check if hovered on a hotspot centroid instead
      let hoveredHS = null;
      if (showCentroids && hotspots && hotspots.length > 0) {
        hotspots.forEach((hs) => {
          const hsX = toCanvasX(hs.centroid_lon) * zoom + pan.x;
          const hsY = toCanvasY(hs.centroid_lat) * zoom + pan.y;
          const distPx = Math.sqrt(Math.pow(hsX - x, 2) + Math.pow(hsY - y, 2));
          if (distPx <= 12) {
            hoveredHS = hs;
          }
        });
      }

      // Check if hovered on a fire spot
      let hoveredFire = null;
      if (showFires && fires && fires.length > 0) {
        fires.forEach((f) => {
          const fX = toCanvasX(f.longitude) * zoom + pan.x;
          const fY = toCanvasY(f.latitude) * zoom + pan.y;
          const distPx = Math.sqrt(Math.pow(fX - x, 2) + Math.pow(fY - y, 2));
          if (distPx <= 8) {
            hoveredFire = f;
          }
        });
      }

      if (hoveredHS) {
        setHoveredCell({
          type: 'hotspot',
          clusterId: hoveredHS.cluster_id,
          nPixels: hoveredHS.n_pixels,
          meanHcho: hoveredHS.mean_hcho,
          fireAssociated: hoveredHS.fire_associated,
          lat: hoveredHS.centroid_lat,
          lon: hoveredHS.centroid_lon
        });
        setTooltipPos({ x, y });
      } else if (hoveredFire) {
        setHoveredCell({
          type: 'fire',
          frp: hoveredFire.frp,
          confidence: hoveredFire.confidence,
          lat: hoveredFire.latitude,
          lon: hoveredFire.longitude
        });
        setTooltipPos({ x, y });
      } else if (nearest) {
        setHoveredCell({
          type: 'grid',
          ...nearest
        });
        setTooltipPos({ x, y });
      } else {
        setHoveredCell(null);
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomIntensity = 0.1;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const wheel = e.deltaY < 0 ? 1 : -1;
    const zoomFactor = Math.exp(wheel * zoomIntensity);

    const newZoom = Math.min(Math.max(zoom * zoomFactor, 0.7), 8.0);

    // Zoom centered on cursor
    setPan({
      x: mouseX - (mouseX - pan.x) * (newZoom / zoom),
      y: mouseY - (mouseY - pan.y) * (newZoom / zoom)
    });
    setZoom(newZoom);
    setHoveredCell(null);
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setHoveredCell(null);
  };

  return (
    <div className="map-canvas-container" ref={containerRef} onWheel={handleWheel}>
      <canvas
        ref={canvasRef}
        width={dims.width}
        height={dims.height}
        className="map-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Map Reset Button */}
      <button 
        onClick={resetView} 
        style={{
          position: 'absolute',
          bottom: '12px',
          right: '12px',
          background: 'rgba(26, 28, 40, 0.85)',
          border: '1px solid var(--border-muted)',
          color: 'var(--text-primary)',
          fontSize: '11px',
          padding: '6px 10px',
          borderRadius: '4px',
          cursor: 'pointer',
          zIndex: 10
        }}
      >
        🛰️ Recenter Map
      </button>

      {/* Interactive Tooltip Overlay */}
      {hoveredCell && (
        <div 
          className="canvas-tooltip"
          style={{
            left: `${tooltipPos.x + 15}px`,
            top: `${tooltipPos.y + 15}px`
          }}
        >
          {hoveredCell.type === 'grid' && (
            <>
              <div className="tooltip-title">Grid Cell Telemetry</div>
              <div className="tooltip-row"><span>Coords:</span><span className="tooltip-val">{hoveredCell.lat.toFixed(2)}°N, {hoveredCell.lon.toFixed(2)}°E</span></div>
              <div className="tooltip-row"><span>Predicted AQI:</span><span className="tooltip-val" style={{color: AQI_COLORS[hoveredCell.AQI_Category]}}>{hoveredCell.AQI.toFixed(1)}</span></div>
              <div className="tooltip-row"><span>AQI Category:</span><span className="tooltip-val" style={{color: AQI_COLORS[hoveredCell.AQI_Category]}}>{hoveredCell.AQI_Category}</span></div>
              <div className="tooltip-row"><span>HCHO (mol/m²):</span><span className="tooltip-val">{(hoveredCell.HCHO * 1e4).toFixed(2)}e-4</span></div>
              <div className="tooltip-row"><span>NO2 (mol/m²):</span><span className="tooltip-val">{(hoveredCell.NO2 * 1e4).toFixed(2)}e-4</span></div>
              <div className="tooltip-row"><span>CO (mol/m²):</span><span className="tooltip-val">{hoveredCell.CO.toFixed(3)}</span></div>
              <div className="tooltip-row"><span>Aerosol Index:</span><span className="tooltip-val">{hoveredCell.aerosol_index.toFixed(2)}</span></div>
              <div className="tooltip-row"><span>Temp / RH:</span><span className="tooltip-val">{hoveredCell.temp}°C / {hoveredCell.RH}%</span></div>
              <div className="tooltip-row"><span>Nearby Fires:</span><span className="tooltip-val" style={{color: hoveredCell.fire_count > 0 ? '#ff3838' : 'inherit'}}>{hoveredCell.fire_count}</span></div>
            </>
          )}

          {hoveredCell.type === 'hotspot' && (
            <>
              <div className="tooltip-title" style={{color: '#ff00ff'}}>DBSCAN Hotspot</div>
              <div className="tooltip-row"><span>ID:</span><span className="tooltip-val">{hoveredCell.clusterId}</span></div>
              <div className="tooltip-row"><span>Centroid:</span><span className="tooltip-val">{hoveredCell.lat.toFixed(2)}°N, {hoveredCell.lon.toFixed(2)}°E</span></div>
              <div className="tooltip-row"><span>Cluster Pixels:</span><span className="tooltip-val">{hoveredCell.nPixels}</span></div>
              <div className="tooltip-row"><span>Mean HCHO:</span><span className="tooltip-val">{(hoveredCell.meanHcho * 1e4).toFixed(2)}e-4</span></div>
              <div className="tooltip-row"><span>Fire Linked:</span><span className="tooltip-val" style={{color: hoveredCell.fireAssociated ? '#ff00ff' : '#00ffff'}}>{hoveredCell.fireAssociated ? 'Yes (Biomass)' : 'No (Industrial)'}</span></div>
            </>
          )}

          {hoveredCell.type === 'fire' && (
            <>
              <div className="tooltip-title" style={{color: '#ff3838'}}>NASA FIRMS Active Fire</div>
              <div className="tooltip-row"><span>Coords:</span><span className="tooltip-val">{hoveredCell.lat.toFixed(4)}°N, {hoveredCell.lon.toFixed(4)}°E</span></div>
              <div className="tooltip-row"><span>FRP (Power):</span><span className="tooltip-val" style={{color: '#ff9f1c'}}>{hoveredCell.frp.toFixed(1)} MW</span></div>
              <div className="tooltip-row"><span>Confidence:</span><span className="tooltip-val">{hoveredCell.confidence}%</span></div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
