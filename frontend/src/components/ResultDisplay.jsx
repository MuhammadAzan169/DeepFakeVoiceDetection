import React from "react";
import "./ResultDisplay.css";

const ResultDisplay = ({ result, onReset }) => {
  const { transcript, label, reasons, processing_time } = result;
  const isFake = label === "Fake";

  return (
    <div className="result-display">
      <div className="result-header">
        <h2>Analysis Result</h2>
        <button className="reset-button" onClick={onReset}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
          Analyze Another
        </button>
      </div>

      <div className={`result-badge ${isFake ? "fake" : "real"}`}>
        <div className="badge-icon" aria-hidden="true">
          {isFake ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          )}
        </div>
        <span className="badge-label">{label}</span>
      </div>

      <div className="result-grid">
        <div className="result-stat reveal-item" style={{ animationDelay: "80ms" }}>
          <span className="stat-label">Processing Time</span>
          <span className="stat-value">{processing_time}s</span>
        </div>
      </div>

      <div className="result-section">
        <h3>Transcript</h3>
        <div className="transcript-box">
          <p>{transcript}</p>
        </div>
      </div>

      <div className="result-section">
        <h3>Reasons</h3>
        <ul className="reasons-list">
          {reasons.map((reason, index) => (
            <li key={index} className="reason-item reveal-item" style={{ animationDelay: `${160 + index * 70}ms` }}>
              <span className="reason-bullet" aria-hidden="true"></span>
              <span className="reason-text">{reason}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ResultDisplay;
