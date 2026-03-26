import React from "react";
import "./LoadingSpinner.css";

const LoadingSpinner = () => {
  return (
    <div className="loading-spinner">
      <div className="wave-loader" aria-hidden="true">
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
      </div>
      <p className="loading-text">Analyzing audio file...</p>
      <p className="loading-subtext">This may take a few moments</p>
      <ul className="loading-steps" aria-label="Analysis steps in progress">
        <li>Uploading sample</li>
        <li>Extracting features</li>
        <li>Running model ensemble</li>
        <li>Preparing explanation</li>
      </ul>
    </div>
  );
};

export default LoadingSpinner;
