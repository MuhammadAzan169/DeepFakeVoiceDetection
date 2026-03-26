import React from "react";
import "./ErrorMessage.css";

const ErrorMessage = ({ message, onRetry }) => {
  const isConnectivityIssue = message.toLowerCase().includes("cannot connect") || message.toLowerCase().includes("network");

  return (
    <div className="error-message">
      <svg className="error-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>

      <h3>Error Processing File</h3>
      <p className="error-detail">{message}</p>

      {isConnectivityIssue && (
        <div className="error-help">
          <p>Quick fix:</p>
          <ol>
            <li>Start backend server in the backend folder</li>
            <li>Ensure it is running on port 8000</li>
            <li>Retry upload in this page</li>
          </ol>
        </div>
      )}

      <button className="retry-button" onClick={onRetry}>
        Try Again
      </button>
    </div>
  );
};

export default ErrorMessage;
