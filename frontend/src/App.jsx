import React, { useEffect, useState } from "react";
import ErrorMessage from "./components/ErrorMessage";
import FileUploader from "./components/FileUploader";
import LoadingSpinner from "./components/LoadingSpinner";
import ResultDisplay from "./components/ResultDisplay";
import { predictAudio } from "./services/api";
import "./App.css";

function App() {
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem("theme");
    return savedTheme === "light" ? "light" : "dark";
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await predictAudio(selectedFile);
      if (response.data.success) {
        setResult(response.data.data);
      } else {
        setError(response.data.error?.message || "Prediction failed");
      }
    } catch (err) {
      const errorMessage =
        err.response?.data?.error?.message ||
        (err.message === "Network Error"
          ? "Cannot connect to backend API. Start the backend server at http://localhost:8000 and try again."
          : err.message) ||
        "An error occurred during prediction";
      setError(errorMessage);
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <div className="bg-orb bg-orb-1" aria-hidden="true"></div>
      <div className="bg-orb bg-orb-2" aria-hidden="true"></div>

      <header className="header">
        <div className="hero-grid">
          <div
            className="theme-icon-switch"
            role="button"
            tabIndex={0}
            onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setTheme((value) => (value === "dark" ? "light" : "dark"));
              }
            }}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1 1 11.21 3c0 0 0 0 0 0A7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </div>

          <div className="hero-copy">
            <h1>Deepfake Voice Detector</h1>
            <p className="subtitle">Upload or record audio to get a transcript and authenticity estimate.</p>


          </div>
        </div>
      </header>

      <main className="main">
        <section className="trust-strip" aria-label="Usage context">
          <article className="trust-card">
            <div className="trust-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <h3>Human Review Required</h3>
            <p>Use this result as decision support, not as a sole source of truth.</p>
          </article>
        </section>

        <section className="panel" aria-label="Voice analysis workspace">
          {!result && !isLoading && (
            <FileUploader onFileSelect={handleFileSelect} onSubmit={handleSubmit} selectedFile={selectedFile} error={error} />
          )}

          {isLoading && <LoadingSpinner />}

          {error && !isLoading && <ErrorMessage message={error} onRetry={handleReset} />}

          {result && !isLoading && <ResultDisplay result={result} onReset={handleReset} />}
        </section>
      </main>

      <footer className="footer">
        <p>Review results with human judgment and domain context.</p>
      </footer>
    </div>
  );
}

export default App;
