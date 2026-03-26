import React, { useCallback, useEffect, useRef, useState } from "react";
import "./FileUploader.css";

const FileUploader = ({ onFileSelect, onSubmit, selectedFile, error }) => {
  const [inputMode, setInputMode] = useState("upload");
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingError, setRecordingError] = useState("");

  const isRecordingRef = useRef(false);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const sampleRateRef = useRef(44100);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      cleanupRecorder();
    };
  }, []);

  const getExtension = (fileName) => {
    const parts = fileName.split(".");
    return parts.length > 1 ? parts.pop().toUpperCase() : "UNKNOWN";
  };

  const formatTime = (value) => {
    const minutes = String(Math.floor(value / 60)).padStart(2, "0");
    const seconds = String(value % 60).padStart(2, "0");
    return `${minutes}:${seconds}`;
  };

  const floatTo16BitPCM = (view, offset, input) => {
    for (let i = 0; i < input.length; i += 1) {
      const s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  };

  const createWavBuffer = (monoFloat32, sampleRate) => {
    const dataLength = monoFloat32.length * 2;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i += 1) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataLength, true);

    floatTo16BitPCM(view, 44, monoFloat32);
    return buffer;
  };

  const mergeChunks = (chunks) => {
    const totalLength = chunks.reduce((sum, arr) => sum + arr.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  };

  const cleanupRecorder = () => {
    isRecordingRef.current = false;

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current.onaudioprocess = null;
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const startRecording = async () => {
    setRecordingError("");

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !AudioCtx) {
      setRecordingError("Recording is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioCtx();
      const sourceNode = audioContext.createMediaStreamSource(stream);
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1);

      sampleRateRef.current = audioContext.sampleRate;
      pcmChunksRef.current = [];
      setRecordingSeconds(0);
      onFileSelect(null);

      processorNode.onaudioprocess = (event) => {
        if (!isRecordingRef.current) return;
        const inputData = event.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(inputData));
      };

      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);

      mediaStreamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceNodeRef.current = sourceNode;
      processorNodeRef.current = processorNode;

      isRecordingRef.current = true;
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingSeconds((value) => value + 1);
      }, 1000);
    } catch {
      setRecordingError("Microphone permission denied or unavailable.");
      setIsRecording(false);
      cleanupRecorder();
    }
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    setIsRecording(false);

    const chunks = pcmChunksRef.current;
    cleanupRecorder();

    if (!chunks || chunks.length === 0) {
      setRecordingError("No audio captured. Please try recording again.");
      return;
    }

    const mono = mergeChunks(chunks);
    const wavBuffer = createWavBuffer(mono, sampleRateRef.current || 44100);
    const wavFile = new File([wavBuffer], `recording-${Date.now()}.wav`, { type: "audio/wav" });
    onFileSelect(wavFile);
  };

  const switchMode = (mode) => {
    setInputMode(mode);
    setRecordingError("");
    setIsDragging(false);

    if (isRecording) {
      stopRecording();
    }

    setRecordingSeconds(0);
    onFileSelect(null);
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith("audio/") || file.name.match(/\.(mp3|wav|webm|ogg|m4a)$/i)) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect]
  );

  const handleFileChange = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (file) {
        setIsDragging(false);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="file-uploader">
      <div className="input-mode-toggle" role="tablist" aria-label="Audio source mode">
        <button type="button" className={`mode-button ${inputMode === "upload" ? "active" : ""}`} onClick={() => switchMode("upload")}>
          Upload File
        </button>
        <button type="button" className={`mode-button ${inputMode === "record" ? "active" : ""}`} onClick={() => switchMode("record")}>
          Record Audio
        </button>
      </div>

      {inputMode === "upload" ? (
        <div
          className={`drop-zone ${selectedFile ? "has-file" : ""} ${error ? "has-error" : ""} ${isDragging ? "is-dragging" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input type="file" id="file-input" className="file-input" accept=".mp3,.wav,.webm,.ogg,.m4a,audio/*" onChange={handleFileChange} />

          {!selectedFile ? (
            <label htmlFor="file-input" className="drop-zone-content">
              <svg className="upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p className="drop-zone-text">
                <span className="primary-text">Click to upload</span> or drag and drop
              </p>
              <p className="drop-zone-hint">MP3, WAV, WEBM, OGG, M4A (max 25MB)</p>
            </label>
          ) : (
            <div className="file-info">
              <svg className="file-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                <polyline points="13 2 13 9 20 9" />
              </svg>
              <div className="file-details">
                <p className="file-name">{selectedFile.name}</p>
                <p className="file-size">{formatFileSize(selectedFile.size)}</p>
                <div className="file-meta-row">
                  <span className="file-meta">{getExtension(selectedFile.name)}</span>
                  <span className="file-meta">Validated</span>
                  <span className="file-meta">Ready</span>
                </div>
              </div>
              <button className="remove-file" onClick={() => onFileSelect(null)} aria-label="Remove file">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className={`record-panel ${error ? "has-error" : ""}`}>
          <div className="record-head">
            <h3>Record from Microphone</h3>
            <p>Use your device mic to capture a sample for analysis.</p>
          </div>

          <div className="record-controls">
            {!isRecording ? (
              <button type="button" className="record-button" onClick={startRecording}>
                Start Recording
              </button>
            ) : (
              <button type="button" className="record-button stop" onClick={stopRecording}>
                Stop Recording ({formatTime(recordingSeconds)})
              </button>
            )}

            {selectedFile && (
              <button type="button" className="secondary-button" onClick={() => onFileSelect(null)}>
                Clear Recording
              </button>
            )}
          </div>

          {recordingError && <p className="record-error">{recordingError}</p>}

          {selectedFile && (
            <div className="file-info recorded">
              <svg className="file-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1v11" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
              <div className="file-details">
                <p className="file-name">{selectedFile.name}</p>
                <p className="file-size">{formatFileSize(selectedFile.size)}</p>
                <div className="file-meta-row">
                  <span className="file-meta">{getExtension(selectedFile.name)}</span>
                  <span className="file-meta">Microphone</span>
                  <span className="file-meta">Ready</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {selectedFile && (
        <button className="submit-button" onClick={onSubmit} disabled={!selectedFile || isRecording}>
          Analyze Voice Sample
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </button>
      )}

    </div>
  );
};

export default FileUploader;
