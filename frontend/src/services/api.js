import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export const predictAudio = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    return await api.post("/predict", formData);
  } catch (error) {
    if (error.code === "ECONNABORTED") {
      throw new Error("Request timeout. The file may be too large or processing is taking too long.");
    }
    throw error;
  }
};

export const getModelsInfo = async () => {
  const response = await api.get("/models/info");
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get("/health");
  return response.data;
};
