/**
 * PredictGuard AI - Enterprise API Client (ES Module)
 */

export class ApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
  }

  async checkHealth() {
    const res = await fetch(`${this.baseUrl}/health/ready`);
    const data = await res.json().catch(() => ({ status: "error" }));
    return { ok: res.ok, status: res.status, data };
  }

  async predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${this.baseUrl}/v1/predict/image`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json().catch(() => ({ message: "Failed to parse JSON response" }));
    if (!res.ok) {
      throw new Error(data.message || data.detail || `Server returned error (${res.status})`);
    }
    return data;
  }

  async predictSensors(payload) {
    const res = await fetch(`${this.baseUrl}/v1/predict/sensors`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => ({ message: "Failed to parse JSON response" }));
    if (!res.ok) {
      throw new Error(data.message || data.detail || `Server returned error (${res.status})`);
    }
    return data;
  }

  async getModels() {
    const res = await fetch(`${this.baseUrl}/v1/models`);
    if (!res.ok) {
      throw new Error(`Failed to fetch models (${res.status})`);
    }
    return await res.json();
  }
}

export const api = new ApiClient();
