const API_BASE = "/api";

export interface TaskInfo {
  id: string;
  filename: string;
  original_url: string;
  restored_url?: string;
  mask_url?: string;
  stage1_url?: string;
  mode: string;
  status: string;
  progress: number;
  created_at: string;
  width?: number;
  height?: number;
  file_size?: number;
  error?: string;
  params?: RestoreParams;
}

export interface RestoreParams {
  mode: string;
  strength: number;
  guidanceScale: number;
  steps: number;
  seed: number;
  blendFactor: number;
}

export interface ServerStatus {
  status: string;
  pipeline_loaded: boolean;
  active_tasks: number;
  cuda_available: boolean;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Accept": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  upload(file: File): Promise<TaskInfo> {
    const form = new FormData();
    form.append("file", file);
    return request<TaskInfo>("/upload", {
      method: "POST",
      body: form,
    });
  },

  detect(taskId: string, sensitivity = 0.5): Promise<{ mask_url: string; coverage: number }> {
    return request("/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, sensitivity }),
    });
  },

  restore(taskId: string, params: RestoreParams): Promise<{ task_id: string; status: string }> {
    return request("/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, ...params }),
    });
  },

  getTask(taskId: string): Promise<TaskInfo> {
    return request(`/task/${taskId}`);
  },

  getHistory(): Promise<{ tasks: TaskInfo[] }> {
    return request("/history");
  },

  deleteHistory(taskId: string): Promise<{ success: boolean }> {
    return request(`/history/${taskId}`, { method: "DELETE" });
  },

  getStatus(): Promise<ServerStatus> {
    return request("/status");
  },
};
