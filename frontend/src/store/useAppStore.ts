import { create } from "zustand";

export type RestoreMode = "quick" | "fine" | "enhance_only";
export type TaskStatus = "idle" | "uploading" | "detecting" | "processing" | "completed" | "error";

export interface RestoreParams {
  mode: RestoreMode;
  strength: number;
  guidanceScale: number;
  steps: number;
  seed: number;
  blendFactor: number;
}

export interface TaskInfo {
  id: string;
  filename: string;
  originalUrl: string;
  restoredUrl?: string;
  maskUrl?: string;
  stage1Url?: string;
  mode: RestoreMode;
  status: TaskStatus;
  progress: number;
  createdAt: string;
  params: RestoreParams;
  width?: number;
  height?: number;
  fileSize?: number;
}

interface AppState {
  currentTask: TaskInfo | null;
  history: TaskInfo[];
  params: RestoreParams;
  activeTab: "workshop" | "history";

  setActiveTab: (tab: "workshop" | "history") => void;
  setParams: (params: Partial<RestoreParams>) => void;
  setCurrentTask: (task: TaskInfo | null) => void;
  updateTaskStatus: (status: TaskStatus, progress?: number) => void;
  setRestoredResult: (restoredUrl: string, maskUrl?: string, stage1Url?: string) => void;
  addToHistory: (task: TaskInfo) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;
  resetParams: () => void;
}

const defaultParams: RestoreParams = {
  mode: "quick",
  strength: 0.35,
  guidanceScale: 7.5,
  steps: 25,
  seed: 42,
  blendFactor: 0.8,
};

export const useAppStore = create<AppState>((set) => ({
  currentTask: null,
  history: [],
  params: { ...defaultParams },
  activeTab: "workshop",

  setActiveTab: (tab) => set({ activeTab: tab }),

  setParams: (partial) =>
    set((state) => ({
      params: { ...state.params, ...partial },
    })),

  setCurrentTask: (task) => set({ currentTask: task }),

  updateTaskStatus: (status, progress) =>
    set((state) => {
      if (!state.currentTask) return state;
      return {
        currentTask: {
          ...state.currentTask,
          status,
          progress: progress ?? state.currentTask.progress,
        },
      };
    }),

  setRestoredResult: (restoredUrl, maskUrl, stage1Url) =>
    set((state) => {
      if (!state.currentTask) return state;
      const updated = {
        ...state.currentTask,
        restoredUrl,
        maskUrl,
        stage1Url,
        status: "completed" as TaskStatus,
        progress: 100,
      };
      return {
        currentTask: updated,
        history: [updated, ...state.history],
      };
    }),

  addToHistory: (task) =>
    set((state) => ({
      history: [task, ...state.history],
    })),

  removeFromHistory: (id) =>
    set((state) => ({
      history: state.history.filter((t) => t.id !== id),
    })),

  clearHistory: () => set({ history: [] }),

  resetParams: () => set({ params: { ...defaultParams } }),
}));
