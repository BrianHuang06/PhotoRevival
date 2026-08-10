import { useCallback, useState } from "react";
import { useAppStore, type RestoreParams } from "@/store/useAppStore";
import UploadArea from "@/components/Upload/UploadArea";
import ModeSelector from "@/components/ModeSelector/ModeSelector";
import ParameterPanel from "@/components/ParameterPanel/ParameterPanel";
import CompareSlider from "@/components/CompareSlider/CompareSlider";
import ProgressBar from "@/components/ProgressBar/ProgressBar";
import {
  Play,
  Download,
  Eye,
  EyeOff,
  Layers,
  Loader,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/utils/api";
import { useRestore } from "@/hooks/useApi";

export default function Workshop() {
  const {
    currentTask,
    params,
    setParams,
    setCurrentTask,
    updateTaskStatus,
    setRestoredResult,
  } = useAppStore();

  const [showMask, setShowMask] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const { startRestore, stopPolling } = useRestore();

  const handleRestore = useCallback(() => {
    if (!currentTask) return;

    const restoreParams: RestoreParams = {
      mode: params.mode,
      strength: params.strength,
      guidanceScale: params.guidanceScale,
      steps: params.steps,
      seed: params.seed,
      blendFactor: params.blendFactor,
    };

    updateTaskStatus("detecting", 5);

    startRestore(
      currentTask.id,
      restoreParams,
      (updatedTask) => {
        const taskInfo = {
          ...currentTask,
          status: updatedTask.status as "idle" | "uploading" | "detecting" | "processing" | "completed" | "error",
          progress: updatedTask.progress,
        };

        if (updatedTask.status === "completed") {
          setRestoredResult(
            updatedTask.restored_url || currentTask.originalUrl,
            updatedTask.mask_url,
            updatedTask.stage1_url,
          );
        } else {
          setCurrentTask(taskInfo);
        }
      },
      (error) => {
        console.error("Restore error:", error);
        updateTaskStatus("error", 0);
      },
    );
  }, [currentTask, params, startRestore, updateTaskStatus, setRestoredResult, setCurrentTask]);

  const handleDetect = useCallback(async () => {
    if (!currentTask) return;
    try {
      const result = await api.detect(currentTask.id, 0.5);
      setMaskUrl(result.mask_url);
      setShowMask(true);
    } catch (e) {
      console.error("Detect error:", e);
    }
  }, [currentTask]);

  const handleDownload = useCallback(() => {
    if (!currentTask?.restoredUrl) return;
    const a = document.createElement("a");
    a.href = currentTask.restoredUrl;
    a.download = `restored_${currentTask.filename}`;
    a.click();
  }, [currentTask]);

  const isProcessing =
    currentTask?.status === "detecting" ||
    currentTask?.status === "processing";

  const stageLabels: Record<string, string> = {
    detecting: "Detecting scratches and damage...",
    processing: "Running AI restoration pipeline (SwinIR + SDXL LoRA)...",
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden">
      <aside className="w-72 shrink-0 overflow-y-auto border-r border-dark-700/50 bg-dark-900/30 p-4">
        <div className="space-y-6">
          <UploadArea />
          {currentTask && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <ModeSelector />
              {params.mode === "fine" && <ParameterPanel />}

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-ivory">
                    Scratch Detection
                  </h3>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleDetect}
                      disabled={isProcessing}
                      className="rounded-md px-2 py-1 text-xs text-amber-accent transition-colors hover:bg-amber-accent/10 disabled:text-ivory-dim disabled:hover:bg-transparent"
                    >
                      Detect
                    </button>
                    {maskUrl && (
                      <button
                        onClick={() => setShowMask(!showMask)}
                        className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors ${
                          showMask
                            ? "bg-amber-accent/10 text-amber-accent"
                            : "text-ivory-dim hover:bg-dark-700 hover:text-ivory"
                        }`}
                      >
                        {showMask ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                        Mask
                      </button>
                    )}
                  </div>
                </div>
              </div>

              <button
                onClick={handleRestore}
                disabled={isProcessing}
                className={`flex w-full items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold transition-all duration-300 ${
                  isProcessing
                    ? "cursor-not-allowed bg-dark-700 text-ivory-dim"
                    : "bg-gradient-to-r from-amber-dark via-amber-accent to-amber-light text-dark-950 shadow-lg shadow-amber-accent/20 hover:shadow-amber-accent/40 hover:brightness-110 active:scale-[0.98]"
                }`}
              >
                {isProcessing ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    Restoring...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Start Restore
                  </>
                )}
              </button>
            </motion.div>
          )}
        </div>
      </aside>

      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-auto p-6">
          <AnimatePresence mode="wait">
            {!currentTask ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full flex-col items-center justify-center gap-6"
              >
                <div className="relative">
                  <div className="absolute -inset-4 rounded-full bg-amber-accent/5 blur-2xl" />
                  <Layers className="relative h-16 w-16 text-dark-600" />
                </div>
                <div className="text-center">
                  <h2 className="font-display text-xl font-semibold text-ivory-muted">
                    Upload an old photo to start
                  </h2>
                  <p className="mt-2 text-sm text-ivory-dim">
                    AI-powered scratch detection and restoration
                  </p>
                </div>
              </motion.div>
            ) : currentTask.status === "completed" && currentTask.restoredUrl ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="mx-auto h-full max-w-4xl"
              >
                <CompareSlider
                  beforeSrc={currentTask.originalUrl}
                  afterSrc={currentTask.restoredUrl}
                  beforeLabel="Original"
                  afterLabel="Restored"
                />
              </motion.div>
            ) : isProcessing ? (
              <motion.div
                key="processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full flex-col items-center justify-center gap-8"
              >
                <div className="w-full max-w-md">
                  <ProgressBar
                    progress={currentTask.progress}
                    status={stageLabels[currentTask.status] || "Processing..."}
                    stage={`Stage: ${
                      currentTask.status === "detecting"
                        ? "Scratch Detection"
                        : "AI Restoration"
                    }`}
                  />
                </div>
                <div className="relative">
                  <img
                    src={currentTask.originalUrl}
                    alt="processing"
                    className="h-64 w-auto rounded-xl object-contain opacity-40 blur-[1px]"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="rounded-xl bg-dark-950/60 px-4 py-2 backdrop-blur-sm">
                      <p className="text-sm font-medium text-amber-accent">
                        AI is restoring...
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mx-auto h-full max-w-4xl"
              >
                <div className="relative overflow-hidden rounded-xl border border-dark-700/50 bg-dark-900">
                  <img
                    src={currentTask.originalUrl}
                    alt={currentTask.filename}
                    className="h-full w-full object-contain"
                  />
                  {showMask && maskUrl && (
                    <img
                      src={maskUrl}
                      alt="mask"
                      className="absolute inset-0 h-full w-full object-contain opacity-50 mix-blend-multiply"
                    />
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {currentTask && (
          <div className="flex items-center justify-between border-t border-dark-700/50 bg-dark-900/50 px-6 py-2.5">
            <div className="flex items-center gap-4 text-xs text-ivory-dim">
              <span>
                Status:{" "}
                <span className="text-ivory-muted">
                  {currentTask.status === "idle"
                    ? "Ready"
                    : currentTask.status === "completed"
                    ? "Completed"
                    : currentTask.status === "error"
                    ? "Error"
                    : "Processing"}
                </span>
              </span>
              {currentTask.width && (
                <span>
                  Size: {currentTask.width} x {currentTask.height}
                </span>
              )}
            </div>
            {currentTask.restoredUrl && (
              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 rounded-lg bg-dark-700 px-3 py-1.5 text-xs font-medium text-ivory transition-colors hover:bg-dark-600 hover:text-amber-accent"
              >
                <Download className="h-3.5 w-3.5" />
                Download
              </button>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
