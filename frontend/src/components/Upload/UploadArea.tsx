import { useCallback, useRef, useState } from "react";
import { Upload, Image as ImageIcon, X, Loader } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/utils/api";

export default function UploadArea() {
  const { setCurrentTask, currentTask } = useAppStore();
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) return;

      setIsUploading(true);
      try {
        const task = await api.upload(file);
        const localUrl = URL.createObjectURL(file);
        setCurrentTask({
          id: task.id,
          filename: task.filename || file.name,
          originalUrl: localUrl,
          mode: "quick",
          status: "idle",
          progress: 0,
          createdAt: task.created_at || new Date().toISOString(),
          params: useAppStore.getState().params,
          width: task.width,
          height: task.height,
          fileSize: task.file_size || file.size,
        });
      } catch (e) {
        console.error("Upload error:", e);
        alert("Upload failed: " + (e instanceof Error ? e.message : "Unknown error"));
      } finally {
        setIsUploading(false);
      }
    },
    [setCurrentTask],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const clearImage = useCallback(() => {
    if (currentTask?.originalUrl) {
      URL.revokeObjectURL(currentTask.originalUrl);
    }
    setCurrentTask(null);
  }, [currentTask, setCurrentTask]);

  if (isUploading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dark-600 bg-dark-900/50 py-10">
        <Loader className="h-8 w-8 animate-spin text-amber-accent" />
        <p className="text-sm text-ivory-muted">Uploading...</p>
      </div>
    );
  }

  if (currentTask) {
    return (
      <div className="relative overflow-hidden rounded-xl border border-dark-700/50 bg-dark-900">
        <img
          src={currentTask.originalUrl}
          alt={currentTask.filename}
          className="h-48 w-full object-cover"
        />
        <button
          onClick={clearImage}
          className="absolute right-2 top-2 rounded-lg bg-dark-950/70 p-1.5 text-ivory-muted backdrop-blur-sm transition-colors hover:bg-red-500/80 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
        <div className="p-3">
          <p className="truncate text-xs font-medium text-ivory">
            {currentTask.filename}
          </p>
          <p className="mt-0.5 text-xs text-ivory-dim">
            {currentTask.width} x {currentTask.height}
            {currentTask.fileSize && (
              <span className="ml-2">
                {(currentTask.fileSize / 1024 / 1024).toFixed(1)} MB
              </span>
            )}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`group relative cursor-pointer rounded-xl border-2 border-dashed transition-all duration-300 ${
        isDragOver
          ? "border-amber-accent bg-amber-accent/5"
          : "border-dark-600 bg-dark-900/50 hover:border-dark-500 hover:bg-dark-800/50"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleInputChange}
      />
      <div className="flex flex-col items-center gap-3 py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={isDragOver ? "drag" : "idle"}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className={`flex h-14 w-14 items-center justify-center rounded-2xl ${
              isDragOver
                ? "bg-amber-accent/20 text-amber-accent"
                : "bg-dark-700 text-ivory-muted group-hover:bg-dark-600 group-hover:text-ivory"
            }`}
          >
            {isDragOver ? (
              <Upload className="h-6 w-6" />
            ) : (
              <ImageIcon className="h-6 w-6" />
            )}
          </motion.div>
        </AnimatePresence>
        <div className="text-center">
          <p className="text-sm font-medium text-ivory">
            {isDragOver ? "Drop photo here" : "Drag photo here"}
          </p>
          <p className="mt-1 text-xs text-ivory-dim">
            or click to select file (JPG / PNG)
          </p>
        </div>
      </div>
    </div>
  );
}
