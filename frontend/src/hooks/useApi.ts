import { useCallback, useRef } from "react";
import { api, type TaskInfo, type RestoreParams } from "@/utils/api";

export function useRestore() {
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const startRestore = useCallback(
    async (
      taskId: string,
      params: RestoreParams,
      onUpdate: (task: TaskInfo) => void,
      onError: (error: string) => void,
    ) => {
      try {
        stopPolling();

        await api.restore(taskId, params);

        pollingRef.current = setInterval(async () => {
          try {
            const task = await api.getTask(taskId);
            onUpdate(task);

            if (task.status === "completed" || task.status === "error") {
              stopPolling();
            }
          } catch {
            stopPolling();
            onError("Failed to get task status");
          }
        }, 1500);
      } catch (e) {
        onError(e instanceof Error ? e.message : "Unknown error");
      }
    },
    [stopPolling],
  );

  return { startRestore, stopPolling };
}
