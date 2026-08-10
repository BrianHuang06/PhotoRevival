import { type TaskInfo, useAppStore } from "@/store/useAppStore";
import {
  Download,
  Trash2,
  Clock,
  Zap,
  SlidersHorizontal,
  Sparkles,
  CheckCircle,
  AlertCircle,
  Loader,
} from "lucide-react";
import { motion } from "framer-motion";

const modeIcons = {
  quick: Zap,
  fine: SlidersHorizontal,
  enhance_only: Sparkles,
};

const modeLabels = {
  quick: "快速修复",
  fine: "精细修复",
  enhance_only: "仅增强",
};

const statusConfig = {
  idle: { icon: Clock, color: "text-ivory-dim", label: "等待中" },
  uploading: { icon: Loader, color: "text-blue-400", label: "上传中" },
  detecting: { icon: Loader, color: "text-purple-400", label: "检测中" },
  processing: { icon: Loader, color: "text-amber-accent", label: "修复中" },
  completed: { icon: CheckCircle, color: "text-emerald-400", label: "已完成" },
  error: { icon: AlertCircle, color: "text-red-400", label: "出错" },
};

interface TaskCardProps {
  task: TaskInfo;
}

export default function TaskCard({ task }: TaskCardProps) {
  const { removeFromHistory } = useAppStore();
  const ModeIcon = modeIcons[task.mode];
  const { icon: StatusIcon, color: statusColor, label: statusLabel } = statusConfig[task.status];

  const handleDownload = () => {
    if (!task.restoredUrl) return;
    const a = document.createElement("a");
    a.href = task.restoredUrl;
    a.download = `restored_${task.filename}`;
    a.click();
  };

  const timeStr = new Date(task.createdAt).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="group overflow-hidden rounded-xl border border-dark-700/50 bg-dark-800/50 transition-all duration-300 hover:border-dark-600"
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-dark-900">
        {task.restoredUrl ? (
          <img
            src={task.restoredUrl}
            alt={task.filename}
            className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
          />
        ) : task.originalUrl ? (
          <img
            src={task.originalUrl}
            alt={task.filename}
            className="h-full w-full object-cover opacity-60"
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <StatusIcon className={`h-8 w-8 ${statusColor}`} />
          </div>
        )}
        <div className="absolute right-2 top-2 rounded-md bg-dark-950/70 px-2 py-0.5 backdrop-blur-sm">
          <span className={`flex items-center gap-1 text-xs font-medium ${statusColor}`}>
            <StatusIcon className="h-3 w-3" />
            {statusLabel}
          </span>
        </div>
      </div>

      <div className="p-3">
        <p className="truncate text-sm font-medium text-ivory">
          {task.filename}
        </p>
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-ivory-dim">
            <span className="flex items-center gap-1">
              <ModeIcon className="h-3 w-3" />
              {modeLabels[task.mode]}
            </span>
            <span>{timeStr}</span>
          </div>
          <div className="flex items-center gap-1">
            {task.restoredUrl && (
              <button
                onClick={handleDownload}
                className="rounded-md p-1.5 text-ivory-dim transition-colors hover:bg-dark-600 hover:text-amber-accent"
                title="下载"
              >
                <Download className="h-3.5 w-3.5" />
              </button>
            )}
            <button
              onClick={() => removeFromHistory(task.id)}
              className="rounded-md p-1.5 text-ivory-dim transition-colors hover:bg-dark-600 hover:text-red-400"
              title="删除"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
