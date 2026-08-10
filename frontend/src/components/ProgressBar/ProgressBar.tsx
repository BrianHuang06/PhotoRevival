import { motion } from "framer-motion";

interface ProgressBarProps {
  progress: number;
  status: string;
  stage?: string;
}

export default function ProgressBar({
  progress,
  status,
  stage,
}: ProgressBarProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="relative h-2.5 w-2.5">
            <motion.div
              className="absolute inset-0 rounded-full bg-amber-accent"
              animate={{ scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          </div>
          <span className="text-sm font-medium text-ivory">{status}</span>
        </div>
        <span className="font-mono text-sm text-amber-accent">
          {Math.round(progress)}%
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-dark-700">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-amber-dark via-amber-accent to-amber-light"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />
      </div>

      {stage && (
        <p className="text-xs text-ivory-dim">{stage}</p>
      )}
    </div>
  );
}
