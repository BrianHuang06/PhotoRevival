import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";

interface CompareSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "原图",
  afterLabel = "修复后",
}: CompareSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const updatePosition = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setPosition(pct);
    },
    []
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setIsDragging(true);
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) return;
      updatePosition(e.clientX);
    },
    [isDragging, updatePosition]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      updatePosition(e.touches[0].clientX);
    },
    [updatePosition]
  );

  return (
    <div
      ref={containerRef}
      className="group relative select-none overflow-hidden rounded-xl border border-dark-700/50 bg-dark-900"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleMouseUp}
      style={{ cursor: isDragging ? "col-resize" : "col-resize" }}
    >
      <img
        src={afterSrc}
        alt={afterLabel}
        className="block h-full w-full object-contain"
        draggable={false}
      />

      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="block h-full w-full object-contain"
          draggable={false}
        />
      </div>

      <motion.div
        className="absolute top-0 bottom-0 z-10 w-0.5 bg-ivory/80"
        style={{ left: `${position}%` }}
        animate={{ opacity: isDragging ? 1 : 0.7 }}
      >
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-ivory/80 bg-dark-950/70 backdrop-blur-sm">
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              className="text-ivory"
            >
              <path
                d="M5 8L2 8M2 8L4 6M2 8L4 10"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M11 8L14 8M14 8L12 6M14 8L12 10"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        </div>
      </motion.div>

      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-dark-950/70 px-2.5 py-1 text-xs font-medium text-ivory backdrop-blur-sm">
        {beforeLabel}
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 rounded-md bg-dark-950/70 px-2.5 py-1 text-xs font-medium text-ivory backdrop-blur-sm">
        {afterLabel}
      </div>
    </div>
  );
}
