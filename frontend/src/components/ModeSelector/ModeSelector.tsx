import { useAppStore, type RestoreMode } from "@/store/useAppStore";
import { Zap, SlidersHorizontal, Sparkles } from "lucide-react";

const modes: {
  value: RestoreMode;
  label: string;
  desc: string;
  icon: typeof Zap;
}[] = [
  {
    value: "quick",
    label: "快速修复",
    desc: "默认参数，一键修复",
    icon: Zap,
  },
  {
    value: "fine",
    label: "精细修复",
    desc: "自定义参数，精细控制",
    icon: SlidersHorizontal,
  },
  {
    value: "enhance_only",
    label: "仅增强",
    desc: "SwinIR 超分辨率增强",
    icon: Sparkles,
  },
];

export default function ModeSelector() {
  const { params, setParams } = useAppStore();

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-ivory">修复模式</h3>
      <div className="space-y-2">
        {modes.map(({ value, label, desc, icon: Icon }) => (
          <button
            key={value}
            onClick={() => setParams({ mode: value })}
            className={`flex w-full items-center gap-3 rounded-xl border px-3.5 py-3 text-left transition-all duration-200 ${
              params.mode === value
                ? "border-amber-accent/40 bg-amber-accent/10"
                : "border-dark-700/50 bg-dark-800/50 hover:border-dark-600 hover:bg-dark-700/50"
            }`}
          >
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                params.mode === value
                  ? "bg-amber-accent/20 text-amber-accent"
                  : "bg-dark-700 text-ivory-muted"
              }`}
            >
              <Icon className="h-4 w-4" />
            </div>
            <div>
              <p
                className={`text-sm font-medium ${
                  params.mode === value ? "text-amber-accent" : "text-ivory"
                }`}
              >
                {label}
              </p>
              <p className="text-xs text-ivory-dim">{desc}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
