import { useAppStore } from "@/store/useAppStore";
import { RotateCcw } from "lucide-react";

const sliders = [
  { key: "strength" as const, label: "修复强度", min: 0, max: 1, step: 0.01 },
  { key: "guidanceScale" as const, label: "引导系数", min: 1, max: 20, step: 0.5 },
  { key: "steps" as const, label: "推理步数", min: 10, max: 50, step: 1 },
  { key: "blendFactor" as const, label: "混合因子", min: 0, max: 1, step: 0.01 },
];

export default function ParameterPanel() {
  const { params, setParams, resetParams } = useAppStore();

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ivory">参数调节</h3>
        <button
          onClick={resetParams}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-ivory-dim transition-colors hover:bg-dark-700 hover:text-ivory"
        >
          <RotateCcw className="h-3 w-3" />
          重置
        </button>
      </div>

      <div className="space-y-4">
        {sliders.map(({ key, label, min, max, step }) => (
          <div key={key}>
            <div className="mb-1.5 flex items-center justify-between">
              <label className="text-xs font-medium text-ivory-muted">
                {label}
              </label>
              <span className="rounded bg-dark-700 px-1.5 py-0.5 font-mono text-xs text-amber-accent">
                {params[key]}
              </span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={params[key]}
              onChange={(e) =>
                setParams({ [key]: parseFloat(e.target.value) })
              }
            />
          </div>
        ))}

        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label className="text-xs font-medium text-ivory-muted">
              随机种子
            </label>
          </div>
          <input
            type="number"
            value={params.seed}
            onChange={(e) =>
              setParams({ seed: parseInt(e.target.value) || 0 })
            }
            className="w-full rounded-lg border border-dark-600 bg-dark-800 px-3 py-2 font-mono text-sm text-ivory outline-none transition-colors focus:border-amber-accent/50"
          />
        </div>
      </div>
    </div>
  );
}
