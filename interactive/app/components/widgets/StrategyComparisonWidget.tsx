"use client";

import { useState, useMemo } from "react";

/* ------------------------------------------------------------------ */
/*  Strategy Comparison Widget                                        */
/*  Side-by-side comparison of all 5 learning strategies              */
/* ------------------------------------------------------------------ */

function fmt(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

interface Strategy {
  id: string;
  name: string;
  color: string;
  chClass: string;
  modifiesWeights: boolean;
  learnTime: number; // seconds
  evalTime: number;
  forgettingRisk: string;
  memoryMB: number;
  complexity: string;
  externalDeps: string;
  bestFor: string;
  mechanism: string;
}

const STRATEGIES: Strategy[] = [
  {
    id: "ttt", name: "TTT-E2E", color: "#4ade80", chClass: "text-ch3",
    modifiesWeights: true, learnTime: 12, evalTime: 2.5,
    forgettingRisk: "Low (TF-IDF gated)", memoryMB: 24000,
    complexity: "High", externalDeps: "None",
    bestFor: "Deep knowledge internalization",
    mechanism: "Gradient descent on trainable MLP with TF-IDF masking",
  },
  {
    id: "mvp", name: "JitRL MVP", color: "#a78bfa", chClass: "text-ch4",
    modifiesWeights: false, learnTime: 0.002, evalTime: 2.4,
    forgettingRisk: "None", memoryMB: 100,
    complexity: "Low", externalDeps: "scikit-learn",
    bestFor: "Fast prototyping, simple QA",
    mechanism: "TF-IDF retrieval + logit biasing",
  },
  {
    id: "full", name: "JitRL Full", color: "#f472b6", chClass: "text-ch5",
    modifiesWeights: false, learnTime: 0.04, evalTime: 33,
    forgettingRisk: "None", memoryMB: 6000,
    complexity: "Medium", externalDeps: "None",
    bestFor: "Semantic understanding tasks",
    mechanism: "Hidden-state embeddings + reward-guided logit modulation",
  },
  {
    id: "ace", name: "ACE", color: "#facc15", chClass: "text-ch6",
    modifiesWeights: false, learnTime: 90, evalTime: 5,
    forgettingRisk: "None", memoryMB: 500,
    complexity: "Medium", externalDeps: "Ollama (7B+ model)",
    bestFor: "Iterative quality improvement",
    mechanism: "Generate-Reflect-Curate loops evolving a playbook",
  },
  {
    id: "d2l", name: "Doc-to-LoRA", color: "#fb923c", chClass: "text-ch7",
    modifiesWeights: true, learnTime: 0.8, evalTime: 2.5,
    forgettingRisk: "Minimal (LoRA deltas)", memoryMB: 8000,
    complexity: "High", externalDeps: "HuggingFace, Sakana AI checkpoint",
    bestFor: "Production deployment, fast adaptation",
    mechanism: "Perceiver hypernetwork generates LoRA in single forward pass",
  },
];

const METRICS = [
  { key: "learnTime", label: "Learn Time", unit: "s", logScale: true },
  { key: "evalTime", label: "Eval Time", unit: "s", logScale: true },
  { key: "memoryMB", label: "Memory", unit: "MB", logScale: true },
] as const;

export default function StrategyComparisonWidget() {
  const [selectedMetric, setSelectedMetric] = useState<typeof METRICS[number]["key"]>("learnTime");
  const [hoveredStrategy, setHoveredStrategy] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);

  const metric = METRICS.find((m) => m.key === selectedMetric)!;

  const maxVal = useMemo(() => {
    return Math.max(...STRATEGIES.map((s) => s[selectedMetric] as number));
  }, [selectedMetric]);

  const getBarWidth = (val: number): number => {
    if (metric.logScale) {
      const logMax = Math.log10(maxVal + 1);
      const logVal = Math.log10(val + 1);
      return (logVal / logMax) * 100;
    }
    return (val / maxVal) * 100;
  };

  return (
    <div className="widget-container ch3">
      <div className="widget-label">
        <span className="text-ch3">Interactive</span>{"  ·  "}Strategy Comparison
      </div>

      {/* Metric selector */}
      <div className="flex flex-wrap gap-2 mb-5">
        {METRICS.map((m) => (
          <button
            key={m.key}
            className={`btn-mono ${selectedMetric === m.key ? "active" : ""}`}
            onClick={() => setSelectedMetric(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Bar chart */}
      <div className="rounded-lg border border-white/[0.06] bg-black/20 p-4 mb-5">
        <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-3">
          {metric.label} ({metric.unit}){metric.logScale ? " — log scale" : ""}
        </div>
        <div className="space-y-3">
          {STRATEGIES.map((s) => {
            const val = s[selectedMetric] as number;
            const width = getBarWidth(val);
            const isHovered = hoveredStrategy === s.id;
            return (
              <div
                key={s.id}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredStrategy(s.id)}
                onMouseLeave={() => setHoveredStrategy(null)}
              >
                <div className="flex items-center gap-3 mb-1">
                  <span className={`font-mono text-xs w-24 ${s.chClass}`}>{s.name}</span>
                  <div className="flex-1 h-5 rounded bg-white/[0.04] overflow-hidden">
                    <div
                      className="h-full rounded transition-all duration-300"
                      style={{
                        width: `${Math.max(2, width)}%`,
                        background: s.color,
                        opacity: isHovered ? 1 : 0.7,
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs text-foreground w-20 text-right">
                    {val < 0.01 ? val.toFixed(3) : val < 1 ? val.toFixed(2) : val >= 1000 ? fmt(val) : val.toFixed(1)} {metric.unit}
                  </span>
                </div>
                {isHovered && (
                  <div className="ml-28 animate-fade-in">
                    <span className="font-mono text-[0.6rem] text-muted-foreground">
                      {s.mechanism}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick comparison cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
        {STRATEGIES.map((s) => (
          <div
            key={s.id}
            className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center cursor-pointer transition-all hover:bg-white/[0.02]"
            onMouseEnter={() => setHoveredStrategy(s.id)}
            onMouseLeave={() => setHoveredStrategy(null)}
            style={{
              borderColor: hoveredStrategy === s.id ? s.color + "40" : undefined,
            }}
          >
            <div className={`font-mono text-[0.65rem] font-semibold ${s.chClass}`}>{s.name}</div>
            <div className="font-mono text-[0.55rem] text-muted-foreground mt-1">
              {s.modifiesWeights ? "Modifies weights" : "No weight changes"}
            </div>
            <div className="font-mono text-[0.55rem] text-muted-foreground">
              {s.forgettingRisk}
            </div>
          </div>
        ))}
      </div>

      {/* Full table toggle */}
      <button className={`btn-mono ${showTable ? "active" : ""}`} onClick={() => setShowTable(!showTable)}>
        {showTable ? "Hide" : "Show"} Full Comparison Table
      </button>
      {showTable && (
        <div className="mt-3 overflow-x-auto animate-fade-in">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="text-muted-foreground">
                <th className="text-left py-1.5 px-2 border-b border-white/5">Strategy</th>
                <th className="text-right py-1.5 px-2 border-b border-white/5">Weights</th>
                <th className="text-right py-1.5 px-2 border-b border-white/5">Learn</th>
                <th className="text-right py-1.5 px-2 border-b border-white/5">Eval</th>
                <th className="text-right py-1.5 px-2 border-b border-white/5">VRAM</th>
                <th className="text-right py-1.5 px-2 border-b border-white/5">Forgetting</th>
                <th className="text-left py-1.5 px-2 border-b border-white/5">Best For</th>
              </tr>
            </thead>
            <tbody>
              {STRATEGIES.map((s) => (
                <tr key={s.id} className="hover:bg-white/[0.02]">
                  <td className={`py-1.5 px-2 border-b border-white/[0.03] ${s.chClass}`}>{s.name}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.modifiesWeights ? "Yes" : "No"}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.learnTime}s</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.evalTime}s</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.memoryMB >= 1000 ? `${(s.memoryMB / 1000).toFixed(0)}GB` : `${s.memoryMB}MB`}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.forgettingRisk}</td>
                  <td className="py-1.5 px-2 border-b border-white/[0.03] text-muted-foreground">{s.bestFor}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
