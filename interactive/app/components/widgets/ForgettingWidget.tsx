"use client";

import { useState, useMemo } from "react";

/* ------------------------------------------------------------------ */
/*  Forgetting Ratio Widget                                           */
/*  Interactive before/after scoring with forgetting visualization    */
/* ------------------------------------------------------------------ */

interface Strategy {
  name: string;
  color: string;
  beforeAcc: number;
  afterAcc: number;
  learnTime: string;
  modifiesWeights: boolean;
}

const STRATEGIES: Strategy[] = [
  { name: "TTT-E2E", color: "#4ade80", beforeAcc: 0.72, afterAcc: 0.68, learnTime: "~12s", modifiesWeights: true },
  { name: "JitRL MVP", color: "#a78bfa", beforeAcc: 0.72, afterAcc: 0.72, learnTime: "~0.002s", modifiesWeights: false },
  { name: "JitRL Full", color: "#f472b6", beforeAcc: 0.72, afterAcc: 0.72, learnTime: "~0.04s", modifiesWeights: false },
  { name: "ACE", color: "#facc15", beforeAcc: 0.72, afterAcc: 0.72, learnTime: "~90s", modifiesWeights: false },
  { name: "Doc-to-LoRA", color: "#fb923c", beforeAcc: 0.72, afterAcc: 0.70, learnTime: "~0.8s", modifiesWeights: true },
];

function computeForgettingRatio(before: number, after: number): number {
  if (before === 0) return 0;
  return (before - after) / before;
}

export default function ForgettingWidget() {
  const [adjustedScores, setAdjustedScores] = useState<Record<string, { before: number; after: number }>>(
    Object.fromEntries(STRATEGIES.map((s) => [s.name, { before: s.beforeAcc, after: s.afterAcc }]))
  );

  const [selectedStrategy, setSelectedStrategy] = useState(0);

  const current = STRATEGIES[selectedStrategy];
  const scores = adjustedScores[current.name];
  const forgetting = computeForgettingRatio(scores.before, scores.after);

  return (
    <div className="widget-container ch3">
      <div className="widget-label">
        <span className="text-ch3">Interactive</span>{"  ·  "}Catastrophic Forgetting Measurement
      </div>

      <p className="text-sm text-muted-foreground mb-4">
        The forgetting ratio measures how much holdout accuracy drops after learning new documents.
        A ratio of 0 means no forgetting. Target: &lt; 0.15 (15%).
      </p>

      {/* Strategy tabs */}
      <div className="flex flex-wrap gap-2 mb-5">
        {STRATEGIES.map((s, i) => (
          <button
            key={s.name}
            className={`btn-mono ${selectedStrategy === i ? "active" : ""}`}
            onClick={() => setSelectedStrategy(i)}
            style={{
              borderColor: selectedStrategy === i ? s.color : undefined,
              color: selectedStrategy === i ? s.color : undefined,
            }}
          >
            {s.name}
          </button>
        ))}
      </div>

      {/* Score sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
        <div>
          <label className="font-mono text-[0.65rem] text-muted-foreground block mb-1">
            Before learning (holdout accuracy)
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(scores.before * 100)}
              onChange={(e) =>
                setAdjustedScores((prev) => ({
                  ...prev,
                  [current.name]: { ...prev[current.name], before: Number(e.target.value) / 100 },
                }))
              }
              className="flex-1"
            />
            <span className="font-mono text-sm font-bold text-foreground w-14 text-right">
              {(scores.before * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <div>
          <label className="font-mono text-[0.65rem] text-muted-foreground block mb-1">
            After learning (holdout accuracy)
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(scores.after * 100)}
              onChange={(e) =>
                setAdjustedScores((prev) => ({
                  ...prev,
                  [current.name]: { ...prev[current.name], after: Number(e.target.value) / 100 },
                }))
              }
              className="flex-1"
            />
            <span className="font-mono text-sm font-bold w-14 text-right" style={{ color: current.color }}>
              {(scores.after * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="code-block mb-5 text-center">
        <span className="text-muted-foreground">forgetting_ratio = (</span>
        <span className="text-foreground">{scores.before.toFixed(2)}</span>
        <span className="text-muted-foreground"> - </span>
        <span style={{ color: current.color }}>{scores.after.toFixed(2)}</span>
        <span className="text-muted-foreground">) / </span>
        <span className="text-foreground">{scores.before.toFixed(2)}</span>
        <span className="text-muted-foreground"> = </span>
        <span
          className="font-bold"
          style={{
            color: forgetting <= 0 ? "#4ade80" : forgetting < 0.15 ? "#facc15" : "#ef4444",
          }}
        >
          {forgetting.toFixed(3)}
        </span>
      </div>

      {/* Visual meter */}
      <div className="rounded-lg border border-white/[0.06] bg-black/20 p-4 mb-4">
        <div className="font-mono text-[0.65rem] text-muted-foreground mb-2">Forgetting Meter</div>
        <div className="h-6 rounded bg-white/[0.06] overflow-hidden relative">
          {/* Target zone */}
          <div
            className="absolute top-0 h-full"
            style={{
              left: 0,
              width: "15%",
              background: "rgba(74,222,128,0.08)",
              borderRight: "2px dashed rgba(74,222,128,0.3)",
            }}
          />
          {/* Danger zone */}
          <div
            className="absolute top-0 h-full"
            style={{
              left: "15%",
              right: 0,
              background: "rgba(239,68,68,0.05)",
            }}
          />
          {/* Marker */}
          <div
            className="absolute top-0 h-full w-1 rounded transition-all duration-300"
            style={{
              left: `${Math.min(Math.max(forgetting * 100, 0), 100)}%`,
              background: forgetting <= 0 ? "#4ade80" : forgetting < 0.15 ? "#facc15" : "#ef4444",
              boxShadow: `0 0 8px ${forgetting <= 0 ? "#4ade80" : forgetting < 0.15 ? "#facc15" : "#ef4444"}`,
            }}
          />
        </div>
        <div className="flex justify-between mt-1 font-mono text-[0.6rem]">
          <span className="text-green-400">0 (no forgetting)</span>
          <span className="text-green-400/50">0.15 target</span>
          <span className="text-red-400">1.0 (total loss)</span>
        </div>
      </div>

      {/* All strategies comparison */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="text-muted-foreground">
              <th className="text-left py-1.5 px-2 border-b border-white/5">Strategy</th>
              <th className="text-right py-1.5 px-2 border-b border-white/5">Weight Mod</th>
              <th className="text-right py-1.5 px-2 border-b border-white/5">Learn Time</th>
              <th className="text-right py-1.5 px-2 border-b border-white/5">Forgetting</th>
              <th className="text-right py-1.5 px-2 border-b border-white/5">Status</th>
            </tr>
          </thead>
          <tbody>
            {STRATEGIES.map((s, i) => {
              const sc = adjustedScores[s.name];
              const f = computeForgettingRatio(sc.before, sc.after);
              return (
                <tr
                  key={s.name}
                  className={`transition-colors cursor-pointer ${selectedStrategy === i ? "bg-white/[0.04]" : "hover:bg-white/[0.02]"}`}
                  onClick={() => setSelectedStrategy(i)}
                >
                  <td className="py-1.5 px-2 border-b border-white/[0.03]" style={{ color: s.color }}>{s.name}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.modifiesWeights ? "Yes" : "No"}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">{s.learnTime}</td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]" style={{ color: f <= 0 ? "#4ade80" : f < 0.15 ? "#facc15" : "#ef4444" }}>
                    {f.toFixed(3)}
                  </td>
                  <td className="py-1.5 px-2 text-right border-b border-white/[0.03]">
                    {f <= 0 ? "none" : f < 0.15 ? "acceptable" : "too high"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
