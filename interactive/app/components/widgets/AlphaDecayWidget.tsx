"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Alpha Decay Widget                                                */
/*  Interactive visualization of alpha schedule during learning       */
/* ------------------------------------------------------------------ */

const CHART_W = 700;
const CHART_H = 240;
const PAD = { top: 20, right: 20, bottom: 35, left: 50 };
const PW = CHART_W - PAD.left - PAD.right;
const PH = CHART_H - PAD.top - PAD.bottom;

function computeAlphaSchedule(
  initial: number,
  decayRate: number,
  minValue: number,
  numBatches: number,
): number[] {
  const values: number[] = [];
  let alpha = initial;
  for (let i = 0; i < numBatches; i++) {
    values.push(alpha);
    alpha = Math.max(minValue, alpha - decayRate);
  }
  return values;
}

export default function AlphaDecayWidget() {
  const [mounted, setMounted] = useState(false);
  const [initial, setInitial] = useState(1.0);
  const [decayRate, setDecayRate] = useState(0.01);
  const [minValue, setMinValue] = useState(0.3);
  const [hoverBatch, setHoverBatch] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => setMounted(true), []);

  const numBatches = Math.ceil((initial - minValue) / decayRate) + 20;
  const schedule = useMemo(
    () => computeAlphaSchedule(initial, decayRate, minValue, numBatches),
    [initial, decayRate, minValue, numBatches],
  );

  const xScale = useCallback(
    (i: number) => PAD.left + (i / (numBatches - 1)) * PW,
    [numBatches],
  );
  const yScale = useCallback(
    (v: number) => PAD.top + PH - ((v - 0) / 1) * PH,
    [],
  );

  const handleMouse = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const x = e.clientX - rect.left;
      const ratio = (x - PAD.left) / PW;
      if (ratio < 0 || ratio > 1) { setHoverBatch(null); return; }
      setHoverBatch(Math.round(ratio * (numBatches - 1)));
    },
    [numBatches],
  );

  // Build SVG path
  const pathD = useMemo(() => {
    return schedule
      .map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(v)}`)
      .join(" ");
  }, [schedule, xScale, yScale]);

  // Frozen vs trainable influence at hover point
  const hoverAlpha = hoverBatch !== null ? schedule[Math.min(hoverBatch, schedule.length - 1)] : initial;
  const frozenPct = (hoverAlpha * 100).toFixed(1);
  const trainablePct = ((1 - hoverAlpha) * 100).toFixed(1);

  // Convergence batch
  const convergeBatch = schedule.findIndex((v) => v <= minValue);

  return (
    <div className="widget-container ch1">
      <div className="widget-label">
        <span className="text-ch1">Interactive</span>{"  ·  "}Alpha Decay Schedule
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
        <div>
          <label className="font-mono text-[0.65rem] text-muted-foreground block mb-1">
            Initial &alpha;
          </label>
          <input
            type="range"
            min={50}
            max={100}
            value={Math.round(initial * 100)}
            onChange={(e) => setInitial(Number(e.target.value) / 100)}
            className="w-full"
          />
          <span className="font-mono text-xs text-ch1">{initial.toFixed(2)}</span>
        </div>
        <div>
          <label className="font-mono text-[0.65rem] text-muted-foreground block mb-1">
            Decay rate
          </label>
          <input
            type="range"
            min={1}
            max={50}
            value={Math.round(decayRate * 1000)}
            onChange={(e) => setDecayRate(Number(e.target.value) / 1000)}
            className="w-full"
          />
          <span className="font-mono text-xs text-ch1">{decayRate.toFixed(3)}</span>
        </div>
        <div>
          <label className="font-mono text-[0.65rem] text-muted-foreground block mb-1">
            Min &alpha;
          </label>
          <input
            type="range"
            min={0}
            max={90}
            value={Math.round(minValue * 100)}
            onChange={(e) => setMinValue(Number(e.target.value) / 100)}
            className="w-full"
          />
          <span className="font-mono text-xs text-ch1">{minValue.toFixed(2)}</span>
        </div>
      </div>

      {/* Chart */}
      <div className="rounded-lg p-2" style={{ background: "rgba(0,0,0,0.25)", border: "1px solid var(--border)" }}>
        {mounted && (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            className="w-full"
            style={{ maxHeight: "280px" }}
            onMouseMove={handleMouse}
            onMouseLeave={() => setHoverBatch(null)}
          >
            {/* Grid */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
              <g key={v}>
                <line x1={PAD.left} y1={yScale(v)} x2={CHART_W - PAD.right} y2={yScale(v)} stroke="rgba(255,255,255,0.06)" strokeDasharray="4,4" />
                <text x={PAD.left - 8} y={yScale(v) + 4} textAnchor="end" fill="#71717a" fontSize={10} fontFamily="var(--font-mono)">
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* X axis */}
            <text x={CHART_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#52525b" fontSize={10} fontFamily="var(--font-mono)">
              mini-batches processed
            </text>
            <text x={12} y={CHART_H / 2} textAnchor="middle" fill="#52525b" fontSize={10} fontFamily="var(--font-mono)" transform={`rotate(-90, 12, ${CHART_H / 2})`}>
              alpha (&alpha;)
            </text>

            {/* Min value line */}
            <line x1={PAD.left} y1={yScale(minValue)} x2={CHART_W - PAD.right} y2={yScale(minValue)} stroke="rgba(249,115,22,0.3)" strokeDasharray="6,3" />
            <text x={CHART_W - PAD.right + 5} y={yScale(minValue) + 4} fill="#f97316" fontSize={9} fontFamily="var(--font-mono)">
              min
            </text>

            {/* Alpha curve */}
            <path d={pathD} fill="none" stroke="#f97316" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} />

            {/* Convergence marker */}
            {convergeBatch >= 0 && (
              <g>
                <line x1={xScale(convergeBatch)} y1={PAD.top} x2={xScale(convergeBatch)} y2={CHART_H - PAD.bottom} stroke="rgba(249,115,22,0.2)" strokeDasharray="3,3" />
                <text x={xScale(convergeBatch)} y={PAD.top - 4} textAnchor="middle" fill="#f97316" fontSize={9} fontFamily="var(--font-mono)">
                  converged @ {convergeBatch}
                </text>
              </g>
            )}

            {/* Hover */}
            {hoverBatch !== null && (
              <g>
                <line x1={xScale(hoverBatch)} y1={PAD.top} x2={xScale(hoverBatch)} y2={CHART_H - PAD.bottom} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
                <circle cx={xScale(hoverBatch)} cy={yScale(hoverAlpha)} r={4} fill="#f97316" stroke="#0a0b0f" strokeWidth={2} />
                <text x={xScale(hoverBatch) + 8} y={yScale(hoverAlpha) - 8} fill="#f97316" fontSize={11} fontFamily="var(--font-mono)" fontWeight="bold">
                  {hoverAlpha.toFixed(3)}
                </text>
              </g>
            )}
          </svg>
        )}
      </div>

      {/* Influence bars */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="flex justify-between font-mono text-[0.65rem] text-muted-foreground mb-1">
            <span>Frozen MLP influence</span>
            <span className="text-blue-400">{frozenPct}%</span>
          </div>
          <div className="h-3 rounded bg-white/[0.06] overflow-hidden">
            <div className="h-full rounded transition-all duration-200" style={{ width: `${frozenPct}%`, background: "rgba(96,165,250,0.5)" }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between font-mono text-[0.65rem] text-muted-foreground mb-1">
            <span>Trainable MLP influence</span>
            <span className="text-ch1">{trainablePct}%</span>
          </div>
          <div className="h-3 rounded bg-white/[0.06] overflow-hidden">
            <div className="h-full rounded transition-all duration-200" style={{ width: `${trainablePct}%`, background: "rgba(249,115,22,0.5)" }} />
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        Alpha decays linearly by {decayRate} per mini-batch. After {convergeBatch >= 0 ? convergeBatch : "many"} batches,
        the trainable MLP contributes {((1 - minValue) * 100).toFixed(0)}% of the output.
        The frozen MLP always provides a stable baseline, preventing catastrophic forgetting.
      </p>
    </div>
  );
}
