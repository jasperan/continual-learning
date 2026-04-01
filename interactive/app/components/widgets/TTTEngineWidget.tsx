"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  TTT Engine Widget                                                 */
/*  Interactive test-time training simulation with loss tracking      */
/* ------------------------------------------------------------------ */

const MINI_BATCH_SIZE = 32;
const MAX_TOKENS = 4096;

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

function fmt(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** Simulate a training loss curve with exponential decay + noise */
function simulateLosses(numBatches: number, lr: number, seed: number): number[] {
  const losses: number[] = [];
  const initLoss = 3.5 + seeded(seed) * 0.5;
  for (let i = 0; i < numBatches; i++) {
    const t = i / numBatches;
    const decay = Math.exp(-4 * t * (lr / 1e-5));
    const noise = (seeded(seed + i * 17) - 0.5) * 0.15 * (1 - t);
    losses.push(Math.max(0.8, initLoss * decay + 0.8 * (1 - decay) + noise));
  }
  return losses;
}

const CHART_W = 700;
const CHART_H = 200;
const PAD = { top: 15, right: 15, bottom: 30, left: 45 };
const PW = CHART_W - PAD.left - PAD.right;
const PH = CHART_H - PAD.top - PAD.bottom;

const DOCUMENTS = [
  { name: "Oracle AI Vector Search Docs", tokens: 2847, seed: 42 },
  { name: "Quantum Computing Primer", tokens: 1523, seed: 99 },
  { name: "Climate Change Research Paper", tokens: 3891, seed: 156 },
];

export default function TTTEngineWidget() {
  const [mounted, setMounted] = useState(false);
  const [docIdx, setDocIdx] = useState(0);
  const [learningRate, setLearningRate] = useState(1e-5);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => setMounted(true), []);

  const doc = DOCUMENTS[docIdx];
  const numBatches = Math.ceil(doc.tokens / MINI_BATCH_SIZE);
  const losses = useMemo(() => simulateLosses(numBatches, learningRate, doc.seed), [numBatches, learningRate, doc.seed]);

  // Alpha decay simulation
  const alphas = useMemo(() => {
    const vals: number[] = [];
    let a = 1.0;
    for (let i = 0; i < numBatches; i++) {
      vals.push(a);
      a = Math.max(0.3, a - 0.01);
    }
    return vals;
  }, [numBatches]);

  const xScale = useCallback((i: number) => PAD.left + (i / (numBatches - 1)) * PW, [numBatches]);
  const yScale = useCallback((v: number) => PAD.top + PH - ((v - 0) / 4.5) * PH, []);

  // Autoplay
  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCurrentBatch((prev) => {
          if (prev >= numBatches - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 80);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, numBatches]);

  const reset = useCallback(() => {
    setPlaying(false);
    setCurrentBatch(0);
  }, []);

  const currentLoss = currentBatch < losses.length ? losses[currentBatch] : losses[losses.length - 1];
  const currentAlpha = currentBatch < alphas.length ? alphas[currentBatch] : alphas[alphas.length - 1];
  const tokensProcessed = Math.min(currentBatch * MINI_BATCH_SIZE, doc.tokens);
  const progress = (tokensProcessed / doc.tokens) * 100;

  const pathD = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i <= Math.min(currentBatch, numBatches - 1); i++) {
      pts.push(`${i === 0 ? "M" : "L"}${xScale(i)},${yScale(losses[i])}`);
    }
    return pts.join(" ");
  }, [currentBatch, numBatches, losses, xScale, yScale]);

  return (
    <div className="widget-container ch3">
      <div className="widget-label">
        <span className="text-ch3">Interactive</span>{"  ·  "}Test-Time Training Engine
      </div>

      {/* Document selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {DOCUMENTS.map((d, i) => (
          <button
            key={i}
            className={`btn-mono ${docIdx === i ? "active" : ""}`}
            onClick={() => { setDocIdx(i); reset(); }}
            style={{
              borderColor: docIdx === i ? "#4ade80" : undefined,
              color: docIdx === i ? "#4ade80" : undefined,
            }}
          >
            {d.name} ({fmt(d.tokens)} tokens)
          </button>
        ))}
      </div>

      {/* Learning rate control */}
      <div className="flex items-center gap-4 mb-4">
        <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          Learning rate:
        </label>
        <input
          type="range"
          min={0}
          max={4}
          step={1}
          value={[1e-6, 5e-6, 1e-5, 5e-5, 1e-4].indexOf(learningRate)}
          onChange={(e) => {
            setLearningRate([1e-6, 5e-6, 1e-5, 5e-5, 1e-4][Number(e.target.value)]);
            reset();
          }}
          className="max-w-[200px]"
        />
        <span className="font-mono text-xs text-ch3">{learningRate.toExponential(0)}</span>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 mb-4">
        <button className="btn-mono" onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : currentBatch >= numBatches - 1 ? "Done" : "Train"}
        </button>
        <button className="btn-mono" onClick={() => setCurrentBatch((b) => Math.min(b + 1, numBatches - 1))} disabled={playing}>
          Step
        </button>
        <button className="btn-mono" onClick={reset}>Reset</button>
        <div className="flex-1" />
        <span className="font-mono text-xs text-muted-foreground">
          Batch {currentBatch + 1}/{numBatches}
        </span>
      </div>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="h-2 rounded bg-white/[0.06] overflow-hidden">
          <div
            className="h-full rounded transition-all duration-100"
            style={{ width: `${progress}%`, background: "linear-gradient(90deg, rgba(74,222,128,0.4), rgba(74,222,128,0.7))" }}
          />
        </div>
        <div className="flex justify-between mt-1 font-mono text-[0.6rem] text-muted-foreground">
          <span>{fmt(tokensProcessed)} tokens</span>
          <span>{fmt(doc.tokens)} total</span>
        </div>
      </div>

      {/* Loss chart */}
      <div className="rounded-lg p-2 mb-4" style={{ background: "rgba(0,0,0,0.25)", border: "1px solid var(--border)" }}>
        {mounted && (
          <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="w-full" style={{ maxHeight: "240px" }}>
            {[0, 1, 2, 3, 4].map((v) => (
              <g key={v}>
                <line x1={PAD.left} y1={yScale(v)} x2={CHART_W - PAD.right} y2={yScale(v)} stroke="rgba(255,255,255,0.06)" strokeDasharray="4,4" />
                <text x={PAD.left - 8} y={yScale(v) + 4} textAnchor="end" fill="#71717a" fontSize={10} fontFamily="var(--font-mono)">{v}</text>
              </g>
            ))}
            <text x={CHART_W / 2} y={CHART_H - 4} textAnchor="middle" fill="#52525b" fontSize={10} fontFamily="var(--font-mono)">mini-batch</text>

            {currentBatch > 0 && <path d={pathD} fill="none" stroke="#4ade80" strokeWidth={2} strokeLinecap="round" opacity={0.85} />}

            {currentBatch > 0 && (
              <circle cx={xScale(currentBatch)} cy={yScale(currentLoss)} r={4} fill="#4ade80" stroke="#0a0b0f" strokeWidth={2} />
            )}
          </svg>
        )}
      </div>

      {/* Live stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Loss</div>
          <div className="font-mono text-lg font-bold text-ch3">{currentLoss.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Alpha</div>
          <div className="font-mono text-lg font-bold text-ch1">{currentAlpha.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Tokens</div>
          <div className="font-mono text-lg font-bold text-foreground">{fmt(tokensProcessed)}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Grad Steps</div>
          <div className="font-mono text-lg font-bold text-foreground">{currentBatch}</div>
        </div>
      </div>

      {/* Pipeline description */}
      <div className="code-block mt-4 text-xs">
        <div className="text-muted-foreground mb-1">Per mini-batch pipeline:</div>
        <div><span className="text-ch3">1.</span> Forward: model(input_ids, labels) &#8594; loss</div>
        <div><span className="text-ch3">2.</span> Backward: loss.backward() &#8594; gradients</div>
        <div><span className="text-ch3">3.</span> Mask: apply TF-IDF gate to trainable MLP gradients</div>
        <div><span className="text-ch3">4.</span> Update: Adam optimizer step (trainable params only)</div>
        <div><span className="text-ch3">5.</span> Decay: alpha -= {0.01} (floor at {0.3})</div>
      </div>
    </div>
  );
}
