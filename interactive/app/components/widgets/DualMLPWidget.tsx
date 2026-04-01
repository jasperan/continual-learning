"use client";

import { useState, useCallback, useMemo } from "react";

/* ------------------------------------------------------------------ */
/*  DualMLP Architecture Widget                                       */
/*  Interactive visualization of frozen + trainable MLP blending      */
/* ------------------------------------------------------------------ */

const HIDDEN_SIZE = 1536;
const INTERMEDIATE_SIZE = 8960;
const NUM_LAYERS = 28;
const MODIFIED_START = 21;
const MODIFIED_END = 28;
const MODIFIED_COUNT = MODIFIED_END - MODIFIED_START;

function fmt(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** Seeded pseudo-random for stable visual values */
function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

function NeuronGrid({
  size,
  color,
  seed,
  label,
  frozen,
}: {
  size: number;
  color: string;
  seed: number;
  label: string;
  frozen?: boolean;
}) {
  const cols = Math.min(size, 16);
  const rows = Math.min(Math.ceil(size / cols), 8);
  const cells = useMemo(() => {
    const arr: number[] = [];
    for (let i = 0; i < rows * cols; i++) arr.push(seeded(seed + i));
    return arr;
  }, [rows, cols, seed]);

  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="font-mono text-xs text-muted-foreground">{label}</span>
      <div
        className="grid gap-[2px]"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {cells.map((v, i) => (
          <div
            key={i}
            className="neuron"
            style={{
              background: color.replace("ALPHA", String(0.2 + v * 0.6)),
              opacity: frozen ? 0.5 : 1,
            }}
          />
        ))}
      </div>
      {frozen && (
        <span className="font-mono text-[0.6rem] text-muted-foreground/50">frozen</span>
      )}
    </div>
  );
}

export default function DualMLPWidget() {
  const [selectedLayer, setSelectedLayer] = useState(24);
  const [alpha, setAlpha] = useState(1.0);
  const [showFormula, setShowFormula] = useState(false);

  // Simulate activation values
  const frozenOut = useMemo(() => seeded(selectedLayer * 100 + 1) * 2 - 1, [selectedLayer]);
  const trainableOut = useMemo(() => seeded(selectedLayer * 100 + 2) * 0.4 - 0.2, [selectedLayer]);
  const blended = frozenOut + (1 - alpha) * trainableOut;

  // Each SwiGLU MLP has 3 projections: gate, up (hidden→intermediate), down (intermediate→hidden)
  const paramsPerMLP = (HIDDEN_SIZE * INTERMEDIATE_SIZE) * 3; // gate_proj + up_proj + down_proj
  const trainableParams = MODIFIED_COUNT * paramsPerMLP;
  const baseModelParams = 1_500_000_000;
  const totalParams = baseModelParams + trainableParams; // base + added trainable MLPs
  const trainablePercent = ((trainableParams / totalParams) * 100).toFixed(1);

  const handleLayerClick = useCallback((idx: number) => {
    setSelectedLayer(idx);
  }, []);

  return (
    <div className="widget-container ch1">
      <div className="widget-label">
        <span className="text-ch1">Interactive</span>{"  ·  "}DualMLP Architecture
      </div>

      {/* ---- Layer selector ---- */}
      <div className="mb-5">
        <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
          Qwen2.5-1.5B Layers (click to inspect)
        </div>
        <div className="flex flex-wrap gap-1">
          {Array.from({ length: NUM_LAYERS }, (_, i) => {
            const isModified = i >= MODIFIED_START && i < MODIFIED_END;
            const isSelected = i === selectedLayer;
            return (
              <button
                key={i}
                onClick={() => handleLayerClick(i)}
                className="font-mono text-[0.65rem] px-1.5 py-0.5 rounded transition-all duration-150"
                style={{
                  background: isSelected
                    ? isModified ? "rgba(249,115,22,0.25)" : "rgba(255,255,255,0.1)"
                    : isModified ? "rgba(249,115,22,0.08)" : "rgba(255,255,255,0.03)",
                  border: `1px solid ${isSelected ? (isModified ? "rgba(249,115,22,0.5)" : "rgba(255,255,255,0.2)") : "rgba(255,255,255,0.05)"}`,
                  color: isModified ? "#f97316" : "var(--muted-foreground)",
                }}
              >
                {i}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-4 mt-2 font-mono text-[0.6rem] text-muted-foreground/60">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: "rgba(249,115,22,0.3)" }} />
            DualMLP (layers {MODIFIED_START}-{MODIFIED_END - 1})
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: "rgba(255,255,255,0.08)" }} />
            Standard (frozen)
          </span>
        </div>
      </div>

      {/* ---- DualMLP diagram ---- */}
      {selectedLayer >= MODIFIED_START && selectedLayer < MODIFIED_END ? (
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-5 mb-5">
          <div className="font-mono text-xs text-ch1 mb-4">
            Layer {selectedLayer} — DualMLP Active
          </div>

          <div className="flex flex-col md:flex-row items-center gap-6 justify-center">
            {/* Input */}
            <div className="text-center">
              <NeuronGrid size={64} color="rgba(148,163,184,ALPHA)" seed={selectedLayer * 10} label="Input x" />
              <div className="font-mono text-[0.6rem] text-muted-foreground/50 mt-1">[1536]</div>
            </div>

            <div className="flow-arrow">&#8594;</div>

            {/* Frozen + Trainable MLPs */}
            <div className="flex flex-col gap-3">
              <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
                <NeuronGrid size={64} color="rgba(96,165,250,ALPHA)" seed={selectedLayer * 20} label="Frozen SwiGLU MLP" frozen />
                <div className="font-mono text-[0.6rem] text-blue-400/60 text-center mt-1">
                  requires_grad = False
                </div>
              </div>
              <div className="rounded-lg border border-orange-500/20 bg-orange-500/5 p-3">
                <NeuronGrid size={64} color="rgba(249,115,22,ALPHA)" seed={selectedLayer * 30} label="Trainable SwiGLU MLP" />
                <div className="font-mono text-[0.6rem] text-orange-400/60 text-center mt-1">
                  init: N(0, 0.001)
                </div>
              </div>
            </div>

            <div className="flow-arrow">&#8594;</div>

            {/* Blending */}
            <div className="text-center">
              <div className="rounded-lg border border-green-500/20 bg-green-500/5 p-3">
                <NeuronGrid size={64} color="rgba(74,222,128,ALPHA)" seed={selectedLayer * 40} label="Blended Output" />
              </div>
            </div>
          </div>

          {/* Formula */}
          <div className="code-block mt-4 text-center text-sm">
            <span className="text-green-400">output</span>
            <span className="text-muted-foreground"> = </span>
            <span className="text-blue-400">frozen_out</span>
            <span className="text-muted-foreground"> + (1 - </span>
            <span className="text-ch1">{alpha.toFixed(2)}</span>
            <span className="text-muted-foreground">) &times; </span>
            <span className="text-orange-400">trainable_out</span>
          </div>

          {/* Alpha slider */}
          <div className="mt-4 flex items-center gap-4">
            <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
              Alpha (&alpha;):
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(alpha * 100)}
              onChange={(e) => setAlpha(Number(e.target.value) / 100)}
              className="flex-1 max-w-[250px]"
            />
            <span className="font-mono text-sm text-ch1 font-semibold w-12 text-right">
              {alpha.toFixed(2)}
            </span>
          </div>

          {/* Blend visualization */}
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-[0.6rem] text-muted-foreground">Frozen influence:</span>
                <span className="font-mono text-xs text-blue-400">{(alpha * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded bg-white/[0.06] overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{
                    width: `${alpha * 100}%`,
                    background: "linear-gradient(90deg, rgba(96,165,250,0.6), rgba(96,165,250,0.3))",
                  }}
                />
              </div>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-[0.6rem] text-muted-foreground">Trainable influence:</span>
                <span className="font-mono text-xs text-ch1">{((1 - alpha) * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded bg-white/[0.06] overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{
                    width: `${(1 - alpha) * 100}%`,
                    background: "linear-gradient(90deg, rgba(249,115,22,0.6), rgba(249,115,22,0.3))",
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-5 mb-5 text-center">
          <div className="font-mono text-xs text-muted-foreground mb-2">
            Layer {selectedLayer} — Standard MLP (Frozen)
          </div>
          <p className="text-sm text-muted-foreground/70">
            This layer uses the original Qwen MLP with all parameters frozen. Only layers {MODIFIED_START}&ndash;{MODIFIED_END - 1} have DualMLP injection.
          </p>
        </div>
      )}

      {/* ---- Stats ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Base Model
          </div>
          <div className="font-mono text-sm text-foreground font-semibold">Qwen2.5-1.5B</div>
          <div className="font-mono text-xs text-muted-foreground">{fmt(baseModelParams)} params</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Modified Layers
          </div>
          <div className="font-mono text-sm text-ch1 font-semibold">{MODIFIED_COUNT} of {NUM_LAYERS}</div>
          <div className="font-mono text-xs text-muted-foreground">layers {MODIFIED_START}&ndash;{MODIFIED_END - 1}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Added Trainable
          </div>
          <div className="font-mono text-sm text-ch1 font-semibold">{fmt(trainableParams)}</div>
          <div className="font-mono text-xs text-muted-foreground">~{trainablePercent}% of total</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Total w/ DualMLP
          </div>
          <div className="font-mono text-sm text-foreground font-semibold">{fmt(totalParams)}</div>
          <div className="font-mono text-xs text-muted-foreground">base + trainable MLPs</div>
        </div>
      </div>
    </div>
  );
}
