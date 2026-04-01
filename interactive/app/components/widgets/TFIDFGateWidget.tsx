"use client";

import { useState, useMemo, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  TF-IDF Gate Widget                                                */
/*  Interactive visualization of neuron-level gradient masking        */
/* ------------------------------------------------------------------ */

const NUM_NEURONS = 128; // Visual grid
const COLS = 16;
const ROWS = NUM_NEURONS / COLS;

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

interface NeuronData {
  tf: number;
  idf: number;
  tfidf: number;
  isGeneral: boolean; // true = general-purpose (masked), false = document-specific
}

function computeNeurons(docSeed: number, threshold: number): NeuronData[] {
  const neurons: NeuronData[] = [];
  for (let i = 0; i < NUM_NEURONS; i++) {
    // IDF: how specialized this neuron is (high = rare = document-specific)
    const idf = seeded(i * 7 + 42);
    // TF: activation frequency in current document
    const tf = seeded(i * 13 + docSeed);
    const tfidf = tf * idf;
    const isGeneral = tfidf < threshold;
    neurons.push({ tf, idf, tfidf, isGeneral });
  }
  return neurons;
}

export default function TFIDFGateWidget() {
  const [threshold, setThreshold] = useState(0.3);
  const [docSeed, setDocSeed] = useState(100);
  const [hoveredNeuron, setHoveredNeuron] = useState<number | null>(null);
  const [showCalibration, setShowCalibration] = useState(false);

  const neurons = useMemo(() => computeNeurons(docSeed, threshold), [docSeed, threshold]);
  const maskedCount = neurons.filter((n) => n.isGeneral).length;
  const passedCount = NUM_NEURONS - maskedCount;

  const docs = [
    { label: "Oracle AI Vector Search", seed: 100 },
    { label: "Quantum Computing Basics", seed: 200 },
    { label: "History of Ancient Rome", seed: 300 },
  ];

  const handleDocChange = useCallback((seed: number) => {
    setDocSeed(seed);
  }, []);

  return (
    <div className="widget-container ch2">
      <div className="widget-label">
        <span className="text-ch2">Interactive</span>{"  ·  "}TF-IDF Gradient Gate
      </div>

      <p className="text-sm text-muted-foreground mb-4">
        Each cell represents a neuron. Green neurons receive gradients (document-specific),
        red neurons are masked (general-purpose, protected from overwriting).
      </p>

      {/* Document selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {docs.map((doc) => (
          <button
            key={doc.seed}
            className={`btn-mono ${docSeed === doc.seed ? "active" : ""}`}
            onClick={() => handleDocChange(doc.seed)}
            style={{
              borderColor: docSeed === doc.seed ? "#22d3ee" : undefined,
              color: docSeed === doc.seed ? "#22d3ee" : undefined,
            }}
          >
            {doc.label}
          </button>
        ))}
      </div>

      {/* Threshold control */}
      <div className="flex items-center gap-4 mb-5">
        <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          Threshold:
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(threshold * 100)}
          onChange={(e) => setThreshold(Number(e.target.value) / 100)}
          className="flex-1 max-w-[250px]"
        />
        <span className="font-mono text-sm text-ch2 font-semibold w-12 text-right">
          {threshold.toFixed(2)}
        </span>
      </div>

      {/* Neuron grid */}
      <div className="rounded-lg border border-white/[0.06] bg-black/20 p-4 mb-4">
        <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-3">
          Neuron Gradient Mask (layer 24, MLP)
        </div>
        <div
          className="grid gap-[3px] mx-auto"
          style={{
            gridTemplateColumns: `repeat(${COLS}, 1fr)`,
            maxWidth: `${COLS * 22}px`,
          }}
        >
          {neurons.map((n, i) => {
            const isHovered = hoveredNeuron === i;
            return (
              <div
                key={i}
                className="rounded-[3px] cursor-pointer transition-all duration-200"
                style={{
                  width: 18,
                  height: 18,
                  background: n.isGeneral
                    ? `rgba(239,68,68,${0.2 + (1 - n.tfidf / threshold) * 0.5})`
                    : `rgba(34,211,238,${0.2 + n.tfidf * 0.6})`,
                  border: isHovered ? "2px solid white" : "1px solid transparent",
                  transform: isHovered ? "scale(1.3)" : "scale(1)",
                }}
                onMouseEnter={() => setHoveredNeuron(i)}
                onMouseLeave={() => setHoveredNeuron(null)}
              />
            );
          })}
        </div>

        {/* Hover tooltip */}
        {hoveredNeuron !== null && (
          <div className="mt-3 p-3 rounded-lg border border-white/[0.06] bg-black/30 animate-fade-in">
            <div className="grid grid-cols-4 gap-3 font-mono text-xs">
              <div>
                <span className="text-muted-foreground block text-[0.6rem]">Neuron</span>
                <span className="text-foreground">#{hoveredNeuron}</span>
              </div>
              <div>
                <span className="text-muted-foreground block text-[0.6rem]">TF</span>
                <span className="text-foreground">{neurons[hoveredNeuron].tf.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-muted-foreground block text-[0.6rem]">IDF</span>
                <span className="text-foreground">{neurons[hoveredNeuron].idf.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-muted-foreground block text-[0.6rem]">TF-IDF</span>
                <span style={{ color: neurons[hoveredNeuron].isGeneral ? "#ef4444" : "#22d3ee" }}>
                  {neurons[hoveredNeuron].tfidf.toFixed(3)}
                  {neurons[hoveredNeuron].isGeneral ? " < " : " >= "}
                  {threshold.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="mt-2 font-mono text-[0.65rem]" style={{ color: neurons[hoveredNeuron].isGeneral ? "#ef4444" : "#22d3ee" }}>
              {neurons[hoveredNeuron].isGeneral
                ? "MASKED — General-purpose neuron, gradient zeroed to prevent forgetting"
                : "PASSED — Document-specific neuron, receives full gradient update"}
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Total Neurons</div>
          <div className="font-mono text-lg font-bold text-foreground">{NUM_NEURONS}</div>
        </div>
        <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch2">Gradient Passed</div>
          <div className="font-mono text-lg font-bold text-ch2">{passedCount}</div>
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            {((passedCount / NUM_NEURONS) * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-red-400">Gradient Masked</div>
          <div className="font-mono text-lg font-bold text-red-400">{maskedCount}</div>
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            {((maskedCount / NUM_NEURONS) * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Threshold</div>
          <div className="font-mono text-lg font-bold text-ch2">{threshold.toFixed(2)}</div>
        </div>
      </div>

      {/* Calibration explanation */}
      <button
        className={`btn-mono mt-4 ${showCalibration ? "active" : ""}`}
        onClick={() => setShowCalibration(!showCalibration)}
      >
        {showCalibration ? "Hide" : "Show"} Calibration Process
      </button>
      {showCalibration && (
        <div className="mt-3 code-block text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">Calibration Phase (runs once before learning):</div>
          <div><span className="text-ch2">1.</span> Run calibration texts through model in eval mode</div>
          <div><span className="text-ch2">2.</span> For each neuron, collect mean activation per sample</div>
          <div><span className="text-ch2">3.</span> Count document frequency: how many docs activate it above mean</div>
          <div><span className="text-ch2">4.</span> Compute IDF = log(num_docs / (1 + doc_freq))</div>
          <div className="mt-2 text-muted-foreground">
            High IDF = neuron is specialized (rare activation) = safe to update{"\n"}
            Low IDF = neuron activates broadly = general-purpose = protect it
          </div>
        </div>
      )}
    </div>
  );
}
