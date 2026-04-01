"use client";

import { useState, useMemo, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Hypernetwork Widget                                               */
/*  Interactive visualization of Perceiver hypernetwork → LoRA        */
/* ------------------------------------------------------------------ */

const LORA_RANK = 8;
const HIDDEN_DIM = 1536;
const INTERMEDIATE_DIM = 8960;
const NUM_TARGET_LAYERS = 7;
const GRID = 8;

function fmt(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

function MatrixVis({
  rows, cols, seed, color, label, dimLabel,
}: {
  rows: number; cols: number; seed: number; color: string; label: string; dimLabel: string;
}) {
  const displayR = Math.min(rows, GRID);
  const displayC = Math.min(cols, GRID);
  const cells = useMemo(() => {
    const arr: number[] = [];
    for (let i = 0; i < displayR * displayC; i++) arr.push(seeded(seed + i));
    return arr;
  }, [displayR, displayC, seed]);
  const sz = Math.max(8, Math.min(24, 200 / Math.max(displayR, displayC)));

  return (
    <div className="flex flex-col items-center gap-1">
      <span className="font-mono text-[0.6rem] text-muted-foreground">{label}</span>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${displayC}, 1fr)` }}>
        {cells.map((v, i) => (
          <div
            key={i}
            className="rounded-[2px]"
            style={{ width: sz, height: sz, background: color.replace("ALPHA", String(0.15 + v * 0.65)) }}
          />
        ))}
      </div>
      <span className="font-mono text-[0.55rem] text-muted-foreground/60">{dimLabel}</span>
    </div>
  );
}

interface ChunkInfo {
  text: string;
  hash: string;
  seed: number;
}

const SAMPLE_CHUNKS: ChunkInfo[] = [
  { text: "Oracle AI Vector Search enables similarity-based queries using embedding vectors stored alongside relational data...", hash: "a3f7c2e1", seed: 1000 },
  { text: "The HNSW index structure organizes vectors into hierarchical layers for approximate nearest neighbor search...", hash: "b8d4e5f9", seed: 2000 },
  { text: "Embedding models transform text, images, and other data into dense numerical representations that capture semantic meaning...", hash: "c1a9f3b7", seed: 3000 },
];

export default function HypernetworkWidget() {
  const [numChunks, setNumChunks] = useState(1);
  const [activeStage, setActiveStage] = useState<number>(0);
  const [showArchitecture, setShowArchitecture] = useState(false);

  const effectiveRank = LORA_RANK * numChunks;
  const paramsPerLayer = HIDDEN_DIM * effectiveRank + effectiveRank * INTERMEDIATE_DIM;
  const totalParams = paramsPerLayer * NUM_TARGET_LAYERS;

  const stages = [
    { name: "Chunk", desc: "Split document into ~1024-word chunks", icon: "1" },
    { name: "Hash", desc: "SHA-256 seed for deterministic generation", icon: "2" },
    { name: "Generate", desc: "Hypernetwork produces LoRA A & B matrices", icon: "3" },
    { name: "Concat", desc: "Multi-chunk: concatenate along rank axis", icon: "4" },
    { name: "Inject", desc: "Add delta W' = W + (A @ B).T to MLP layers", icon: "5" },
  ];

  const handleStageClick = useCallback((idx: number) => {
    setActiveStage(idx);
  }, []);

  return (
    <div className="widget-container ch7">
      <div className="widget-label">
        <span className="text-ch7">Interactive</span>{"  ·  "}Perceiver Hypernetwork Pipeline
      </div>

      {/* Pipeline stages */}
      <div className="flex flex-wrap gap-2 mb-5">
        {stages.map((s, i) => (
          <button
            key={i}
            className={`pipeline-stage flex items-center gap-2 ${activeStage === i ? "active" : ""}`}
            onClick={() => handleStageClick(i)}
            style={{
              borderColor: activeStage === i ? "#fb923c" : undefined,
            }}
          >
            <span className="font-mono text-xs font-bold text-ch7">{s.icon}</span>
            <div className="text-left">
              <div className="font-mono text-xs text-foreground">{s.name}</div>
              <div className="font-mono text-[0.6rem] text-muted-foreground">{s.desc}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Chunk count control */}
      <div className="flex items-center gap-4 mb-5">
        <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          Document chunks:
        </label>
        <input
          type="range"
          min={1}
          max={3}
          value={numChunks}
          onChange={(e) => setNumChunks(Number(e.target.value))}
          className="max-w-[150px]"
        />
        <span className="font-mono text-sm text-ch7 font-semibold">{numChunks}</span>
        <span className="font-mono text-[0.65rem] text-muted-foreground">
          (effective rank: {LORA_RANK} &times; {numChunks} = {effectiveRank})
        </span>
      </div>

      {/* Stage detail */}
      <div className="rounded-lg border border-white/[0.06] bg-black/20 p-5 mb-5">
        {activeStage === 0 && (
          <div className="animate-fade-in">
            <div className="font-mono text-xs text-ch7 mb-3">Stage 1: Document Chunking</div>
            <div className="space-y-2">
              {SAMPLE_CHUNKS.slice(0, numChunks).map((chunk, i) => (
                <div key={i} className="p-3 rounded border border-white/[0.06] bg-black/20">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-[0.6rem] text-ch7">Chunk {i + 1}</span>
                    <span className="font-mono text-[0.55rem] text-muted-foreground">~1024 words</span>
                  </div>
                  <p className="font-mono text-xs text-muted-foreground/80 line-clamp-2">{chunk.text}</p>
                </div>
              ))}
            </div>
            <div className="code-block mt-3 text-xs">
              <span className="text-muted-foreground">mode = &quot;doc&quot;: </span>
              <span className="text-ch7">split by ~1024 words</span>
              <br />
              <span className="text-muted-foreground">mode = &quot;text&quot;: </span>
              <span className="text-ch7">single chunk (task description)</span>
            </div>
          </div>
        )}

        {activeStage === 1 && (
          <div className="animate-fade-in">
            <div className="font-mono text-xs text-ch7 mb-3">Stage 2: Deterministic Hashing</div>
            <div className="space-y-2">
              {SAMPLE_CHUNKS.slice(0, numChunks).map((chunk, i) => (
                <div key={i} className="flex items-center gap-3 p-2 rounded border border-white/[0.06]">
                  <span className="font-mono text-[0.6rem] text-muted-foreground">Chunk {i + 1}</span>
                  <span className="flow-arrow">&#8594;</span>
                  <span className="font-mono text-xs text-muted-foreground">SHA-256</span>
                  <span className="flow-arrow">&#8594;</span>
                  <span className="font-mono text-xs text-ch7">seed = 0x{chunk.hash}</span>
                </div>
              ))}
            </div>
            <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
              Same document always produces the same LoRA matrices. Deterministic seeding ensures reproducibility.
            </p>
          </div>
        )}

        {activeStage === 2 && (
          <div className="animate-fade-in">
            <div className="font-mono text-xs text-ch7 mb-3">Stage 3: LoRA Matrix Generation</div>
            <div className="flex flex-wrap items-center gap-4 justify-center">
              {Array.from({ length: numChunks }, (_, ci) => (
                <div key={ci} className="flex items-center gap-2">
                  <MatrixVis rows={GRID} cols={2} seed={SAMPLE_CHUNKS[ci].seed} color="rgba(251,146,60,ALPHA)" label={`A_${ci}`} dimLabel={`${HIDDEN_DIM}×${LORA_RANK}`} />
                  <span className="font-mono text-muted-foreground">&times;</span>
                  <MatrixVis rows={2} cols={GRID} seed={SAMPLE_CHUNKS[ci].seed + 500} color="rgba(251,146,60,ALPHA)" label={`B_${ci}`} dimLabel={`${LORA_RANK}×${INTERMEDIATE_DIM}`} />
                </div>
              ))}
            </div>
            <div className="code-block mt-3 text-xs text-center">
              <span className="text-muted-foreground">Per chunk: </span>
              <span className="text-ch7">A</span>
              <span className="text-muted-foreground">[{HIDDEN_DIM}&times;{LORA_RANK}] &times; </span>
              <span className="text-ch7">B</span>
              <span className="text-muted-foreground">[{LORA_RANK}&times;{INTERMEDIATE_DIM}] = </span>
              <span className="text-ch7">&Delta;W</span>
              <span className="text-muted-foreground">[{HIDDEN_DIM}&times;{INTERMEDIATE_DIM}]</span>
            </div>
          </div>
        )}

        {activeStage === 3 && (
          <div className="animate-fade-in">
            <div className="font-mono text-xs text-ch7 mb-3">Stage 4: Rank Concatenation ({numChunks} chunk{numChunks > 1 ? "s" : ""})</div>
            {numChunks === 1 ? (
              <div className="text-center">
                <p className="font-mono text-xs text-muted-foreground mb-3">Single chunk &mdash; no concatenation needed.</p>
                <div className="flex items-center gap-2 justify-center">
                  <MatrixVis rows={GRID} cols={2} seed={1000} color="rgba(251,146,60,ALPHA)" label="A" dimLabel={`${HIDDEN_DIM}×${LORA_RANK}`} />
                  <MatrixVis rows={2} cols={GRID} seed={1500} color="rgba(251,146,60,ALPHA)" label="B" dimLabel={`${LORA_RANK}×${INTERMEDIATE_DIM}`} />
                </div>
              </div>
            ) : (
              <div className="text-center">
                <p className="font-mono text-xs text-muted-foreground mb-3">
                  Concatenate along rank axis: effective rank = {LORA_RANK} &times; {numChunks} = {effectiveRank}
                </p>
                <div className="flex items-center gap-3 justify-center">
                  <MatrixVis rows={GRID} cols={numChunks * 2} seed={1000} color="rgba(251,146,60,ALPHA)" label="A_concat" dimLabel={`${HIDDEN_DIM}×${effectiveRank}`} />
                  <span className="font-mono text-muted-foreground">&times;</span>
                  <MatrixVis rows={numChunks * 2} cols={GRID} seed={2000} color="rgba(251,146,60,ALPHA)" label="B_concat" dimLabel={`${effectiveRank}×${INTERMEDIATE_DIM}`} />
                </div>
              </div>
            )}
          </div>
        )}

        {activeStage === 4 && (
          <div className="animate-fade-in">
            <div className="font-mono text-xs text-ch7 mb-3">Stage 5: Weight Injection</div>
            <div className="flex items-center gap-3 justify-center flex-wrap mb-3">
              <MatrixVis rows={GRID} cols={GRID} seed={42} color="rgba(96,165,250,ALPHA)" label="W_original" dimLabel={`${HIDDEN_DIM}×${INTERMEDIATE_DIM}`} />
              <span className="font-mono text-lg text-muted-foreground">+</span>
              <div className="flex items-center gap-1">
                <span className="font-mono text-xs text-muted-foreground">(</span>
                <MatrixVis rows={GRID} cols={numChunks * 2} seed={1000} color="rgba(251,146,60,ALPHA)" label="A" dimLabel={`${HIDDEN_DIM}×${effectiveRank}`} />
                <span className="font-mono text-xs text-muted-foreground">@</span>
                <MatrixVis rows={numChunks * 2} cols={GRID} seed={2000} color="rgba(251,146,60,ALPHA)" label="B" dimLabel={`${effectiveRank}×${INTERMEDIATE_DIM}`} />
                <span className="font-mono text-xs text-muted-foreground">).T</span>
              </div>
              <span className="font-mono text-lg text-muted-foreground">=</span>
              <MatrixVis rows={GRID} cols={GRID} seed={500} color="rgba(74,222,128,ALPHA)" label="W_modified" dimLabel={`${HIDDEN_DIM}×${INTERMEDIATE_DIM}`} />
            </div>
            <div className="code-block text-xs text-center">
              <span className="text-green-400">W&apos;</span>
              <span className="text-muted-foreground"> = </span>
              <span className="text-blue-400">W</span>
              <span className="text-muted-foreground"> + (</span>
              <span className="text-ch7">A @ B</span>
              <span className="text-muted-foreground">).T</span>
              <span className="text-muted-foreground/50 ml-4">Applied to {NUM_TARGET_LAYERS} MLP layers</span>
            </div>
          </div>
        )}
      </div>

      {/* Architecture toggle */}
      <button
        className={`btn-mono mb-4 ${showArchitecture ? "active" : ""}`}
        onClick={() => setShowArchitecture(!showArchitecture)}
      >
        {showArchitecture ? "Hide" : "Show"} Perceiver Architecture
      </button>
      {showArchitecture && (
        <div className="code-block mb-4 text-xs animate-fade-in">
          <div className="text-ch7 mb-2">PerceiverHypernetwork Architecture:</div>
          <div className="text-muted-foreground">
            <div>Input: frozen model activations [{`batch, seq_len, ${HIDDEN_DIM}`}]</div>
            <div>&nbsp;&nbsp;&#8594; Input projection &#8594; latent space</div>
            <div>&nbsp;&nbsp;&#8594; Learned latent queries [{NUM_TARGET_LAYERS * 2}, latent_dim]</div>
            <div>&nbsp;&nbsp;&#8594; 8 cross-attention blocks (4 heads each)</div>
            <div>&nbsp;&nbsp;&#8594; Output head A: latent &#8594; [{HIDDEN_DIM} &times; {LORA_RANK}]</div>
            <div>&nbsp;&nbsp;&#8594; Output head B: latent &#8594; [{LORA_RANK} &times; {INTERMEDIATE_DIM}]</div>
            <div className="mt-2 text-ch7">Result: complete LoRA for {NUM_TARGET_LAYERS} layers in one forward pass</div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Base Rank</div>
          <div className="font-mono text-lg font-bold text-ch7">{LORA_RANK}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Effective Rank</div>
          <div className="font-mono text-lg font-bold text-ch7">{effectiveRank}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">LoRA Params</div>
          <div className="font-mono text-lg font-bold text-foreground">{fmt(totalParams)}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">Target Layers</div>
          <div className="font-mono text-lg font-bold text-foreground">{NUM_TARGET_LAYERS}</div>
        </div>
      </div>
    </div>
  );
}
