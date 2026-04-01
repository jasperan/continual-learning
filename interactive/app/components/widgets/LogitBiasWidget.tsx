"use client";

import { useState, useMemo, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Logit Bias Widget                                                  */
/*  Interactive visualization of logit biasing in JitRL MVP           */
/* ------------------------------------------------------------------ */

/** Seeded pseudo-random for stable values. */
function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Token data ---- */

interface TokenEntry {
  token: string;
  originalLogit: number;
  chunkCount: number; // how many times this token appears in retrieved chunks
  isChunkToken: boolean; // whether this token appears in retrieved chunks at all
}

/** Build a stable set of token logit data using seeded random */
function buildTokenData(seed: number): TokenEntry[] {
  // Tokens that appear in retrieved chunks (will get boosted)
  const chunkTokens: { token: string; count: number }[] = [
    { token: "vector", count: 7 },
    { token: "similarity", count: 5 },
    { token: "embedding", count: 6 },
    { token: "search", count: 4 },
    { token: "cosine", count: 3 },
    { token: "index", count: 4 },
    { token: "HNSW", count: 2 },
    { token: "Oracle", count: 5 },
    { token: "ANN", count: 3 },
    { token: "query", count: 4 },
  ];

  // Tokens that do NOT appear in chunks (no bias)
  const otherTokens = [
    "the", "is", "a", "function", "while", "return",
    "class", "import", "model", "training",
  ];

  const maxCount = Math.max(...chunkTokens.map((t) => t.count));

  const entries: TokenEntry[] = [];

  // Chunk tokens
  for (let i = 0; i < chunkTokens.length; i++) {
    const logit = Math.round((seeded(seed + i * 7) * 6 - 2) * 100) / 100;
    entries.push({
      token: chunkTokens[i].token,
      originalLogit: logit,
      chunkCount: chunkTokens[i].count,
      isChunkToken: true,
    });
  }

  // Non-chunk tokens
  for (let i = 0; i < otherTokens.length; i++) {
    const logit =
      Math.round((seeded(seed + 100 + i * 11) * 6 - 2) * 100) / 100;
    entries.push({
      token: otherTokens[i],
      originalLogit: logit,
      chunkCount: 0,
      isChunkToken: false,
    });
  }

  return entries;
}

const SEED = 42;
const MAX_CHUNK_COUNT = 7; // max count across chunk tokens

/* ---- Chart constants ---- */

const BAR_H = 22;
const BAR_GAP = 4;
const LABEL_W = 90;
const LOGIT_LABEL_W = 50;
const MAX_BAR_W = 200;
const CHART_PAD = 12;

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function LogitBiasWidget() {
  const [biasStrength, setBiasStrength] = useState(2.0);
  const [showFormula, setShowFormula] = useState(false);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);

  const tokens = useMemo(() => buildTokenData(SEED), []);

  /** Compute biased logit for a token */
  const computeBias = useCallback(
    (entry: TokenEntry): number => {
      if (!entry.isChunkToken) return 0;
      return (
        Math.round(
          biasStrength * (entry.chunkCount / MAX_CHUNK_COUNT) * 100,
        ) / 100
      );
    },
    [biasStrength],
  );

  /** Sorted tokens: chunk tokens first (by count desc), then others */
  const sortedTokens = useMemo(() => {
    return [...tokens].sort((a, b) => {
      if (a.isChunkToken && !b.isChunkToken) return -1;
      if (!a.isChunkToken && b.isChunkToken) return 1;
      return b.chunkCount - a.chunkCount;
    });
  }, [tokens]);

  /** Find max absolute logit (original or biased) for scaling */
  const maxAbsLogit = useMemo(() => {
    let max = 0;
    for (const t of sortedTokens) {
      const bias = computeBias(t);
      max = Math.max(max, Math.abs(t.originalLogit), Math.abs(t.originalLogit + bias));
    }
    return Math.max(max, 1);
  }, [sortedTokens, computeBias]);

  const totalH =
    sortedTokens.length * (BAR_H + BAR_GAP) - BAR_GAP + CHART_PAD * 2;
  const chartW = LABEL_W + LOGIT_LABEL_W + MAX_BAR_W * 2 + LOGIT_LABEL_W + 40;

  /** Map a logit value to a bar width and direction */
  function logitBar(
    value: number,
    color: string,
    yPos: number,
    opacity: number = 1,
  ) {
    const barW = (Math.abs(value) / maxAbsLogit) * MAX_BAR_W;
    const xCenter = LABEL_W + LOGIT_LABEL_W + MAX_BAR_W;
    const xStart = value >= 0 ? xCenter : xCenter - barW;
    return (
      <rect
        x={xStart}
        y={yPos}
        width={Math.max(1, barW)}
        height={BAR_H * 0.65}
        rx={3}
        fill={color}
        opacity={opacity}
      />
    );
  }

  return (
    <div className="widget-container ch4">
      <div className="widget-label">
        <span className="text-ch4">Interactive</span>{"  ·  "}JitRL MVP Logit Biasing
      </div>

      {/* ---- Formula display ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">biased_logit</span>
        <span className="text-ch4"> = </span>
        <span className="text-muted-foreground">original_logit</span>
        <span className="text-ch4"> + </span>
        <span className="text-muted-foreground">strength</span>
        <span className="text-ch4"> &times; </span>
        <span className="text-muted-foreground">(count / max_count)</span>
      </div>

      {/* ---- Bias strength control ---- */}
      <div className="flex items-center gap-4 mb-5">
        <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          bias_strength:
        </label>
        <input
          type="range"
          min={0}
          max={50}
          value={Math.round(biasStrength * 10)}
          onChange={(e) => setBiasStrength(Number(e.target.value) / 10)}
          className="flex-1 max-w-[250px]"
        />
        <span className="font-mono text-sm text-ch4 font-semibold w-12 text-right">
          {biasStrength.toFixed(1)}
        </span>
      </div>

      {/* ---- Side-by-side chart ---- */}
      <div className="rounded-lg p-2 overflow-x-auto" style={{ background: "rgba(0,0,0,0.25)", border: "1px solid var(--border)" }}>
        <svg
          viewBox={`0 0 ${chartW} ${totalH}`}
          className="w-full"
          style={{ minWidth: "600px", maxHeight: `${totalH + 20}px` }}
        >
          {/* Center axis */}
          <line
            x1={LABEL_W + LOGIT_LABEL_W + MAX_BAR_W}
            y1={CHART_PAD}
            x2={LABEL_W + LOGIT_LABEL_W + MAX_BAR_W}
            y2={totalH - CHART_PAD}
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={1}
          />

          {/* Column labels */}
          <text
            x={LABEL_W + LOGIT_LABEL_W + MAX_BAR_W - MAX_BAR_W / 2}
            y={CHART_PAD - 2}
            textAnchor="middle"
            fill="#71717a"
            fontSize={9}
            fontFamily="var(--font-mono)"
          >
            Original Logits
          </text>
          <text
            x={LABEL_W + LOGIT_LABEL_W + MAX_BAR_W + MAX_BAR_W / 2}
            y={CHART_PAD - 2}
            textAnchor="middle"
            fill="#a78bfa"
            fontSize={9}
            fontFamily="var(--font-mono)"
          >
            + Bias Applied
          </text>

          {sortedTokens.map((entry, i) => {
            const y = CHART_PAD + i * (BAR_H + BAR_GAP);
            const bias = computeBias(entry);
            const biasedLogit =
              Math.round((entry.originalLogit + bias) * 100) / 100;
            const isHovered = hoveredToken === i;
            const barYCenter = y + BAR_H * 0.17;

            return (
              <g
                key={entry.token}
                onMouseEnter={() => setHoveredToken(i)}
                onMouseLeave={() => setHoveredToken(null)}
                style={{ cursor: "pointer" }}
              >
                {/* Hover highlight */}
                {isHovered && (
                  <rect
                    x={0}
                    y={y - 2}
                    width={chartW}
                    height={BAR_H}
                    fill="rgba(167,139,250,0.06)"
                    rx={4}
                  />
                )}

                {/* Token label */}
                <text
                  x={LABEL_W - 4}
                  y={y + BAR_H * 0.5}
                  textAnchor="end"
                  fill={
                    entry.isChunkToken
                      ? isHovered
                        ? "#c4b5fd"
                        : "#a78bfa"
                      : isHovered
                        ? "#a1a1aa"
                        : "#52525b"
                  }
                  fontSize={11}
                  fontFamily="var(--font-mono)"
                  fontWeight={entry.isChunkToken ? 600 : 400}
                >
                  {entry.token}
                </text>

                {/* Original logit value */}
                <text
                  x={LABEL_W + LOGIT_LABEL_W - 6}
                  y={y + BAR_H * 0.5}
                  textAnchor="end"
                  fill="#71717a"
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                >
                  {entry.originalLogit.toFixed(2)}
                </text>

                {/* Original logit bar */}
                {logitBar(
                  entry.originalLogit,
                  "rgba(255,255,255,0.2)",
                  barYCenter,
                  0.6,
                )}

                {/* Biased logit bar (overlaid) */}
                {bias > 0 &&
                  logitBar(
                    biasedLogit,
                    "rgba(167,139,250,0.5)",
                    barYCenter,
                    0.8,
                  )}

                {/* Biased logit value */}
                {bias > 0 && (
                  <text
                    x={
                      LABEL_W +
                      LOGIT_LABEL_W +
                      MAX_BAR_W +
                      (Math.abs(biasedLogit) / maxAbsLogit) * MAX_BAR_W +
                      8
                    }
                    y={y + BAR_H * 0.5}
                    textAnchor="start"
                    fill="#a78bfa"
                    fontSize={10}
                    fontFamily="var(--font-mono)"
                    fontWeight={600}
                  >
                    {biasedLogit.toFixed(2)}
                    <tspan fill="#52525b" fontWeight={400}>
                      {" "}(+{bias.toFixed(2)})
                    </tspan>
                  </text>
                )}

                {/* No bias label for non-chunk tokens */}
                {!entry.isChunkToken && (
                  <text
                    x={LABEL_W + LOGIT_LABEL_W + MAX_BAR_W + 8}
                    y={y + BAR_H * 0.5}
                    textAnchor="start"
                    fill="#3f3f46"
                    fontSize={9}
                    fontFamily="var(--font-mono)"
                  >
                    no bias
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* ---- Hover detail ---- */}
      {hoveredToken !== null && (
        <div className="mt-3 p-3 rounded-lg border border-white/[0.06] bg-black/30 animate-fade-in">
          <div className="grid grid-cols-5 gap-3 font-mono text-xs">
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Token
              </span>
              <span
                style={{
                  color: sortedTokens[hoveredToken].isChunkToken
                    ? "#a78bfa"
                    : "#a1a1aa",
                }}
              >
                {sortedTokens[hoveredToken].token}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Original
              </span>
              <span className="text-foreground">
                {sortedTokens[hoveredToken].originalLogit.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Chunk Count
              </span>
              <span className="text-foreground">
                {sortedTokens[hoveredToken].chunkCount} / {MAX_CHUNK_COUNT}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Bias
              </span>
              <span className="text-ch4">
                +{computeBias(sortedTokens[hoveredToken]).toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Biased Logit
              </span>
              <span className="text-ch4 font-semibold">
                {(
                  sortedTokens[hoveredToken].originalLogit +
                  computeBias(sortedTokens[hoveredToken])
                ).toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ---- Explanation toggle ---- */}
      <button
        className={`btn-mono mt-4 ${showFormula ? "active" : ""}`}
        onClick={() => setShowFormula(!showFormula)}
        style={{
          borderColor: showFormula ? "#a78bfa" : undefined,
          color: showFormula ? "#a78bfa" : undefined,
        }}
      >
        {showFormula ? "Hide" : "Show"} How It Works
      </button>
      {showFormula && (
        <div className="mt-3 code-block text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Logit Bias Computation (per token):
          </div>
          <div>
            <span className="text-ch4">1.</span> Count token occurrences in
            retrieved chunks
          </div>
          <div>
            <span className="text-ch4">2.</span> Normalize:{" "}
            <span className="text-ch4">ratio</span> = count / max_count
          </div>
          <div>
            <span className="text-ch4">3.</span> Compute bias:{" "}
            <span className="text-ch4">bias</span> = strength &times; ratio
          </div>
          <div>
            <span className="text-ch4">4.</span> Add to logit:{" "}
            <span className="text-ch4">biased_logit</span> = original + bias
          </div>
          <div className="mt-2 text-muted-foreground">
            Tokens that appear frequently in retrieved chunks get larger boosts,
            making the model more likely to generate domain-relevant terminology.
            Tokens absent from chunks are unaffected.
          </div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-purple-500/20 bg-purple-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch4">Biased Tokens</div>
          <div className="font-mono text-lg font-bold text-ch4">
            {sortedTokens.filter((t) => t.isChunkToken).length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Unbiased Tokens
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {sortedTokens.filter((t) => !t.isChunkToken).length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Bias Strength
          </div>
          <div className="font-mono text-lg font-bold text-ch4">
            {biasStrength.toFixed(1)}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Max Bias
          </div>
          <div className="font-mono text-lg font-bold text-ch4">
            +{biasStrength.toFixed(2)}
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        JitRL MVP applies logit biasing at inference time to steer generation toward
        domain-relevant tokens. Tokens appearing more frequently in the top-3 retrieved
        chunks receive proportionally larger logit boosts, no weight updates required.
      </p>
    </div>
  );
}
