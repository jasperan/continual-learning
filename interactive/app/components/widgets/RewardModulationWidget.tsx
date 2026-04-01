"use client";

import { useState, useMemo, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Reward-Guided Logit Modulation Widget                              */
/*  Interactive visualization of reward computation and logit          */
/*  modulation pipeline in JitRL Full                                  */
/* ------------------------------------------------------------------ */

/* ---- Seeded PRNG for stable values ---- */

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Constants ---- */

const SEED = 53;
const NUM_TOKENS = 10;
const EMBED_DIM = 4; // displayed embedding dimensions (conceptual)

/* ---- Token data ---- */

interface TokenData {
  token: string;
  originalLogit: number;
}

function buildTokenData(seed: number): TokenData[] {
  const tokens = [
    "vector", "search", "embedding", "index", "cosine",
    "the", "is", "query", "model", "data",
  ];
  return tokens.map((token, i) => ({
    token,
    originalLogit: Math.round((seeded(seed + i * 7) * 8 - 4) * 100) / 100,
  }));
}

/* ---- Simulated vector computations ---- */

function buildQueryEmbedding(seed: number): number[] {
  return Array.from({ length: EMBED_DIM }, (_, i) =>
    Math.round((seeded(seed + i * 11) * 2 - 1) * 1000) / 1000,
  );
}

function buildKnowledgeEmbedding(seed: number): number[] {
  return Array.from({ length: EMBED_DIM }, (_, i) =>
    Math.round((seeded(seed + 200 + i * 13) * 2 - 1) * 1000) / 1000,
  );
}

/** Element-wise multiply of normalized vectors, then mean (simplified from full hidden_size) */
function elementWiseReward(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return Math.round((sum / a.length) * 1000) / 1000;
}

/** Simulate reward logits via lm_head projection */
function buildRewardLogits(rewardValue: number, seed: number, numTokens: number): number[] {
  return Array.from({ length: numTokens }, (_, i) => {
    const projWeight = seeded(seed + 500 + i * 17) * 2 - 1;
    return Math.round(rewardValue * projWeight * 1000) / 1000;
  });
}

/* ---- Chart constants ---- */

const BAR_H = 24;
const BAR_GAP = 6;
const LABEL_W = 70;
const VALUE_W = 55;
const MAX_BAR_W = 140;
const CHART_PAD = 16;

/* ---- Pipeline stages ---- */

const PIPELINE_STAGES = [
  { key: "embed", label: "Element-wise Multiply", desc: "normalize(query) * normalize(knowledge)" },
  { key: "reward", label: "Reward Vector", desc: "Continuous semantic relevance" },
  { key: "project", label: "LM Head Projection", desc: "reward @ lm_head.T" },
  { key: "modulate", label: "Logit Modulation", desc: "logits + temp \u00d7 reward_logits" },
];

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function RewardModulationWidget() {
  const [temperature, setTemperature] = useState(0.5);
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [showComputation, setShowComputation] = useState(false);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);

  const tokens = useMemo(() => buildTokenData(SEED), []);
  const queryEmb = useMemo(() => buildQueryEmbedding(SEED), []);
  const knowledgeEmb = useMemo(() => buildKnowledgeEmbedding(SEED), []);
  const rewardValue = useMemo(
    () => elementWiseReward(queryEmb, knowledgeEmb),
    [queryEmb, knowledgeEmb],
  );
  const rewardLogits = useMemo(
    () => buildRewardLogits(rewardValue, SEED, NUM_TOKENS),
    [rewardValue],
  );

  /** Compute modulated logit for a given token index */
  const modulatedLogit = useCallback(
    (idx: number): number => {
      return (
        Math.round(
          (tokens[idx].originalLogit + temperature * rewardLogits[idx]) * 100,
        ) / 100
      );
    },
    [tokens, temperature, rewardLogits],
  );

  /** Compute delta for a given token index */
  const logitDelta = useCallback(
    (idx: number): number => {
      return Math.round(temperature * rewardLogits[idx] * 100) / 100;
    },
    [temperature, rewardLogits],
  );

  /** Max absolute logit for chart scaling */
  const maxAbsLogit = useMemo(() => {
    let max = 0;
    for (let i = 0; i < tokens.length; i++) {
      max = Math.max(
        max,
        Math.abs(tokens[i].originalLogit),
        Math.abs(modulatedLogit(i)),
      );
    }
    return Math.max(max, 1);
  }, [tokens, modulatedLogit]);

  const totalH =
    tokens.length * (BAR_H + BAR_GAP) - BAR_GAP + CHART_PAD * 2 + 20;
  const chartW = LABEL_W + VALUE_W + MAX_BAR_W + 20 + MAX_BAR_W + VALUE_W + 80;

  /** Helper: draw a horizontal bar from center axis */
  function renderBar(
    value: number,
    color: string,
    yPos: number,
    opacity: number = 1,
  ) {
    const centerX = LABEL_W + VALUE_W + MAX_BAR_W;
    const barW = (Math.abs(value) / maxAbsLogit) * MAX_BAR_W;
    const xStart = value >= 0 ? centerX : centerX - barW;
    return (
      <rect
        x={xStart}
        y={yPos}
        width={Math.max(1, barW)}
        height={BAR_H * 0.55}
        rx={3}
        fill={color}
        opacity={opacity}
      />
    );
  }

  return (
    <div className="widget-container ch5">
      <div className="widget-label">
        <span className="text-ch5">Interactive</span>{"  ·  "}Reward-Guided Logit Modulation
      </div>

      {/* ---- Formula display ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">modulated</span>
        <span className="text-ch5"> = </span>
        <span className="text-muted-foreground">logits</span>
        <span className="text-ch5"> + </span>
        <span className="text-muted-foreground">temperature</span>
        <span className="text-ch5"> &times; </span>
        <span className="text-muted-foreground">(reward @ lm_head.T)</span>
      </div>

      {/* ---- Pipeline stages ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-5">
        {PIPELINE_STAGES.map((stage) => (
          <div
            key={stage.key}
            className={`pipeline-stage ${activeStage === stage.key ? "active" : ""}`}
            onClick={() =>
              setActiveStage(activeStage === stage.key ? null : stage.key)
            }
            style={{
              borderColor:
                activeStage === stage.key
                  ? "rgba(244,114,182,0.4)"
                  : undefined,
            }}
          >
            <div
              className="font-mono text-[0.65rem] font-semibold mb-0.5"
              style={{
                color:
                  activeStage === stage.key ? "#f472b6" : "#a1a1aa",
              }}
            >
              {stage.label}
            </div>
            <div className="font-mono text-[0.55rem] text-muted-foreground/60">
              {stage.desc}
            </div>
          </div>
        ))}
      </div>

      {/* ---- Stage detail panels ---- */}
      {activeStage === "embed" && (
        <div className="code-block mb-5 text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Step 1: Element-wise multiply of normalized query and knowledge embeddings
          </div>
          <div className="mb-1">
            <span className="text-ch5">query_emb{"    "}</span>
            <span className="text-muted-foreground">= [</span>
            {queryEmb.map((v, i) => (
              <span key={i}>
                <span style={{ color: v >= 0 ? "#f9a8d4" : "#67e8f9" }}>
                  {v >= 0 ? " " : ""}{v.toFixed(3)}
                </span>
                {i < queryEmb.length - 1 && (
                  <span className="text-muted-foreground/30">, </span>
                )}
              </span>
            ))}
            <span className="text-muted-foreground">]</span>
          </div>
          <div className="mb-1">
            <span className="text-ch5">knowledge_emb</span>
            <span className="text-muted-foreground"> = [</span>
            {knowledgeEmb.map((v, i) => (
              <span key={i}>
                <span style={{ color: v >= 0 ? "#f9a8d4" : "#67e8f9" }}>
                  {v >= 0 ? " " : ""}{v.toFixed(3)}
                </span>
                {i < knowledgeEmb.length - 1 && (
                  <span className="text-muted-foreground/30">, </span>
                )}
              </span>
            ))}
            <span className="text-muted-foreground">]</span>
          </div>
          <div className="mt-2 pt-2 border-t border-white/[0.06]">
            <span className="text-muted-foreground">reward = mean(</span>
            {queryEmb.map((v, i) => (
              <span key={i}>
                <span className="text-muted-foreground/50">
                  {v.toFixed(3)}&times;{knowledgeEmb[i].toFixed(3)}
                </span>
                {i < queryEmb.length - 1 && (
                  <span className="text-muted-foreground/30"> + </span>
                )}
              </span>
            ))}
            <span className="text-ch5 font-semibold">
              {") "}= {rewardValue.toFixed(3)}
            </span>
          </div>
        </div>
      )}

      {activeStage === "reward" && (
        <div className="code-block mb-5 text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Step 2: Reward is a continuous scalar &mdash; not binary
          </div>
          <div className="flex items-center gap-4">
            <div>
              <span className="text-muted-foreground">reward = </span>
              <span className="text-ch5 font-semibold text-sm">
                {rewardValue.toFixed(3)}
              </span>
            </div>
            <div className="flex-1">
              <div className="h-3 rounded bg-white/[0.06] overflow-hidden relative">
                <div
                  className="absolute top-0 left-1/2 w-px h-full"
                  style={{ background: "rgba(255,255,255,0.15)" }}
                />
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{
                    width: `${Math.abs(rewardValue) / 2 * 100}%`,
                    marginLeft: rewardValue >= 0 ? "50%" : `${50 - Math.abs(rewardValue) / 2 * 100}%`,
                    background: rewardValue >= 0
                      ? "rgba(244,114,182,0.6)"
                      : "rgba(34,211,238,0.6)",
                  }}
                />
              </div>
              <div className="flex justify-between font-mono text-[0.55rem] text-muted-foreground/40 mt-0.5">
                <span>-2.0</span>
                <span>0.0</span>
                <span>+2.0</span>
              </div>
            </div>
          </div>
          <div className="text-muted-foreground/60 mt-2">
            Higher reward = query is semantically closer to stored knowledge.
            The continuous value captures degree of relevance, not just yes/no.
          </div>
        </div>
      )}

      {activeStage === "project" && (
        <div className="code-block mb-5 text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Step 3: Project reward through language model head
          </div>
          <div className="mb-1">
            <span className="text-ch5">reward_logits</span>
            <span className="text-muted-foreground"> = reward &times; lm_head.T</span>
          </div>
          <div className="mt-2 space-y-0.5">
            {tokens.map((t, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-muted-foreground w-16 text-right">
                  {t.token}
                </span>
                <span
                  className="font-semibold w-16"
                  style={{
                    color:
                      rewardLogits[i] >= 0
                        ? `rgba(244,114,182,${0.5 + Math.min(Math.abs(rewardLogits[i]), 1) * 0.5})`
                        : `rgba(34,211,238,${0.5 + Math.min(Math.abs(rewardLogits[i]), 1) * 0.5})`,
                  }}
                >
                  {rewardLogits[i] >= 0 ? "+" : ""}{rewardLogits[i].toFixed(3)}
                </span>
                <div className="flex-1 h-1.5 rounded bg-white/[0.06] overflow-hidden">
                  <div
                    className="h-full rounded"
                    style={{
                      width: `${(Math.abs(rewardLogits[i]) / Math.max(...rewardLogits.map(Math.abs), 0.01)) * 100}%`,
                      background:
                        rewardLogits[i] >= 0
                          ? "rgba(244,114,182,0.4)"
                          : "rgba(34,211,238,0.4)",
                      marginLeft: rewardLogits[i] < 0 ? "auto" : undefined,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeStage === "modulate" && (
        <div className="code-block mb-5 text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Step 4: Add scaled reward logits to original logits
          </div>
          <div className="space-y-0.5">
            {tokens.map((t, i) => (
              <div key={i} className="font-mono">
                <span className="text-muted-foreground">{t.token.padEnd(10)}</span>
                <span className="text-muted-foreground/60">
                  {t.originalLogit >= 0 ? " " : ""}
                  {t.originalLogit.toFixed(2)}
                </span>
                <span className="text-ch5"> + </span>
                <span className="text-muted-foreground/60">
                  {temperature.toFixed(1)} &times;{" "}
                </span>
                <span
                  style={{
                    color:
                      rewardLogits[i] >= 0
                        ? "rgba(244,114,182,0.8)"
                        : "rgba(34,211,238,0.8)",
                  }}
                >
                  {rewardLogits[i] >= 0 ? "+" : ""}
                  {rewardLogits[i].toFixed(3)}
                </span>
                <span className="text-ch5"> = </span>
                <span className="text-ch5 font-semibold">
                  {modulatedLogit(i).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ---- Temperature control ---- */}
      <div className="flex items-center gap-4 mb-5">
        <label className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          temperature:
        </label>
        <input
          type="range"
          min={0}
          max={20}
          value={Math.round(temperature * 10)}
          onChange={(e) => setTemperature(Number(e.target.value) / 10)}
          className="flex-1 max-w-[250px]"
        />
        <span className="font-mono text-sm text-ch5 font-semibold w-12 text-right">
          {temperature.toFixed(1)}
        </span>
      </div>

      {/* ---- Side-by-side bar chart ---- */}
      <div
        className="rounded-lg p-2 overflow-x-auto"
        style={{
          background: "rgba(0,0,0,0.25)",
          border: "1px solid var(--border)",
        }}
      >
        <svg
          viewBox={`0 0 ${chartW} ${totalH}`}
          className="w-full"
          style={{ minWidth: "600px", maxHeight: `${totalH + 20}px` }}
        >
          {/* Center axis */}
          <line
            x1={LABEL_W + VALUE_W + MAX_BAR_W}
            y1={CHART_PAD}
            x2={LABEL_W + VALUE_W + MAX_BAR_W}
            y2={totalH - CHART_PAD}
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={1}
          />

          {/* Column headers */}
          <text
            x={LABEL_W + VALUE_W + MAX_BAR_W / 2}
            y={CHART_PAD - 2}
            textAnchor="middle"
            fill="#71717a"
            fontSize={9}
            fontFamily="var(--font-mono)"
          >
            Original Logits
          </text>
          <text
            x={LABEL_W + VALUE_W + MAX_BAR_W + MAX_BAR_W / 2 + 10}
            y={CHART_PAD - 2}
            textAnchor="middle"
            fill="#f472b6"
            fontSize={9}
            fontFamily="var(--font-mono)"
          >
            + Reward Modulated
          </text>

          {tokens.map((entry, i) => {
            const y = CHART_PAD + 10 + i * (BAR_H + BAR_GAP);
            const mod = modulatedLogit(i);
            const delta = logitDelta(i);
            const isHovered = hoveredToken === i;
            const barY = y + BAR_H * 0.22;

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
                    fill="rgba(244,114,182,0.06)"
                    rx={4}
                  />
                )}

                {/* Token label */}
                <text
                  x={LABEL_W - 4}
                  y={y + BAR_H * 0.5}
                  textAnchor="end"
                  fill={
                    isHovered
                      ? "#f9a8d4"
                      : delta !== 0
                        ? "#f472b6"
                        : "#52525b"
                  }
                  fontSize={11}
                  fontFamily="var(--font-mono)"
                  fontWeight={Math.abs(delta) > 0.1 ? 600 : 400}
                >
                  {entry.token}
                </text>

                {/* Original logit value */}
                <text
                  x={LABEL_W + VALUE_W - 6}
                  y={y + BAR_H * 0.5}
                  textAnchor="end"
                  fill="#71717a"
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                >
                  {entry.originalLogit.toFixed(2)}
                </text>

                {/* Original logit bar (muted) */}
                {renderBar(
                  entry.originalLogit,
                  "rgba(255,255,255,0.18)",
                  barY,
                  0.6,
                )}

                {/* Modulated logit bar (overlaid) */}
                {renderBar(
                  mod,
                  delta >= 0
                    ? "rgba(244,114,182,0.5)"
                    : "rgba(34,211,238,0.4)",
                  barY,
                  0.8,
                )}

                {/* Modulated logit value + delta */}
                <text
                  x={
                    LABEL_W +
                    VALUE_W +
                    MAX_BAR_W +
                    (Math.abs(mod) / maxAbsLogit) * MAX_BAR_W +
                    12
                  }
                  y={y + BAR_H * 0.5}
                  textAnchor="start"
                  fill={delta >= 0 ? "#f472b6" : "#67e8f9"}
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                  fontWeight={600}
                >
                  {mod.toFixed(2)}
                  <tspan fill="#52525b" fontWeight={400}>
                    {" "}({delta >= 0 ? "+" : ""}{delta.toFixed(2)})
                  </tspan>
                </text>
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
              <span className="text-ch5">
                {tokens[hoveredToken].token}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Original
              </span>
              <span className="text-foreground">
                {tokens[hoveredToken].originalLogit.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Reward Logit
              </span>
              <span
                style={{
                  color:
                    rewardLogits[hoveredToken] >= 0 ? "#f9a8d4" : "#67e8f9",
                }}
              >
                {rewardLogits[hoveredToken] >= 0 ? "+" : ""}
                {rewardLogits[hoveredToken].toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Delta
              </span>
              <span className="text-ch5">
                {logitDelta(hoveredToken) >= 0 ? "+" : ""}
                {logitDelta(hoveredToken).toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[0.6rem]">
                Modulated
              </span>
              <span className="text-ch5 font-semibold">
                {modulatedLogit(hoveredToken).toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ---- Show computation toggle ---- */}
      <button
        className={`btn-mono mt-4 ${showComputation ? "active" : ""}`}
        onClick={() => setShowComputation(!showComputation)}
        style={{
          borderColor: showComputation ? "#f472b6" : undefined,
          color: showComputation ? "#f472b6" : undefined,
        }}
      >
        {showComputation ? "Hide" : "Show"} How It Works
      </button>
      {showComputation && (
        <div className="mt-3 code-block text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            Reward-Guided Logit Modulation Pipeline:
          </div>
          <div>
            <span className="text-ch5">1.</span> Run query through model &rarr; get{" "}
            <span className="text-ch5">query_embedding</span> [{EMBED_DIM} dims shown]
          </div>
          <div>
            <span className="text-ch5">2.</span> Dot product with stored{" "}
            <span className="text-ch5">knowledge_embedding</span> &rarr;{" "}
            <span className="text-ch5">reward</span> = {rewardValue.toFixed(3)}
          </div>
          <div>
            <span className="text-ch5">3.</span> Project reward through{" "}
            <span className="text-ch5">lm_head.T</span> &rarr;{" "}
            <span className="text-ch5">reward_logits</span> [vocab_size]
          </div>
          <div>
            <span className="text-ch5">4.</span> Modulate:{" "}
            <span className="text-ch5">output</span> = logits + {temperature.toFixed(1)} &times; reward_logits
          </div>
          <div className="mt-2 text-muted-foreground">
            The reward signal is continuous (not binary), capturing the degree of
            semantic relevance between the query and stored knowledge. Temperature
            controls modulation strength: higher values amplify the reward signal,
            causing stronger shifts toward knowledge-aligned tokens.
          </div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-pink-500/20 bg-pink-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch5">Reward</div>
          <div className="font-mono text-lg font-bold text-ch5">
            {rewardValue.toFixed(3)}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Temperature
          </div>
          <div className="font-mono text-lg font-bold text-ch5">
            {temperature.toFixed(1)}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Max Delta
          </div>
          <div className="font-mono text-lg font-bold text-ch5">
            {Math.max(...tokens.map((_, i) => Math.abs(logitDelta(i)))).toFixed(2)}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Tokens
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {NUM_TOKENS}
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        JitRL Full computes a continuous reward signal via element-wise multiplication of normalized query and
        knowledge embeddings, then projects this through the language model head to produce
        per-token reward logits. These are scaled by temperature and added to the original
        logits, steering generation toward knowledge-relevant tokens without any weight updates.
      </p>
    </div>
  );
}
