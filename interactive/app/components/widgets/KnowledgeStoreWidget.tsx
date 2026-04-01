"use client";

import { useState, useMemo, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  JitRL Full Knowledge Store Widget                                  */
/*  Interactive visualization of document embedding and retrieval     */
/* ------------------------------------------------------------------ */

/* ---- Seeded PRNG for stable values ---- */

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Preset documents ---- */

interface DocEntry {
  id: number;
  title: string;
  text: string;
  topic: string;
}

const PRESET_DOCS: DocEntry[] = [
  {
    id: 0,
    title: "Vector Search",
    text: "Oracle AI Vector Search stores embeddings alongside relational data for semantic similarity queries using HNSW and IVF indexes.",
    topic: "database",
  },
  {
    id: 1,
    title: "Neural Networks",
    text: "Transformer architectures use multi-head self-attention to capture long-range dependencies in sequential data for language modeling.",
    topic: "ml",
  },
  {
    id: 2,
    title: "Continual Learning",
    text: "Elastic weight consolidation prevents catastrophic forgetting by penalizing changes to parameters important for previously learned tasks.",
    topic: "ml",
  },
  {
    id: 3,
    title: "SQL Optimization",
    text: "Query execution plans use cost-based optimization with histogram statistics to choose between nested loop, hash, and merge join strategies.",
    topic: "database",
  },
  {
    id: 4,
    title: "LoRA Adapters",
    text: "Low-rank adaptation injects trainable rank decomposition matrices into frozen transformer layers, reducing fine-tuning parameters by 10000x.",
    topic: "ml",
  },
];

const HIDDEN_SIZE = 1536;
const NUM_DISPLAY_DIMS = 8;

/* ---- Simulate embedding from seeded random ---- */

function computeEmbedding(docId: number, seed: number): number[] {
  const dims: number[] = [];
  for (let i = 0; i < NUM_DISPLAY_DIMS; i++) {
    dims.push(Math.round((seeded(seed + docId * 97 + i * 13) * 2 - 1) * 1000) / 1000);
  }
  return dims;
}

/** Project high-dim embedding to 2D for visualization */
function project2D(embedding: number[]): { x: number; y: number } {
  // Simple deterministic projection using first few dims
  const x = embedding[0] * 0.6 + embedding[2] * 0.3 + embedding[4] * 0.1;
  const y = embedding[1] * 0.6 + embedding[3] * 0.3 + embedding[5] * 0.1;
  return { x, y };
}

/** Cosine similarity between two vectors */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return Math.round((dot / (Math.sqrt(normA) * Math.sqrt(normB))) * 1000) / 1000;
}

/* ---- Preset queries ---- */

const PRESET_QUERIES = [
  "How does vector similarity search work?",
  "Explain catastrophic forgetting prevention",
  "What are low-rank adapters?",
];

/* ---- Chart constants ---- */

const EMBED_CHART_SIZE = 280;
const EMBED_PAD = 30;
const EMBED_INNER = EMBED_CHART_SIZE - EMBED_PAD * 2;

/* ---- Topic colors ---- */

const TOPIC_COLORS: Record<string, string> = {
  database: "#22d3ee",
  ml: "#f472b6",
};

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function KnowledgeStoreWidget() {
  const [addedDocs, setAddedDocs] = useState<number[]>([0, 1]);
  const [query, setQuery] = useState(PRESET_QUERIES[0]);
  const [showDimensions, setShowDimensions] = useState(false);
  const [hoveredDoc, setHoveredDoc] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const EMBED_SEED = 77;

  /* ---- Compute embeddings for all added docs ---- */
  const docEmbeddings = useMemo(() => {
    return addedDocs.map((docId) => ({
      docId,
      embedding: computeEmbedding(docId, EMBED_SEED),
      pos: project2D(computeEmbedding(docId, EMBED_SEED)),
    }));
  }, [addedDocs]);

  /* ---- Compute query embedding and similarities ---- */
  const queryResult = useMemo(() => {
    // Deterministic query embedding from query string hash
    let qHash = 0;
    for (let i = 0; i < query.length; i++) {
      qHash = ((qHash << 5) - qHash + query.charCodeAt(i)) | 0;
    }
    const qEmbedding = computeEmbedding(1000 + Math.abs(qHash % 500), EMBED_SEED);
    const qPos = project2D(qEmbedding);

    const similarities = docEmbeddings.map((de) => ({
      docId: de.docId,
      similarity: cosineSimilarity(qEmbedding, de.embedding),
    }));

    const sorted = [...similarities].sort((a, b) => b.similarity - a.similarity);
    const top3Ids = new Set(sorted.slice(0, 3).map((s) => s.docId));

    return { qEmbedding, qPos, similarities, sorted, top3Ids };
  }, [query, docEmbeddings]);

  /* ---- Handlers ---- */
  const handleAddDoc = useCallback(
    (docId: number) => {
      if (!addedDocs.includes(docId)) {
        setAddedDocs((prev) => [...prev, docId]);
      }
    },
    [addedDocs],
  );

  const handleRemoveDoc = useCallback((docId: number) => {
    setAddedDocs((prev) => prev.filter((id) => id !== docId));
  }, []);

  const handlePresetQuery = useCallback((preset: string) => {
    setQuery(preset);
  }, []);

  /* ---- Normalize positions to chart space ---- */
  const allPositions = useMemo(() => {
    const positions = [...docEmbeddings.map((d) => d.pos), queryResult.qPos];
    const xs = positions.map((p) => p.x);
    const ys = positions.map((p) => p.y);
    const minX = Math.min(...xs, -1);
    const maxX = Math.max(...xs, 1);
    const minY = Math.min(...ys, -1);
    const maxY = Math.max(...ys, 1);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const normalize = (pos: { x: number; y: number }) => ({
      x: EMBED_PAD + ((pos.x - minX) / rangeX) * EMBED_INNER,
      y: EMBED_PAD + ((pos.y - minY) / rangeY) * EMBED_INNER,
    });

    return {
      docs: docEmbeddings.map((d) => ({
        docId: d.docId,
        ...normalize(d.pos),
      })),
      query: normalize(queryResult.qPos),
    };
  }, [docEmbeddings, queryResult.qPos]);

  return (
    <div className="widget-container ch5">
      <div className="widget-label">
        <span className="text-ch5">Interactive</span>{"  ·  "}JitRL Full Knowledge Store
      </div>

      {/* ---- Pipeline overview ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">Document</span>
        <span className="text-ch5"> &rarr; </span>
        <span className="text-muted-foreground">Forward Pass</span>
        <span className="text-ch5"> &rarr; </span>
        <span className="text-muted-foreground">Last Hidden State</span>
        <span className="text-ch5"> &rarr; </span>
        <span className="text-muted-foreground">Mean Pool</span>
        <span className="text-ch5"> &rarr; </span>
        <span className="text-muted-foreground">Embedding [{HIDDEN_SIZE}]</span>
        <span className="text-ch5"> &rarr; </span>
        <span className="text-muted-foreground">Knowledge Store</span>
      </div>

      {/* ---- Main three-column layout ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-5">

        {/* ---- Left: Document list ---- */}
        <div>
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Documents
          </div>
          <div className="space-y-2">
            {PRESET_DOCS.map((doc) => {
              const isAdded = addedDocs.includes(doc.id);
              const isTop3 = queryResult.top3Ids.has(doc.id);
              const sim = queryResult.similarities.find(
                (s) => s.docId === doc.id,
              );
              return (
                <div
                  key={doc.id}
                  className="rounded-lg border p-2.5 transition-all duration-300"
                  style={{
                    borderColor: isAdded
                      ? isTop3
                        ? "rgba(244,114,182,0.45)"
                        : "rgba(244,114,182,0.2)"
                      : "rgba(255,255,255,0.06)",
                    background: isAdded
                      ? isTop3
                        ? "rgba(244,114,182,0.08)"
                        : "rgba(244,114,182,0.03)"
                      : "rgba(0,0,0,0.2)",
                  }}
                  onMouseEnter={() => setHoveredDoc(doc.id)}
                  onMouseLeave={() => setHoveredDoc(null)}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{
                          background: TOPIC_COLORS[doc.topic] ?? "#71717a",
                          opacity: isAdded ? 1 : 0.3,
                        }}
                      />
                      <span
                        className="font-mono text-xs font-semibold"
                        style={{
                          color: isAdded ? "#f472b6" : "#52525b",
                        }}
                      >
                        {doc.title}
                      </span>
                    </div>
                    {isAdded && sim && (
                      <span
                        className="font-mono text-[0.6rem]"
                        style={{
                          color: isTop3 ? "#f472b6" : "#71717a",
                        }}
                      >
                        {sim.similarity.toFixed(3)}
                      </span>
                    )}
                  </div>
                  <p
                    className="font-mono text-[0.65rem] leading-relaxed mb-2"
                    style={{ color: isAdded ? "#a1a1aa" : "#52525b" }}
                  >
                    {doc.text.length > 90
                      ? doc.text.slice(0, 87) + "..."
                      : doc.text}
                  </p>
                  <div className="flex items-center gap-1.5">
                    {isAdded ? (
                      <>
                        {isTop3 && (
                          <span
                            className="font-mono text-[0.55rem] px-1.5 py-0.5 rounded"
                            style={{
                              background: "rgba(244,114,182,0.2)",
                              color: "#f472b6",
                            }}
                          >
                            TOP-{queryResult.sorted.findIndex(
                              (s) => s.docId === doc.id,
                            ) + 1}
                          </span>
                        )}
                        <button
                          className="btn-mono"
                          style={{ fontSize: "0.6rem", padding: "0.15rem 0.5rem" }}
                          onClick={() => handleRemoveDoc(doc.id)}
                        >
                          Remove
                        </button>
                      </>
                    ) : (
                      <button
                        className="btn-mono"
                        style={{
                          fontSize: "0.6rem",
                          padding: "0.15rem 0.5rem",
                          borderColor: "rgba(244,114,182,0.3)",
                          color: "#f472b6",
                        }}
                        onClick={() => handleAddDoc(doc.id)}
                      >
                        + Add to Store
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ---- Center: Embedding space visualization ---- */}
        <div>
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Embedding Space (2D Projection)
          </div>
          <div
            className="rounded-lg p-2"
            style={{
              background: "rgba(0,0,0,0.25)",
              border: "1px solid var(--border)",
            }}
          >
            <svg
              ref={svgRef}
              viewBox={`0 0 ${EMBED_CHART_SIZE} ${EMBED_CHART_SIZE}`}
              className="w-full"
              style={{ maxHeight: `${EMBED_CHART_SIZE + 20}px` }}
            >
              {/* Grid */}
              <line
                x1={EMBED_PAD}
                y1={EMBED_CHART_SIZE / 2}
                x2={EMBED_CHART_SIZE - EMBED_PAD}
                y2={EMBED_CHART_SIZE / 2}
                stroke="rgba(255,255,255,0.05)"
                strokeDasharray="4,4"
              />
              <line
                x1={EMBED_CHART_SIZE / 2}
                y1={EMBED_PAD}
                x2={EMBED_CHART_SIZE / 2}
                y2={EMBED_CHART_SIZE - EMBED_PAD}
                stroke="rgba(255,255,255,0.05)"
                strokeDasharray="4,4"
              />

              {/* Axis labels */}
              <text
                x={EMBED_CHART_SIZE / 2}
                y={EMBED_CHART_SIZE - 6}
                textAnchor="middle"
                fill="#3f3f46"
                fontSize={8}
                fontFamily="var(--font-mono)"
              >
                dim_0 projection
              </text>
              <text
                x={8}
                y={EMBED_CHART_SIZE / 2}
                textAnchor="middle"
                fill="#3f3f46"
                fontSize={8}
                fontFamily="var(--font-mono)"
                transform={`rotate(-90, 8, ${EMBED_CHART_SIZE / 2})`}
              >
                dim_1 projection
              </text>

              {/* Similarity lines from query to docs */}
              {addedDocs.length > 0 &&
                allPositions.docs.map((dp) => {
                  const isTop3 = queryResult.top3Ids.has(dp.docId);
                  return (
                    <line
                      key={`line-${dp.docId}`}
                      x1={allPositions.query.x}
                      y1={allPositions.query.y}
                      x2={dp.x}
                      y2={dp.y}
                      stroke={
                        isTop3
                          ? "rgba(244,114,182,0.4)"
                          : "rgba(255,255,255,0.08)"
                      }
                      strokeWidth={isTop3 ? 1.5 : 0.5}
                      strokeDasharray={isTop3 ? "none" : "3,3"}
                    />
                  );
                })}

              {/* Document dots */}
              {allPositions.docs.map((dp) => {
                const doc = PRESET_DOCS.find((d) => d.id === dp.docId)!;
                const isTop3 = queryResult.top3Ids.has(dp.docId);
                const isHovered = hoveredDoc === dp.docId;
                const color = TOPIC_COLORS[doc.topic] ?? "#71717a";
                return (
                  <g key={`dot-${dp.docId}`}>
                    {/* Glow for top-3 */}
                    {isTop3 && (
                      <circle
                        cx={dp.x}
                        cy={dp.y}
                        r={12}
                        fill="rgba(244,114,182,0.1)"
                      />
                    )}
                    <circle
                      cx={dp.x}
                      cy={dp.y}
                      r={isHovered ? 7 : isTop3 ? 6 : 5}
                      fill={color}
                      opacity={isTop3 ? 1 : 0.6}
                      stroke={isTop3 ? "#f472b6" : "none"}
                      strokeWidth={isTop3 ? 1.5 : 0}
                      style={{ transition: "r 0.2s" }}
                    />
                    <text
                      x={dp.x}
                      y={dp.y - 10}
                      textAnchor="middle"
                      fill={isTop3 ? "#f472b6" : "#71717a"}
                      fontSize={8}
                      fontFamily="var(--font-mono)"
                      fontWeight={isTop3 ? 600 : 400}
                    >
                      {doc.title}
                    </text>
                  </g>
                );
              })}

              {/* Query dot */}
              <circle
                cx={allPositions.query.x}
                cy={allPositions.query.y}
                r={14}
                fill="rgba(244,114,182,0.08)"
              />
              <circle
                cx={allPositions.query.x}
                cy={allPositions.query.y}
                r={7}
                fill="#f472b6"
                stroke="#0a0b0f"
                strokeWidth={2}
              />
              <text
                x={allPositions.query.x}
                y={allPositions.query.y + 16}
                textAnchor="middle"
                fill="#f472b6"
                fontSize={9}
                fontFamily="var(--font-mono)"
                fontWeight={600}
              >
                query
              </text>
            </svg>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 mt-2 font-mono text-[0.6rem] text-muted-foreground/60">
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: "#f472b6" }}
              />
              Query
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: "#22d3ee" }}
              />
              Database
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: "#f472b6", opacity: 0.6 }}
              />
              ML
            </span>
          </div>
        </div>

        {/* ---- Right: Query matching ---- */}
        <div>
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Query &amp; Retrieval
          </div>

          {/* Query input */}
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-ch5/50 transition-colors mb-2"
            placeholder="Type a query..."
          />
          <div className="flex flex-wrap gap-1.5 mb-4">
            {PRESET_QUERIES.map((p, i) => (
              <button
                key={i}
                className={`btn-mono ${query === p ? "active" : ""}`}
                onClick={() => handlePresetQuery(p)}
                style={{
                  fontSize: "0.6rem",
                  padding: "0.2rem 0.5rem",
                  borderColor: query === p ? "#f472b6" : undefined,
                  color: query === p ? "#f472b6" : undefined,
                }}
              >
                {p.length > 30 ? p.slice(0, 27) + "..." : p}
              </button>
            ))}
          </div>

          {/* Similarity ranking */}
          <div className="font-mono text-[0.6rem] uppercase tracking-wider text-muted-foreground mb-2">
            Cosine Similarity Ranking
          </div>
          {addedDocs.length === 0 ? (
            <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
              <span className="font-mono text-xs text-muted-foreground/50">
                Add documents to the store
              </span>
            </div>
          ) : (
            <div className="space-y-1.5">
              {queryResult.sorted.map((item, rank) => {
                const doc = PRESET_DOCS.find((d) => d.id === item.docId)!;
                const isTop3 = rank < 3;
                const maxSim = Math.max(
                  ...queryResult.similarities.map((s) => Math.abs(s.similarity)),
                  0.001,
                );
                const barWidth = Math.max(
                  2,
                  (Math.abs(item.similarity) / maxSim) * 100,
                );
                return (
                  <div
                    key={item.docId}
                    className="rounded border p-2 transition-all duration-300"
                    style={{
                      borderColor: isTop3
                        ? "rgba(244,114,182,0.3)"
                        : "rgba(255,255,255,0.06)",
                      background: isTop3
                        ? "rgba(244,114,182,0.05)"
                        : "rgba(0,0,0,0.15)",
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1.5">
                        {isTop3 && (
                          <span
                            className="font-mono text-[0.55rem] px-1 py-0.5 rounded"
                            style={{
                              background: "rgba(244,114,182,0.2)",
                              color: "#f472b6",
                            }}
                          >
                            #{rank + 1}
                          </span>
                        )}
                        <span
                          className="font-mono text-[0.7rem]"
                          style={{
                            color: isTop3 ? "#f9a8d4" : "#71717a",
                          }}
                        >
                          {doc.title}
                        </span>
                      </div>
                      <span
                        className="font-mono text-[0.65rem] font-semibold"
                        style={{ color: isTop3 ? "#f472b6" : "#52525b" }}
                      >
                        {item.similarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-1.5 rounded bg-white/[0.06] overflow-hidden">
                      <div
                        className="h-full rounded transition-all duration-500 ease-out"
                        style={{
                          width: `${barWidth}%`,
                          background: isTop3
                            ? "rgba(244,114,182,0.5)"
                            : "rgba(255,255,255,0.1)",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ---- Embedding dimensions toggle ---- */}
      <button
        className={`btn-mono ${showDimensions ? "active" : ""}`}
        onClick={() => setShowDimensions(!showDimensions)}
        style={{
          borderColor: showDimensions ? "#f472b6" : undefined,
          color: showDimensions ? "#f472b6" : undefined,
        }}
      >
        {showDimensions ? "Hide" : "Show"} Embedding Dimensions
      </button>

      {showDimensions && (
        <div className="mt-4 animate-fade-in">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Embedding Pipeline: [1, num_tokens, {HIDDEN_SIZE}] &rarr; mean_pool &rarr; [{HIDDEN_SIZE}]
          </div>
          <div className="code-block text-xs overflow-x-auto">
            <div className="text-muted-foreground mb-2">
              First {NUM_DISPLAY_DIMS} of {HIDDEN_SIZE} dimensions (showing stored embeddings):
            </div>
            {docEmbeddings.map((de) => {
              const doc = PRESET_DOCS.find((d) => d.id === de.docId)!;
              const isTop3 = queryResult.top3Ids.has(de.docId);
              return (
                <div key={de.docId} className="mb-1">
                  <span style={{ color: isTop3 ? "#f472b6" : "#71717a" }}>
                    {doc.title.padEnd(18)}
                  </span>
                  <span className="text-muted-foreground">[</span>
                  {de.embedding.map((v, i) => (
                    <span key={i}>
                      <span
                        style={{
                          color:
                            v > 0
                              ? `rgba(244,114,182,${0.4 + Math.abs(v) * 0.6})`
                              : `rgba(34,211,238,${0.4 + Math.abs(v) * 0.6})`,
                        }}
                      >
                        {v >= 0 ? " " : ""}
                        {v.toFixed(3)}
                      </span>
                      {i < de.embedding.length - 1 && (
                        <span className="text-muted-foreground/30">, </span>
                      )}
                    </span>
                  ))}
                  <span className="text-muted-foreground">
                    , ...{HIDDEN_SIZE - NUM_DISPLAY_DIMS} more]
                  </span>
                </div>
              );
            })}
            <div className="mt-2 border-t border-white/[0.06] pt-2">
              <span className="text-ch5">query{"            "}</span>
              <span className="text-muted-foreground">[</span>
              {queryResult.qEmbedding.map((v, i) => (
                <span key={i}>
                  <span
                    style={{
                      color:
                        v > 0
                          ? `rgba(244,114,182,${0.4 + Math.abs(v) * 0.6})`
                          : `rgba(34,211,238,${0.4 + Math.abs(v) * 0.6})`,
                    }}
                  >
                    {v >= 0 ? " " : ""}
                    {v.toFixed(3)}
                  </span>
                  {i < queryResult.qEmbedding.length - 1 && (
                    <span className="text-muted-foreground/30">, </span>
                  )}
                </span>
              ))}
              <span className="text-muted-foreground">
                , ...{HIDDEN_SIZE - NUM_DISPLAY_DIMS} more]
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-pink-500/20 bg-pink-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch5">Stored Docs</div>
          <div className="font-mono text-lg font-bold text-ch5">
            {addedDocs.length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Available
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {PRESET_DOCS.length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Embed Dims
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {HIDDEN_SIZE}
          </div>
        </div>
        <div className="rounded-lg border border-pink-500/20 bg-pink-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch5">Best Match</div>
          <div className="font-mono text-lg font-bold text-ch5">
            {queryResult.sorted[0]?.similarity.toFixed(3) ?? "---"}
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        JitRL Full encodes documents by running a forward pass through the base model, capturing
        the last hidden state [1, num_tokens, {HIDDEN_SIZE}], then mean-pooling across the token
        dimension to produce a single [{HIDDEN_SIZE}] embedding vector. At query time, the query
        is embedded identically and cosine similarity selects the top-3 most relevant stored
        documents for reward-guided logit modulation.
      </p>
    </div>
  );
}
