"use client";

import { useState, useMemo, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  JitRL MVP Retrieval Widget                                        */
/*  Interactive visualization of TF-IDF retrieval for JitRL MVP       */
/* ------------------------------------------------------------------ */

/* ---- Sample document chunks ---- */

const SAMPLE_CHUNKS = [
  {
    id: 0,
    text: "Oracle AI Vector Search enables semantic search by storing vector embeddings alongside relational data. It supports approximate nearest neighbor (ANN) indexing using IVF and HNSW algorithms for high-performance similarity queries.",
  },
  {
    id: 1,
    text: "Vector embeddings can be generated using ONNX models loaded directly into Oracle Database. The DBMS_VECTOR package provides functions for embedding generation, chunking, and similarity search within SQL queries.",
  },
  {
    id: 2,
    text: "Partitioning in Oracle Database divides large tables into smaller segments for improved query performance. Range, list, and hash partitioning strategies are available, each suited for different data access patterns.",
  },
  {
    id: 3,
    text: "The VECTOR data type in Oracle 23ai stores dense and sparse embeddings natively. Cosine similarity, dot product, and Euclidean distance functions enable flexible similarity computations directly in SQL.",
  },
  {
    id: 4,
    text: "Oracle Autonomous Database automates patching, tuning, and scaling. It provides built-in machine learning capabilities through OML4Py and supports JSON, graph, and spatial data alongside traditional relational workloads.",
  },
];

const TOP_K = 3;

/* ---- Seeded PRNG for stable values ---- */

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Simplified TF-IDF simulation ---- */

/** Tokenize text into lowercase terms */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

/** Compute term frequency map for a token list */
function termFrequency(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  // Normalize by total tokens
  for (const [k, v] of tf) tf.set(k, v / tokens.length);
  return tf;
}

/** Compute IDF for terms across all chunks */
function inverseDocFrequency(
  chunks: string[][],
  terms: Set<string>,
): Map<string, number> {
  const idf = new Map<string, number>();
  const N = chunks.length;
  for (const term of terms) {
    const df = chunks.filter((tokens) => tokens.includes(term)).length;
    idf.set(term, Math.log((N + 1) / (df + 1)) + 1);
  }
  return idf;
}

/** Compute TF-IDF vector for a token list given global IDF */
function tfidfVector(
  tokens: string[],
  idf: Map<string, number>,
  allTerms: string[],
): number[] {
  const tf = termFrequency(tokens);
  return allTerms.map((term) => {
    const tfVal = tf.get(term) ?? 0;
    const idfVal = idf.get(term) ?? 0;
    return tfVal * idfVal;
  });
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
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* ---- Preset queries ---- */

const PRESET_QUERIES = [
  "How does vector similarity search work in Oracle?",
  "What partitioning strategies are available?",
  "How do I generate embeddings in the database?",
];

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function RetrievalWidget() {
  const [query, setQuery] = useState(PRESET_QUERIES[0]);
  const [showPrompt, setShowPrompt] = useState(false);

  /* ---- Compute retrieval ---- */
  const results = useMemo(() => {
    // Tokenize all chunks + query
    const chunkTokens = SAMPLE_CHUNKS.map((c) => tokenize(c.text));
    const queryTokens = tokenize(query);

    // Build vocabulary from all terms
    const allTermsSet = new Set<string>();
    for (const tokens of chunkTokens) for (const t of tokens) allTermsSet.add(t);
    for (const t of queryTokens) allTermsSet.add(t);
    const allTerms = Array.from(allTermsSet).sort();

    // Compute IDF
    const idf = inverseDocFrequency(chunkTokens, allTermsSet);

    // Get TF-IDF vectors
    const queryVec = tfidfVector(queryTokens, idf, allTerms);
    const chunkVecs = chunkTokens.map((tokens) =>
      tfidfVector(tokens, idf, allTerms),
    );

    // Compute similarities
    const similarities = chunkVecs.map((vec, i) => ({
      chunkId: i,
      similarity: Math.round(cosineSimilarity(queryVec, vec) * 1000) / 1000,
    }));

    // Sort by similarity descending
    const sorted = [...similarities].sort(
      (a, b) => b.similarity - a.similarity,
    );

    // Top-K set
    const topKIds = new Set(sorted.slice(0, TOP_K).map((s) => s.chunkId));

    // Key terms from query that appear in chunks
    const queryTermSet = new Set(queryTokens);
    const matchingTerms = allTerms.filter(
      (t) => queryTermSet.has(t) && chunkTokens.some((ct) => ct.includes(t)),
    );

    return { similarities, sorted, topKIds, matchingTerms, allTerms };
  }, [query]);

  const handlePreset = useCallback((preset: string) => {
    setQuery(preset);
    setShowPrompt(false);
  }, []);

  // Assemble the context prompt
  const assembledPrompt = useMemo(() => {
    const topChunks = results.sorted
      .slice(0, TOP_K)
      .map((s) => SAMPLE_CHUNKS[s.chunkId].text);
    return `Context:\n${topChunks.join("\n\n")}\n\nQuestion: ${query}\nAnswer:`;
  }, [results, query]);

  return (
    <div className="widget-container ch4">
      <div className="widget-label">
        <span className="text-ch4">Interactive</span>{"  ·  "}JitRL MVP TF-IDF Retrieval
      </div>

      {/* ---- Pipeline overview ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">Query</span>
        <span className="text-ch4"> &rarr; </span>
        <span className="text-muted-foreground">TF-IDF Vectorize</span>
        <span className="text-ch4"> &rarr; </span>
        <span className="text-muted-foreground">Cosine Similarity</span>
        <span className="text-ch4"> &rarr; </span>
        <span className="text-muted-foreground">Top-{TOP_K} Chunks</span>
        <span className="text-ch4"> &rarr; </span>
        <span className="text-muted-foreground">Context Prompt</span>
      </div>

      {/* ---- Query input ---- */}
      <div className="flex flex-col gap-3 mb-5">
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setShowPrompt(false); }}
            className="flex-1 rounded-lg border border-white/10 bg-black/30 px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-ch4/50 transition-colors"
            placeholder="Type a query..."
          />
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESET_QUERIES.map((p, i) => (
            <button
              key={i}
              className={`btn-mono ${query === p ? "active" : ""}`}
              onClick={() => handlePreset(p)}
              style={{
                borderColor: query === p ? "#a78bfa" : undefined,
                color: query === p ? "#a78bfa" : undefined,
              }}
            >
              {p.length > 40 ? p.slice(0, 37) + "..." : p}
            </button>
          ))}
        </div>
      </div>

      {/* ---- Key matching terms ---- */}
      {results.matchingTerms.length > 0 && (
        <div className="mb-5">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Query Terms Matched in Chunks
          </div>
          <div className="flex flex-wrap gap-1.5">
            {results.matchingTerms.slice(0, 12).map((term) => (
              <span
                key={term}
                className="token-box"
                style={{
                  background: "rgba(167,139,250,0.12)",
                  borderColor: "rgba(167,139,250,0.3)",
                  color: "#a78bfa",
                }}
              >
                {term}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ---- Chunk similarity results ---- */}
      <div className="space-y-3 mb-5">
        <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground">
          Document Chunks &mdash; Cosine Similarity Scores
        </div>
        {results.sorted.map((item, rank) => {
          const chunk = SAMPLE_CHUNKS[item.chunkId];
          const isTopK = rank < TOP_K;
          const maxSim = Math.max(
            ...results.similarities.map((s) => s.similarity),
            0.001,
          );
          const barWidth = Math.max(2, (item.similarity / maxSim) * 100);

          return (
            <div
              key={item.chunkId}
              className="rounded-lg border p-3 transition-all duration-300"
              style={{
                borderColor: isTopK
                  ? "rgba(167,139,250,0.35)"
                  : "rgba(255,255,255,0.06)",
                background: isTopK
                  ? "rgba(167,139,250,0.06)"
                  : "rgba(0,0,0,0.2)",
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {isTopK && (
                    <span
                      className="font-mono text-[0.6rem] px-1.5 py-0.5 rounded"
                      style={{
                        background: "rgba(167,139,250,0.2)",
                        color: "#a78bfa",
                      }}
                    >
                      TOP-{rank + 1}
                    </span>
                  )}
                  <span className="font-mono text-xs text-muted-foreground">
                    Chunk #{chunk.id}
                  </span>
                </div>
                <span
                  className="font-mono text-sm font-semibold"
                  style={{
                    color: isTopK ? "#a78bfa" : "#71717a",
                  }}
                >
                  {item.similarity.toFixed(3)}
                </span>
              </div>

              {/* Similarity bar */}
              <div className="h-2 rounded bg-white/[0.06] overflow-hidden mb-2">
                <div
                  className="h-full rounded transition-all duration-500 ease-out"
                  style={{
                    width: `${barWidth}%`,
                    background: isTopK
                      ? "rgba(167,139,250,0.6)"
                      : "rgba(255,255,255,0.12)",
                  }}
                />
              </div>

              {/* Chunk text */}
              <p
                className="font-mono text-[0.7rem] leading-relaxed"
                style={{
                  color: isTopK ? "#c4b5fd" : "#71717a",
                }}
              >
                {chunk.text}
              </p>
            </div>
          );
        })}
      </div>

      {/* ---- Show assembled prompt ---- */}
      <button
        className={`btn-mono ${showPrompt ? "active" : ""}`}
        onClick={() => setShowPrompt(!showPrompt)}
        style={{
          borderColor: showPrompt ? "#a78bfa" : undefined,
          color: showPrompt ? "#a78bfa" : undefined,
        }}
      >
        {showPrompt ? "Hide" : "Show"} Assembled Prompt
      </button>

      {showPrompt && (
        <div className="mt-4 animate-fade-in">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Prompt Sent to Model
          </div>
          <div className="code-block text-xs whitespace-pre-wrap">
            {assembledPrompt.split("\n").map((line, i) => {
              if (line.startsWith("Context:"))
                return (
                  <div key={i}>
                    <span className="text-ch4">{line}</span>
                  </div>
                );
              if (line.startsWith("Question:"))
                return (
                  <div key={i}>
                    <span className="text-ch4">Question:</span>
                    <span className="text-foreground">
                      {line.slice("Question:".length)}
                    </span>
                  </div>
                );
              if (line.startsWith("Answer:"))
                return (
                  <div key={i}>
                    <span className="text-ch4">{line}</span>
                  </div>
                );
              if (line.trim() === "")
                return <div key={i}>&nbsp;</div>;
              return (
                <div key={i} className="text-muted-foreground">
                  {line}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Total Chunks
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {SAMPLE_CHUNKS.length}
          </div>
        </div>
        <div className="rounded-lg border border-purple-500/20 bg-purple-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch4">Retrieved</div>
          <div className="font-mono text-lg font-bold text-ch4">{TOP_K}</div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Vocabulary
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {results.allTerms.length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Best Score
          </div>
          <div className="font-mono text-lg font-bold text-ch4">
            {results.sorted[0]?.similarity.toFixed(3) ?? "---"}
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        JitRL MVP uses TF-IDF vectorization with cosine similarity to retrieve the top-{TOP_K} most
        relevant document chunks (~300 words each). Retrieved chunks are prepended as context to the
        prompt, enabling the model to answer questions about new documents without weight updates.
      </p>
    </div>
  );
}
