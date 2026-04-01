"use client";

import { useState, useMemo, useCallback, useEffect, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Playbook Evolution Widget                                          */
/*  Shows how the playbook (list of strategies) evolves over ACE loops */
/* ------------------------------------------------------------------ */

/* ---- Seeded PRNG for stable values ---- */

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Types ---- */

interface Strategy {
  id: string;
  rule: string;
  source: string; // "loop_N"
  status: "active" | "removed" | "updated";
  addedAt: number; // loop index
  removedAt?: number;
  updatedAt?: number;
  previousRule?: string;
  history: { loop: number; action: string; rule: string }[];
}

interface LoopEvent {
  loop: number;
  operations: {
    op: "add" | "remove" | "update";
    strategyId: string;
    rule: string;
  }[];
}

/* ---- Generate deterministic hex IDs ---- */

function hexId(seed: number, n: number): string {
  let h = "";
  for (let i = 0; i < 8; i++) {
    h += "0123456789abcdef"[Math.floor(seeded(seed + n * 31 + i * 7) * 16)];
  }
  return "s" + h;
}

/* ---- Deterministic loop events ---- */

const SEED = 67;

function buildLoopEvents(numLoops: number): LoopEvent[] {
  const strategies = [
    "When discussing Oracle vector storage, emphasize hybrid SQL+vector query support",
    "Include CREATE TABLE ... (embedding VECTOR(1536)) syntax examples in answers",
    "Recommend IVF for large static datasets, HNSW for dynamic workloads",
    "Always verify Oracle-specific function names against official documentation",
    "For multi-step processes, explicitly enumerate prerequisite operations",
    "Embedding generation requires: 1) LOAD_ONNX_MODEL, 2) VECTOR_EMBEDDING() in SQL",
    "Mention cosine similarity, dot product, and Euclidean distance as supported metrics",
    "Reference DBMS_VECTOR package for all in-database AI operations",
    "When comparing indexing, note HNSW has O(log n) query time vs IVF O(sqrt(n))",
    "Include memory and storage requirements when discussing vector indexes",
    "Cite Oracle 23ai as the version introducing native VECTOR data type",
    "Distinguish between dense and sparse embeddings in storage recommendations",
    "Always mention partitioning benefits for tables with >1M vectors",
    "Note that ONNX model import supports transformer architectures natively",
    "Include JSON document support alongside vector search capabilities",
    "Recommend batch embedding generation for datasets >10K documents",
    "Reference SELECT-FROM-WHERE with VECTOR_DISTANCE() for similarity queries",
    "Mention that approximate search trades recall for speed at scale",
    "Include connection pooling advice for high-throughput embedding workloads",
    "Add EXPLAIN PLAN guidance for debugging vector query performance",
    "Note OML4Py integration for Python-based vector pipeline orchestration",
    "Include cross-partition vector search with global index considerations",
    "Recommend dimension reduction for embeddings >2048 dimensions",
    "Always note GPU acceleration availability for HNSW index building",
    "Mention hybrid search combining keyword BM25 with vector cosine similarity",
  ];

  const events: LoopEvent[] = [];

  for (let loop = 0; loop < numLoops; loop++) {
    const ops: LoopEvent["operations"] = [];

    // Each loop adds 2-4 strategies
    const numAdds = 2 + Math.floor(seeded(SEED + loop * 100) * 3);
    for (let j = 0; j < numAdds && loop * 4 + j < strategies.length; j++) {
      const idx = loop * 3 + j;
      if (idx < strategies.length) {
        ops.push({
          op: "add",
          strategyId: hexId(SEED, loop * 10 + j),
          rule: strategies[idx],
        });
      }
    }

    // After loop 2, start removing some old strategies
    if (loop >= 2) {
      const removeIdx = loop - 2;
      ops.push({
        op: "remove",
        strategyId: hexId(SEED, removeIdx * 10),
        rule: strategies[removeIdx * 3] ?? "Outdated strategy",
      });
    }

    // After loop 1, start updating some strategies
    if (loop >= 1 && loop % 2 === 1) {
      const updateIdx = loop - 1;
      const origIdx = updateIdx * 3 + 1;
      if (origIdx < strategies.length) {
        ops.push({
          op: "update",
          strategyId: hexId(SEED, updateIdx * 10 + 1),
          rule: strategies[origIdx] + " (refined with additional context)",
        });
      }
    }

    events.push({ loop, operations: ops });
  }

  return events;
}

/* ---- Op color helper ---- */

function opColor(op: string): string {
  if (op === "add") return "#4ade80";
  if (op === "remove") return "#f87171";
  return "#facc15";
}

/* ---- Max loops for the widget ---- */

const MAX_AVAILABLE_LOOPS = 8;

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function PlaybookWidget() {
  const [maxStrategies, setMaxStrategies] = useState(50);
  const [currentLoop, setCurrentLoop] = useState(0);
  const [isAutoplay, setIsAutoplay] = useState(false);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null);
  const [showJSON, setShowJSON] = useState(false);
  const autoplayRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* ---- Build all events ---- */
  const allEvents = useMemo(
    () => buildLoopEvents(MAX_AVAILABLE_LOOPS),
    [],
  );

  /* ---- Build strategy state at current loop ---- */
  const { strategies, stats } = useMemo(() => {
    const stratMap = new Map<string, Strategy>();
    let totalAdded = 0;
    let totalRemoved = 0;
    let totalUpdated = 0;

    for (let i = 0; i <= currentLoop && i < allEvents.length; i++) {
      const event = allEvents[i];
      for (const op of event.operations) {
        if (op.op === "add") {
          totalAdded++;
          const existing = stratMap.get(op.strategyId);
          if (existing) {
            // Re-adding a removed strategy
            existing.status = "active";
            existing.removedAt = undefined;
            existing.history.push({
              loop: i,
              action: "re-added",
              rule: op.rule,
            });
          } else {
            stratMap.set(op.strategyId, {
              id: op.strategyId,
              rule: op.rule,
              source: `loop_${i + 1}`,
              status: "active",
              addedAt: i,
              history: [{ loop: i, action: "added", rule: op.rule }],
            });
          }
        } else if (op.op === "remove") {
          totalRemoved++;
          const existing = stratMap.get(op.strategyId);
          if (existing) {
            existing.status = "removed";
            existing.removedAt = i;
            existing.history.push({
              loop: i,
              action: "removed",
              rule: existing.rule,
            });
          }
        } else if (op.op === "update") {
          totalUpdated++;
          const existing = stratMap.get(op.strategyId);
          if (existing) {
            existing.previousRule = existing.rule;
            existing.rule = op.rule;
            existing.status = "updated";
            existing.updatedAt = i;
            existing.history.push({
              loop: i,
              action: "updated",
              rule: op.rule,
            });
          }
        }
      }
    }

    // Apply FIFO eviction
    let strats = Array.from(stratMap.values());
    const activeStrats = strats.filter((s) => s.status !== "removed");
    if (activeStrats.length > maxStrategies) {
      // Sort by addedAt ascending, evict oldest
      const sorted = [...activeStrats].sort((a, b) => a.addedAt - b.addedAt);
      const toEvict = sorted.slice(0, activeStrats.length - maxStrategies);
      for (const s of toEvict) {
        const strat = stratMap.get(s.id);
        if (strat) {
          strat.status = "removed";
          strat.removedAt = currentLoop;
          strat.history.push({
            loop: currentLoop,
            action: "evicted (FIFO)",
            rule: strat.rule,
          });
          totalRemoved++;
        }
      }
      strats = Array.from(stratMap.values());
    }

    return {
      strategies: strats,
      stats: {
        total: strats.filter((s) => s.status !== "removed").length,
        added: totalAdded,
        removed: totalRemoved,
        updated: totalUpdated,
      },
    };
  }, [currentLoop, allEvents, maxStrategies]);

  /* ---- Active and removed lists ---- */
  const activeStrategies = useMemo(
    () => strategies.filter((s) => s.status !== "removed"),
    [strategies],
  );
  const removedStrategies = useMemo(
    () => strategies.filter((s) => s.status === "removed"),
    [strategies],
  );

  /* ---- Selected strategy detail ---- */
  const selectedStrategy = useMemo(
    () =>
      selectedStrategyId
        ? strategies.find((s) => s.id === selectedStrategyId)
        : null,
    [selectedStrategyId, strategies],
  );

  /* ---- New strategies at current loop (for animation) ---- */
  const newAtCurrentLoop = useMemo(() => {
    if (currentLoop === 0) return new Set(strategies.map((s) => s.id));
    const event = allEvents[currentLoop];
    if (!event) return new Set<string>();
    return new Set(
      event.operations
        .filter((op) => op.op === "add")
        .map((op) => op.strategyId),
    );
  }, [currentLoop, allEvents, strategies]);

  /* ---- Removed at current loop ---- */
  const removedAtCurrentLoop = useMemo(() => {
    if (currentLoop === 0) return new Set<string>();
    const event = allEvents[currentLoop];
    if (!event) return new Set<string>();
    return new Set(
      event.operations
        .filter((op) => op.op === "remove")
        .map((op) => op.strategyId),
    );
  }, [currentLoop, allEvents]);

  /* ---- Step forward / back ---- */
  const stepForward = useCallback(() => {
    setCurrentLoop((prev) =>
      Math.min(prev + 1, allEvents.length - 1),
    );
  }, [allEvents.length]);

  const stepBack = useCallback(() => {
    setCurrentLoop((prev) => Math.max(prev - 1, 0));
  }, []);

  const reset = useCallback(() => {
    setCurrentLoop(0);
    setIsAutoplay(false);
    setSelectedStrategyId(null);
    if (autoplayRef.current) clearTimeout(autoplayRef.current);
  }, []);

  /* ---- Autoplay ---- */
  useEffect(() => {
    if (isAutoplay && currentLoop < allEvents.length - 1) {
      autoplayRef.current = setTimeout(() => {
        stepForward();
      }, 1500);
    } else if (currentLoop >= allEvents.length - 1) {
      setIsAutoplay(false);
    }
    return () => {
      if (autoplayRef.current) clearTimeout(autoplayRef.current);
    };
  }, [isAutoplay, currentLoop, allEvents.length, stepForward]);

  const toggleAutoplay = useCallback(() => {
    if (currentLoop >= allEvents.length - 1) {
      reset();
      setTimeout(() => setIsAutoplay(true), 100);
    } else {
      setIsAutoplay((prev) => !prev);
    }
  }, [currentLoop, allEvents.length, reset]);

  /* ---- JSON export data ---- */
  const playbookJSON = useMemo(() => {
    return {
      playbook: {
        max_strategies: maxStrategies,
        current_size: activeStrategies.length,
        strategies: activeStrategies.map((s) => ({
          id: s.id,
          rule: s.rule,
          source: s.source,
        })),
      },
    };
  }, [activeStrategies, maxStrategies]);

  /* ---- FIFO bar fill ---- */
  const fifoFill = useMemo(() => {
    return Math.min(100, (activeStrategies.length / maxStrategies) * 100);
  }, [activeStrategies.length, maxStrategies]);

  return (
    <div className="widget-container ch6">
      <div className="widget-label">
        <span className="text-ch6">Interactive</span>{"  ·  "}Playbook Evolution
      </div>

      {/* ---- Pipeline overview ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">Empty Playbook</span>
        <span className="text-ch6"> &rarr; </span>
        <span style={{ color: "#4ade80" }}>+add</span>
        <span className="text-muted-foreground"> strategies</span>
        <span className="text-ch6"> &rarr; </span>
        <span style={{ color: "#facc15" }}>~update</span>
        <span className="text-muted-foreground"> rules</span>
        <span className="text-ch6"> &rarr; </span>
        <span style={{ color: "#f87171" }}>-remove</span>
        <span className="text-muted-foreground"> outdated</span>
        <span className="text-ch6"> &rarr; </span>
        <span className="text-muted-foreground">FIFO eviction at max</span>
      </div>

      {/* ---- Controls row ---- */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <button
          className="btn-mono"
          onClick={stepBack}
          disabled={currentLoop === 0 || isAutoplay}
          style={{
            opacity: currentLoop === 0 || isAutoplay ? 0.4 : 1,
            borderColor: "rgba(250,204,21,0.3)",
            color: "#facc15",
          }}
        >
          &larr; Prev
        </button>
        <button
          className="btn-mono"
          onClick={stepForward}
          disabled={currentLoop >= allEvents.length - 1 || isAutoplay}
          style={{
            opacity: currentLoop >= allEvents.length - 1 || isAutoplay ? 0.4 : 1,
            borderColor: "rgba(250,204,21,0.3)",
            color: "#facc15",
          }}
        >
          Next &rarr;
        </button>
        <button
          className={`btn-mono ${isAutoplay ? "active" : ""}`}
          onClick={toggleAutoplay}
          style={{
            borderColor: isAutoplay ? "#facc15" : "rgba(250,204,21,0.3)",
            color: isAutoplay ? "#facc15" : "#a1a1aa",
          }}
        >
          {isAutoplay
            ? "Pause"
            : currentLoop >= allEvents.length - 1
              ? "Replay"
              : "Autoplay"}
        </button>
        <button
          className="btn-mono"
          onClick={reset}
          style={{ opacity: currentLoop === 0 ? 0.4 : 1 }}
          disabled={currentLoop === 0}
        >
          Reset
        </button>
        <div className="flex-1" />
        <div className="font-mono text-xs text-muted-foreground">
          Loop{" "}
          <span className="text-ch6 font-semibold">{currentLoop + 1}</span>
          <span className="text-muted-foreground/40"> / {allEvents.length}</span>
        </div>
      </div>

      {/* ---- FIFO capacity bar ---- */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1">
          <span className="font-mono text-[0.6rem] text-muted-foreground">
            Playbook Capacity
          </span>
          <span className="font-mono text-[0.6rem] text-muted-foreground">
            {activeStrategies.length} / {maxStrategies}
            {fifoFill >= 90 && (
              <span style={{ color: "#f87171" }}> (near limit)</span>
            )}
          </span>
        </div>
        <div className="h-2.5 rounded bg-white/[0.06] overflow-hidden">
          <div
            className="h-full rounded transition-all duration-500 ease-out"
            style={{
              width: `${fifoFill}%`,
              background:
                fifoFill >= 90
                  ? "rgba(248,113,113,0.6)"
                  : fifoFill >= 70
                    ? "rgba(250,204,21,0.5)"
                    : "rgba(74,222,128,0.5)",
            }}
          />
        </div>
        <div className="flex items-center gap-3 mt-2">
          <label className="font-mono text-[0.6rem] text-muted-foreground whitespace-nowrap">
            max_strategies:
          </label>
          <input
            type="range"
            min={5}
            max={50}
            value={maxStrategies}
            onChange={(e) => setMaxStrategies(Number(e.target.value))}
            className="flex-1 max-w-[250px]"
          />
          <span className="font-mono text-sm text-ch6 font-semibold w-8 text-right">
            {maxStrategies}
          </span>
        </div>
      </div>

      {/* ---- Main layout: strategies + detail ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">

        {/* ---- Left + Center: Strategy cards ---- */}
        <div className="lg:col-span-2">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Active Strategies ({activeStrategies.length})
          </div>

          {activeStrategies.length === 0 ? (
            <div className="rounded-lg border border-white/[0.06] bg-black/20 p-6 text-center">
              <span className="font-mono text-xs text-muted-foreground/50">
                Playbook is empty. Advance loops to add strategies.
              </span>
            </div>
          ) : (
            <div className="space-y-1.5" style={{ maxHeight: "400px", overflowY: "auto" }}>
              {activeStrategies.map((strat) => {
                const isNew = newAtCurrentLoop.has(strat.id);
                const isSelected = selectedStrategyId === strat.id;
                const isUpdated = strat.status === "updated" && strat.updatedAt === currentLoop;
                return (
                  <div
                    key={strat.id}
                    className={`strategy-card ${isNew ? "new" : ""} cursor-pointer`}
                    onClick={() =>
                      setSelectedStrategyId(
                        isSelected ? null : strat.id,
                      )
                    }
                    style={{
                      borderColor: isSelected
                        ? "rgba(250,204,21,0.5)"
                        : isUpdated
                          ? "rgba(250,204,21,0.3)"
                          : isNew
                            ? "rgba(74,222,128,0.3)"
                            : undefined,
                      background: isSelected
                        ? "rgba(250,204,21,0.08)"
                        : isUpdated
                          ? "rgba(250,204,21,0.05)"
                          : undefined,
                    }}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="font-mono text-[0.55rem] text-muted-foreground/40">
                        {strat.id}
                      </span>
                      <span
                        className="font-mono text-[0.5rem] px-1 py-0.5 rounded"
                        style={{
                          background: "rgba(250,204,21,0.12)",
                          color: "#facc15",
                        }}
                      >
                        {strat.source}
                      </span>
                      {isNew && (
                        <span
                          className="font-mono text-[0.5rem] px-1 py-0.5 rounded"
                          style={{
                            background: "rgba(74,222,128,0.15)",
                            color: "#4ade80",
                          }}
                        >
                          new
                        </span>
                      )}
                      {isUpdated && (
                        <span
                          className="font-mono text-[0.5rem] px-1 py-0.5 rounded"
                          style={{
                            background: "rgba(250,204,21,0.15)",
                            color: "#facc15",
                          }}
                        >
                          updated
                        </span>
                      )}
                    </div>
                    <div className="font-mono text-[0.7rem] text-muted-foreground leading-relaxed">
                      {strat.rule}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Removed strategies (collapsed) */}
          {removedStrategies.length > 0 && (
            <div className="mt-3">
              <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground/50 mb-1.5">
                Removed ({removedStrategies.length})
              </div>
              <div className="space-y-1">
                {removedStrategies.map((strat) => {
                  const justRemoved = removedAtCurrentLoop.has(strat.id);
                  return (
                    <div
                      key={strat.id}
                      className="strategy-card removed cursor-pointer"
                      onClick={() =>
                        setSelectedStrategyId(
                          selectedStrategyId === strat.id ? null : strat.id,
                        )
                      }
                      style={{
                        borderColor: justRemoved
                          ? "rgba(248,113,113,0.3)"
                          : undefined,
                        opacity: justRemoved ? 0.5 : 0.3,
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[0.55rem] text-muted-foreground/30">
                          {strat.id}
                        </span>
                        {justRemoved && (
                          <span
                            className="font-mono text-[0.5rem] px-1 py-0.5 rounded"
                            style={{
                              background: "rgba(248,113,113,0.15)",
                              color: "#f87171",
                            }}
                          >
                            removed
                          </span>
                        )}
                      </div>
                      <div className="font-mono text-[0.65rem] text-muted-foreground/40">
                        {strat.rule.length > 80
                          ? strat.rule.slice(0, 77) + "..."
                          : strat.rule}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* ---- Right: Strategy history + current loop ops ---- */}
        <div>
          {/* Current loop operations */}
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Loop {currentLoop + 1} Operations
          </div>
          {allEvents[currentLoop] && (
            <div className="space-y-1.5 mb-5">
              {allEvents[currentLoop].operations.map((op, i) => (
                <div
                  key={i}
                  className="rounded border p-2 transition-all animate-slide-in"
                  style={{
                    borderColor: `${opColor(op.op)}30`,
                    background: `${opColor(op.op)}08`,
                    animationDelay: `${i * 100}ms`,
                  }}
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span
                      className="font-mono text-[0.5rem] px-1.5 py-0.5 rounded uppercase"
                      style={{
                        background: `${opColor(op.op)}20`,
                        color: opColor(op.op),
                      }}
                    >
                      {op.op}
                    </span>
                    <span className="font-mono text-[0.55rem] text-muted-foreground/40">
                      {op.strategyId}
                    </span>
                  </div>
                  <div className="font-mono text-[0.65rem] text-muted-foreground">
                    {op.rule.length > 70 ? op.rule.slice(0, 67) + "..." : op.rule}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Selected strategy history */}
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            Strategy History
          </div>
          {selectedStrategy ? (
            <div className="rounded-lg border border-ch6/20 bg-yellow-500/5 p-3 animate-fade-in">
              <div className="font-mono text-[0.6rem] text-ch6 font-semibold mb-1">
                {selectedStrategy.id}
              </div>
              <div className="font-mono text-[0.7rem] text-muted-foreground mb-3">
                {selectedStrategy.rule}
              </div>
              <div className="space-y-1.5">
                {selectedStrategy.history.map((h, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 font-mono text-[0.6rem]"
                  >
                    <span className="text-muted-foreground/40 flex-shrink-0">
                      loop_{h.loop + 1}
                    </span>
                    <span
                      className="px-1 py-0.5 rounded flex-shrink-0"
                      style={{
                        background:
                          h.action === "added" || h.action === "re-added"
                            ? "rgba(74,222,128,0.15)"
                            : h.action === "removed" ||
                                h.action.startsWith("evicted")
                              ? "rgba(248,113,113,0.15)"
                              : "rgba(250,204,21,0.15)",
                        color:
                          h.action === "added" || h.action === "re-added"
                            ? "#4ade80"
                            : h.action === "removed" ||
                                h.action.startsWith("evicted")
                              ? "#f87171"
                              : "#facc15",
                        fontSize: "0.55rem",
                      }}
                    >
                      {h.action}
                    </span>
                    <span className="text-muted-foreground/60 break-all">
                      {h.rule.length > 50 ? h.rule.slice(0, 47) + "..." : h.rule}
                    </span>
                  </div>
                ))}
              </div>
              {selectedStrategy.previousRule && (
                <div className="mt-2 pt-2 border-t border-white/[0.06]">
                  <div className="font-mono text-[0.55rem] text-muted-foreground/40 mb-0.5">
                    Previous rule:
                  </div>
                  <div className="font-mono text-[0.6rem] text-muted-foreground/50 line-through">
                    {selectedStrategy.previousRule}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
              <span className="font-mono text-[0.65rem] text-muted-foreground/40">
                Click a strategy card to see its history
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ---- JSON export toggle ---- */}
      <button
        className={`btn-mono ${showJSON ? "active" : ""}`}
        onClick={() => setShowJSON(!showJSON)}
        style={{
          borderColor: showJSON ? "#facc15" : undefined,
          color: showJSON ? "#facc15" : undefined,
        }}
      >
        {showJSON ? "Hide" : "Show"} Playbook JSON
      </button>
      {showJSON && (
        <div className="mt-3 code-block text-xs animate-fade-in overflow-x-auto" style={{ maxHeight: "300px", overflowY: "auto" }}>
          <div className="text-muted-foreground">{"{"}</div>
          <div className="pl-3">
            <span className="text-ch6">&quot;playbook&quot;</span>
            <span className="text-muted-foreground">: {"{"}</span>
          </div>
          <div className="pl-6">
            <span className="text-ch6">&quot;max_strategies&quot;</span>
            <span className="text-muted-foreground">: </span>
            <span className="text-foreground">{maxStrategies}</span>
            <span className="text-muted-foreground">,</span>
          </div>
          <div className="pl-6">
            <span className="text-ch6">&quot;current_size&quot;</span>
            <span className="text-muted-foreground">: </span>
            <span className="text-foreground">{activeStrategies.length}</span>
            <span className="text-muted-foreground">,</span>
          </div>
          <div className="pl-6">
            <span className="text-ch6">&quot;strategies&quot;</span>
            <span className="text-muted-foreground">: [</span>
          </div>
          {activeStrategies.map((s, i) => (
            <div key={s.id} className="pl-9">
              <span className="text-muted-foreground">{"{ "}</span>
              <span className="text-ch6">&quot;id&quot;</span>
              <span className="text-muted-foreground">: </span>
              <span style={{ color: "#4ade80" }}>&quot;{s.id}&quot;</span>
              <span className="text-muted-foreground">, </span>
              <span className="text-ch6">&quot;rule&quot;</span>
              <span className="text-muted-foreground">: </span>
              <span className="text-foreground/70">
                &quot;{s.rule.length > 60 ? s.rule.slice(0, 57) + "..." : s.rule}&quot;
              </span>
              <span className="text-muted-foreground">, </span>
              <span className="text-ch6">&quot;source&quot;</span>
              <span className="text-muted-foreground">: </span>
              <span style={{ color: "#facc15" }}>&quot;{s.source}&quot;</span>
              <span className="text-muted-foreground">
                {" }"}
                {i < activeStrategies.length - 1 ? "," : ""}
              </span>
            </div>
          ))}
          <div className="pl-6 text-muted-foreground">]</div>
          <div className="pl-3 text-muted-foreground">{"}"}</div>
          <div className="text-muted-foreground">{"}"}</div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch6">Active</div>
          <div className="font-mono text-lg font-bold text-ch6">
            {stats.total}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Total Added
          </div>
          <div className="font-mono text-lg font-bold" style={{ color: "#4ade80" }}>
            {stats.added}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Removed
          </div>
          <div className="font-mono text-lg font-bold" style={{ color: "#f87171" }}>
            {stats.removed}
          </div>
        </div>
        <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch6">Updated</div>
          <div className="font-mono text-lg font-bold text-ch6">
            {stats.updated}
          </div>
        </div>
      </div>

      <p className="font-mono text-[0.65rem] text-muted-foreground/60 mt-3">
        The playbook is a FIFO-bounded list of natural language strategies that guide the Generator.
        Each ACE loop iteration may add new strategies from the Reflector&apos;s suggestions, remove
        outdated rules that led to errors, or update existing strategies with refined guidance. When
        the playbook reaches max_strategies ({maxStrategies}), the oldest entries are evicted first.
      </p>
    </div>
  );
}
