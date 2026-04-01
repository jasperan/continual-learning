"use client";

import { useState, useMemo, useCallback, useRef, useEffect } from "react";

/* ------------------------------------------------------------------ */
/*  ACE Generate-Reflect-Curate Loop Widget                            */
/*  Interactive simulation of the agentic context engineering loop     */
/* ------------------------------------------------------------------ */

/* ---- Seeded PRNG for stable values ---- */

function seeded(seed: number): number {
  const x = Math.sin(seed * 9301 + 4973) * 49297;
  return Math.round((x - Math.floor(x)) * 1000) / 1000;
}

/* ---- Types ---- */

type Assessment = "correct" | "partially_correct" | "incorrect";

interface ReflectorOutput {
  assessment: Assessment;
  strengths: string[];
  weaknesses: string[];
  suggested_strategies: string[];
}

interface CuratorOperation {
  op: "add" | "remove" | "update";
  strategy_id: string;
  rule: string;
}

interface LoopSnapshot {
  loop: number;
  question: string;
  answer: string;
  reflection: ReflectorOutput;
  curation: CuratorOperation[];
  playbook_size: number;
}

/* ---- Sample QA about Oracle AI Vector Search ---- */

const SAMPLE_QA: { question: string; context: string }[] = [
  {
    question: "How does Oracle AI Vector Search store embeddings alongside relational data?",
    context: "Oracle AI Vector Search uses the VECTOR data type in Oracle 23ai to store dense and sparse embeddings directly in database columns, enabling hybrid queries that combine semantic similarity with traditional SQL predicates.",
  },
  {
    question: "What indexing strategies does Oracle AI Vector Search support for approximate nearest neighbor queries?",
    context: "Oracle AI Vector Search supports IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World) indexing algorithms for efficient approximate nearest neighbor search at scale.",
  },
  {
    question: "How can ONNX models be used within Oracle Database for embedding generation?",
    context: "The DBMS_VECTOR package allows loading ONNX models directly into Oracle Database, enabling in-database embedding generation without external API calls, using functions like VECTOR_EMBEDDING().",
  },
];

/* ---- Deterministic simulation data per loop ---- */

function generateAnswer(loopIdx: number, seed: number): string {
  const answers = [
    "Oracle AI Vector Search stores embeddings using the VECTOR data type in Oracle 23ai columns. It supports both dense and sparse vector formats and allows SQL-based similarity queries using cosine distance, dot product, and Euclidean distance metrics.",
    "Oracle supports IVF (Inverted File Index) for partitioned search spaces and HNSW (Hierarchical Navigable Small World) graphs for logarithmic-time approximate nearest neighbor lookups. Both can be created with CREATE VECTOR INDEX syntax.",
    "ONNX models are imported via DBMS_VECTOR.LOAD_ONNX_MODEL() and stored in the database catalog. The VECTOR_EMBEDDING() SQL function then calls the model inline during query execution to generate embeddings without round-trips to external services.",
  ];
  return answers[loopIdx % answers.length];
}

function generateReflection(loopIdx: number, seed: number): ReflectorOutput {
  const reflections: ReflectorOutput[] = [
    {
      assessment: "partially_correct",
      strengths: ["Correctly identifies VECTOR data type", "Mentions distance metrics"],
      weaknesses: ["Missing hybrid query capability detail", "No mention of partitioning integration"],
      suggested_strategies: [
        "When discussing storage, always mention hybrid SQL+vector query support",
        "Include example CREATE TABLE syntax with VECTOR columns",
      ],
    },
    {
      assessment: "correct",
      strengths: ["Covers both IVF and HNSW accurately", "Mentions CREATE VECTOR INDEX syntax"],
      weaknesses: ["Could elaborate on when to choose IVF vs HNSW"],
      suggested_strategies: [
        "Add guidance: use IVF for large static datasets, HNSW for dynamic workloads",
        "Remove generic indexing advice that contradicts Oracle-specific patterns",
      ],
    },
    {
      assessment: "incorrect",
      strengths: ["Correctly identifies DBMS_VECTOR package"],
      weaknesses: [
        "Wrong function name — should be VECTOR_EMBEDDING not EMBED_TEXT",
        "Missing LOAD_ONNX_MODEL prerequisite step",
      ],
      suggested_strategies: [
        "Always verify Oracle function names against documentation",
        "Update embedding generation strategy to include model loading step first",
        "When describing multi-step processes, enumerate prerequisite steps explicitly",
      ],
    },
  ];
  return reflections[loopIdx % reflections.length];
}

function generateCuration(
  loopIdx: number,
  seed: number,
): CuratorOperation[] {
  const hexId = (n: number) => {
    let h = "";
    for (let i = 0; i < 8; i++) {
      h += "0123456789abcdef"[Math.floor(seeded(seed + n * 31 + i * 7) * 16)];
    }
    return "s" + h;
  };

  const curations: CuratorOperation[][] = [
    [
      {
        op: "add",
        strategy_id: hexId(loopIdx * 10),
        rule: "When discussing Oracle vector storage, emphasize hybrid SQL+vector query support",
      },
      {
        op: "add",
        strategy_id: hexId(loopIdx * 10 + 1),
        rule: "Include CREATE TABLE ... (embedding VECTOR(1536)) syntax examples",
      },
    ],
    [
      {
        op: "add",
        strategy_id: hexId(loopIdx * 10 + 2),
        rule: "Recommend IVF for large static datasets, HNSW for dynamic workloads with frequent updates",
      },
      {
        op: "remove",
        strategy_id: hexId(loopIdx * 10 + 3),
        rule: "Generic indexing advice (superseded by Oracle-specific guidance)",
      },
    ],
    [
      {
        op: "update",
        strategy_id: hexId(loopIdx * 10 + 4),
        rule: "Embedding generation requires: 1) LOAD_ONNX_MODEL, 2) VECTOR_EMBEDDING() in SQL",
      },
      {
        op: "add",
        strategy_id: hexId(loopIdx * 10 + 5),
        rule: "Always verify Oracle-specific function names against official documentation",
      },
      {
        op: "add",
        strategy_id: hexId(loopIdx * 10 + 6),
        rule: "For multi-step processes, explicitly enumerate prerequisite operations",
      },
    ],
  ];
  return curations[loopIdx % curations.length];
}

/* ---- Circular flow diagram constants ---- */

const FLOW_SIZE = 200;
const FLOW_CX = FLOW_SIZE / 2;
const FLOW_CY = FLOW_SIZE / 2;
const FLOW_R = 68;

/** Compute position on circle for 3 nodes at 90deg, 210deg, 330deg */
function flowNodePos(index: number): { x: number; y: number } {
  const angles = [-90, 150, 30]; // top, bottom-left, bottom-right
  const rad = (angles[index] * Math.PI) / 180;
  return {
    x: FLOW_CX + FLOW_R * Math.cos(rad),
    y: FLOW_CY + FLOW_R * Math.sin(rad),
  };
}

/** SVG arc path between two points on a circle */
function arcPath(from: number, to: number): string {
  const p1 = flowNodePos(from);
  const p2 = flowNodePos(to);
  // Midpoint for the arc curvature
  const mx = (p1.x + p2.x) / 2 + (p1.y - p2.y) * 0.15;
  const my = (p1.y + p2.y) / 2 + (p2.x - p1.x) * 0.15;
  return `M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}`;
}

/* ---- Assessment badge colors ---- */

const ASSESSMENT_COLORS: Record<Assessment, { bg: string; text: string; border: string }> = {
  correct: { bg: "rgba(74,222,128,0.15)", text: "#4ade80", border: "rgba(74,222,128,0.3)" },
  partially_correct: { bg: "rgba(250,204,21,0.15)", text: "#facc15", border: "rgba(250,204,21,0.3)" },
  incorrect: { bg: "rgba(248,113,113,0.15)", text: "#f87171", border: "rgba(248,113,113,0.3)" },
};

/* ---- Stage labels ---- */

const STAGE_LABELS = ["Generate", "Reflect", "Curate"];
const STAGE_COLORS = ["#facc15", "#fb923c", "#4ade80"];

/* ------------------------------------------------------------------ */
/*  Main Widget                                                        */
/* ------------------------------------------------------------------ */

export default function ACELoopWidget() {
  const [maxLoops, setMaxLoops] = useState(3);
  const [currentStep, setCurrentStep] = useState(0); // 0=idle, then loop*3+stage (1-indexed)
  const [isAutoplay, setIsAutoplay] = useState(false);
  const [snapshots, setSnapshots] = useState<LoopSnapshot[]>([]);
  const [expandedLoop, setExpandedLoop] = useState<number | null>(null);
  const [showJSON, setShowJSON] = useState(false);
  const autoplayRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const SEED = 61;

  /* ---- Derived values ---- */
  const totalSteps = maxLoops * 3;
  const currentLoop = Math.floor((currentStep - 1) / 3); // 0-indexed loop
  const currentStage = (currentStep - 1) % 3; // 0=generate, 1=reflect, 2=curate
  const isRunning = currentStep > 0 && currentStep <= totalSteps;
  const isComplete = currentStep > totalSteps && snapshots.length === maxLoops;

  /* ---- Build current snapshot up to the current stage ---- */
  const activeSnapshot = useMemo(() => {
    if (!isRunning) return null;
    const qa = SAMPLE_QA[currentLoop % SAMPLE_QA.length];
    const answer = generateAnswer(currentLoop, SEED);
    const reflection = generateReflection(currentLoop, SEED);
    const curation = generateCuration(currentLoop, SEED);

    let playbookSize = 0;
    for (const s of snapshots) playbookSize = s.playbook_size;
    if (currentStage >= 2) {
      const adds = curation.filter((c) => c.op === "add").length;
      const removes = curation.filter((c) => c.op === "remove").length;
      playbookSize = playbookSize + adds - removes;
    }

    return {
      loop: currentLoop,
      question: qa.question,
      answer,
      reflection,
      curation,
      playbook_size: playbookSize,
    } as LoopSnapshot;
  }, [currentStep, currentLoop, currentStage, isRunning, snapshots]);

  /* ---- Step forward ---- */
  const stepForward = useCallback(() => {
    setCurrentStep((prev) => {
      const next = prev + 1;
      if (next > totalSteps + 1) return prev;

      // When completing curate stage (stage index 2), save snapshot
      if ((next - 1) % 3 === 0 && next > 1 && next <= totalSteps + 1) {
        // The previous step completed a full loop
        const completedLoop = Math.floor((next - 2) / 3);
        const qa = SAMPLE_QA[completedLoop % SAMPLE_QA.length];
        const answer = generateAnswer(completedLoop, SEED);
        const reflection = generateReflection(completedLoop, SEED);
        const curation = generateCuration(completedLoop, SEED);

        setSnapshots((prevSnaps) => {
          if (prevSnaps.length > completedLoop) return prevSnaps;
          let playbookSize = 0;
          if (prevSnaps.length > 0)
            playbookSize = prevSnaps[prevSnaps.length - 1].playbook_size;
          const adds = curation.filter((c) => c.op === "add").length;
          const removes = curation.filter((c) => c.op === "remove").length;
          playbookSize = playbookSize + adds - removes;

          return [
            ...prevSnaps,
            {
              loop: completedLoop,
              question: qa.question,
              answer,
              reflection,
              curation,
              playbook_size: playbookSize,
            },
          ];
        });
      }

      return next;
    });
  }, [totalSteps]);

  /* ---- Reset ---- */
  const reset = useCallback(() => {
    setCurrentStep(0);
    setSnapshots([]);
    setIsAutoplay(false);
    setExpandedLoop(null);
    if (autoplayRef.current) clearTimeout(autoplayRef.current);
  }, []);

  /* ---- Autoplay ---- */
  useEffect(() => {
    if (isAutoplay && currentStep <= totalSteps) {
      autoplayRef.current = setTimeout(() => {
        stepForward();
      }, 1200);
    } else if (currentStep > totalSteps) {
      setIsAutoplay(false);
    }
    return () => {
      if (autoplayRef.current) clearTimeout(autoplayRef.current);
    };
  }, [isAutoplay, currentStep, totalSteps, stepForward]);

  const toggleAutoplay = useCallback(() => {
    if (currentStep === 0) {
      setCurrentStep(1);
      setIsAutoplay(true);
    } else if (isComplete) {
      reset();
      setTimeout(() => {
        setCurrentStep(1);
        setIsAutoplay(true);
      }, 100);
    } else {
      setIsAutoplay((prev) => !prev);
    }
  }, [currentStep, isComplete, reset]);

  /* ---- Assessment history ---- */
  const assessmentHistory = useMemo(() => {
    return snapshots.map((s) => s.reflection.assessment);
  }, [snapshots]);

  /* ---- Total strategies ---- */
  const totalStrategies = useMemo(() => {
    if (snapshots.length === 0) return 0;
    return snapshots[snapshots.length - 1].playbook_size;
  }, [snapshots]);

  /* ---- Stats counters ---- */
  const stats = useMemo(() => {
    let added = 0, removed = 0, updated = 0;
    for (const s of snapshots) {
      for (const c of s.curation) {
        if (c.op === "add") added++;
        if (c.op === "remove") removed++;
        if (c.op === "update") updated++;
      }
    }
    return { added, removed, updated };
  }, [snapshots]);

  /* ---- Op color helper ---- */
  const opColor = (op: string) => {
    if (op === "add") return "#4ade80";
    if (op === "remove") return "#f87171";
    return "#facc15";
  };

  return (
    <div className="widget-container ch6">
      <div className="widget-label">
        <span className="text-ch6">Interactive</span>{"  ·  "}ACE Generate-Reflect-Curate Loop
      </div>

      {/* ---- Pipeline overview ---- */}
      <div className="code-block mb-5 text-center text-xs">
        <span className="text-muted-foreground">Question</span>
        <span className="text-ch6"> &rarr; </span>
        <span className="text-ch6">Generator</span>
        <span className="text-muted-foreground"> (LLM + playbook)</span>
        <span className="text-ch6"> &rarr; </span>
        <span style={{ color: "#fb923c" }}>Reflector</span>
        <span className="text-muted-foreground"> (critique)</span>
        <span className="text-ch6"> &rarr; </span>
        <span style={{ color: "#4ade80" }}>Curator</span>
        <span className="text-muted-foreground"> (patch playbook)</span>
        <span className="text-ch6"> &rarr; </span>
        <span className="text-muted-foreground">repeat</span>
      </div>

      {/* ---- Main layout: flow diagram + controls ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">

        {/* ---- Left: Circular flow diagram ---- */}
        <div className="flex flex-col items-center">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2 self-start">
            Loop Cycle
          </div>
          <div
            className="rounded-lg p-2"
            style={{
              background: "rgba(0,0,0,0.25)",
              border: "1px solid var(--border)",
            }}
          >
            <svg
              viewBox={`0 0 ${FLOW_SIZE} ${FLOW_SIZE}`}
              className="w-full"
              style={{ maxWidth: "220px", maxHeight: "220px" }}
            >
              {/* Connecting arcs */}
              {[0, 1, 2].map((from) => {
                const to = (from + 1) % 3;
                const isActive = isRunning && currentStage === from;
                return (
                  <g key={`arc-${from}`}>
                    <path
                      d={arcPath(from, to)}
                      fill="none"
                      stroke={
                        isActive
                          ? STAGE_COLORS[from]
                          : "rgba(255,255,255,0.08)"
                      }
                      strokeWidth={isActive ? 2.5 : 1.5}
                      strokeDasharray={isActive ? "none" : "4,4"}
                      markerEnd={`url(#arrow-${isActive ? "active" : "dim"})`}
                    />
                  </g>
                );
              })}

              {/* Arrow markers */}
              <defs>
                <marker
                  id="arrow-active"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto"
                >
                  <path d="M 0 0 L 10 5 L 0 10 Z" fill="#facc15" />
                </marker>
                <marker
                  id="arrow-dim"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto"
                >
                  <path
                    d="M 0 0 L 10 5 L 0 10 Z"
                    fill="rgba(255,255,255,0.15)"
                  />
                </marker>
              </defs>

              {/* Stage nodes */}
              {STAGE_LABELS.map((label, i) => {
                const pos = flowNodePos(i);
                const isActive = isRunning && currentStage === i;
                const isDone =
                  isRunning && currentStage > i
                  || (isRunning && currentLoop > 0 && !isActive)
                  || isComplete;
                const color = STAGE_COLORS[i];
                return (
                  <g key={label}>
                    {isActive && (
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={30}
                        fill={`${color}10`}
                      />
                    )}
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={22}
                      fill={
                        isActive
                          ? `${color}22`
                          : "rgba(0,0,0,0.4)"
                      }
                      stroke={isActive ? color : "rgba(255,255,255,0.12)"}
                      strokeWidth={isActive ? 2 : 1}
                    />
                    <text
                      x={pos.x}
                      y={pos.y + 1}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill={isActive ? color : "#71717a"}
                      fontSize={10}
                      fontFamily="var(--font-mono)"
                      fontWeight={isActive ? 700 : 400}
                    >
                      {label}
                    </text>
                  </g>
                );
              })}

              {/* Center loop counter */}
              <text
                x={FLOW_CX}
                y={FLOW_CY - 6}
                textAnchor="middle"
                fill="#facc15"
                fontSize={18}
                fontFamily="var(--font-mono)"
                fontWeight={700}
              >
                {isRunning ? currentLoop + 1 : isComplete ? maxLoops : 0}
              </text>
              <text
                x={FLOW_CX}
                y={FLOW_CY + 10}
                textAnchor="middle"
                fill="#71717a"
                fontSize={8}
                fontFamily="var(--font-mono)"
              >
                / {maxLoops} loops
              </text>
            </svg>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2 mt-3">
            <button
              className="btn-mono"
              onClick={stepForward}
              disabled={isAutoplay || isComplete}
              style={{
                opacity: isAutoplay || isComplete ? 0.4 : 1,
                borderColor: "rgba(250,204,21,0.3)",
                color: "#facc15",
              }}
            >
              Step
            </button>
            <button
              className={`btn-mono ${isAutoplay ? "active" : ""}`}
              onClick={toggleAutoplay}
              style={{
                borderColor: isAutoplay ? "#facc15" : "rgba(250,204,21,0.3)",
                color: isAutoplay ? "#facc15" : "#a1a1aa",
              }}
            >
              {isAutoplay ? "Pause" : isComplete ? "Replay" : currentStep === 0 ? "Autoplay" : "Resume"}
            </button>
            <button
              className="btn-mono"
              onClick={reset}
              style={{ opacity: currentStep === 0 ? 0.4 : 1 }}
              disabled={currentStep === 0}
            >
              Reset
            </button>
          </div>

          {/* Loop count slider */}
          <div className="flex items-center gap-3 mt-3 w-full max-w-[220px]">
            <label className="font-mono text-[0.6rem] text-muted-foreground whitespace-nowrap">
              loops:
            </label>
            <input
              type="range"
              min={1}
              max={5}
              value={maxLoops}
              onChange={(e) => {
                reset();
                setMaxLoops(Number(e.target.value));
              }}
              className="flex-1"
            />
            <span className="font-mono text-sm text-ch6 font-semibold w-6 text-right">
              {maxLoops}
            </span>
          </div>
        </div>

        {/* ---- Center + Right: Active step detail ---- */}
        <div className="lg:col-span-2">
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
            {isRunning
              ? `Loop ${currentLoop + 1} / ${maxLoops} — ${STAGE_LABELS[currentStage]} Stage`
              : isComplete
                ? "All Loops Complete"
                : "Click Step or Autoplay to Begin"}
          </div>

          {/* Idle state */}
          {currentStep === 0 && (
            <div className="rounded-lg border border-white/[0.06] bg-black/20 p-6 text-center">
              <div className="font-mono text-sm text-muted-foreground/50 mb-2">
                ACE Loop Simulation
              </div>
              <div className="font-mono text-[0.7rem] text-muted-foreground/40">
                The Agentic Context Engineering loop uses an LLM to generate answers,
                reflect on correctness, and curate a playbook of strategies &mdash;
                all without weight updates. Powered by Ollama for meta-learning.
              </div>
            </div>
          )}

          {/* Generate stage */}
          {isRunning && currentStage >= 0 && activeSnapshot && (
            <div className="space-y-3 animate-fade-in">
              {/* Question */}
              <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
                <div className="font-mono text-[0.6rem] text-muted-foreground mb-1">
                  Question
                </div>
                <div className="font-mono text-xs text-foreground">
                  {activeSnapshot.question}
                </div>
              </div>

              {/* Generated answer */}
              <div
                className="rounded-lg border p-3 transition-all duration-300"
                style={{
                  borderColor: currentStage === 0 ? "rgba(250,204,21,0.35)" : "rgba(255,255,255,0.06)",
                  background: currentStage === 0 ? "rgba(250,204,21,0.05)" : "rgba(0,0,0,0.2)",
                }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="font-mono text-[0.6rem] font-semibold"
                    style={{ color: "#facc15" }}
                  >
                    Generator Output
                  </span>
                  {currentStage === 0 && (
                    <span
                      className="font-mono text-[0.5rem] px-1.5 py-0.5 rounded"
                      style={{ background: "rgba(250,204,21,0.2)", color: "#facc15" }}
                    >
                      ACTIVE
                    </span>
                  )}
                </div>
                <div className="font-mono text-[0.7rem] text-muted-foreground leading-relaxed">
                  {activeSnapshot.answer}
                </div>
              </div>

              {/* Reflection (shown from stage 1+) */}
              {currentStage >= 1 && (
                <div
                  className="rounded-lg border p-3 animate-slide-in"
                  style={{
                    borderColor: currentStage === 1 ? "rgba(251,146,60,0.35)" : "rgba(255,255,255,0.06)",
                    background: currentStage === 1 ? "rgba(251,146,60,0.05)" : "rgba(0,0,0,0.2)",
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="font-mono text-[0.6rem] font-semibold"
                      style={{ color: "#fb923c" }}
                    >
                      Reflector Critique
                    </span>
                    {currentStage === 1 && (
                      <span
                        className="font-mono text-[0.5rem] px-1.5 py-0.5 rounded"
                        style={{ background: "rgba(251,146,60,0.2)", color: "#fb923c" }}
                      >
                        ACTIVE
                      </span>
                    )}
                    <span
                      className="font-mono text-[0.55rem] px-1.5 py-0.5 rounded ml-auto"
                      style={{
                        background: ASSESSMENT_COLORS[activeSnapshot.reflection.assessment].bg,
                        color: ASSESSMENT_COLORS[activeSnapshot.reflection.assessment].text,
                        border: `1px solid ${ASSESSMENT_COLORS[activeSnapshot.reflection.assessment].border}`,
                      }}
                    >
                      {activeSnapshot.reflection.assessment.replace("_", " ")}
                    </span>
                  </div>

                  <div className="code-block text-[0.65rem]">
                    <div className="text-muted-foreground">{"{"}</div>
                    <div className="pl-3">
                      <span style={{ color: "#fb923c" }}>&quot;assessment&quot;</span>
                      <span className="text-muted-foreground">: </span>
                      <span style={{ color: ASSESSMENT_COLORS[activeSnapshot.reflection.assessment].text }}>
                        &quot;{activeSnapshot.reflection.assessment}&quot;
                      </span>
                      <span className="text-muted-foreground">,</span>
                    </div>
                    <div className="pl-3">
                      <span style={{ color: "#fb923c" }}>&quot;strengths&quot;</span>
                      <span className="text-muted-foreground">: [</span>
                    </div>
                    {activeSnapshot.reflection.strengths.map((s, i) => (
                      <div key={i} className="pl-6 text-muted-foreground">
                        <span style={{ color: "#4ade80" }}>&quot;{s}&quot;</span>
                        {i < activeSnapshot.reflection.strengths.length - 1 && ","}
                      </div>
                    ))}
                    <div className="pl-3 text-muted-foreground">],</div>
                    <div className="pl-3">
                      <span style={{ color: "#fb923c" }}>&quot;weaknesses&quot;</span>
                      <span className="text-muted-foreground">: [</span>
                    </div>
                    {activeSnapshot.reflection.weaknesses.map((w, i) => (
                      <div key={i} className="pl-6 text-muted-foreground">
                        <span style={{ color: "#f87171" }}>&quot;{w}&quot;</span>
                        {i < activeSnapshot.reflection.weaknesses.length - 1 && ","}
                      </div>
                    ))}
                    <div className="pl-3 text-muted-foreground">],</div>
                    <div className="pl-3">
                      <span style={{ color: "#fb923c" }}>&quot;suggested_strategies&quot;</span>
                      <span className="text-muted-foreground">: [</span>
                    </div>
                    {activeSnapshot.reflection.suggested_strategies.map((ss, i) => (
                      <div key={i} className="pl-6 text-muted-foreground">
                        <span style={{ color: "#facc15" }}>&quot;{ss}&quot;</span>
                        {i < activeSnapshot.reflection.suggested_strategies.length - 1 && ","}
                      </div>
                    ))}
                    <div className="pl-3 text-muted-foreground">]</div>
                    <div className="text-muted-foreground">{"}"}</div>
                  </div>
                </div>
              )}

              {/* Curation (shown at stage 2) */}
              {currentStage >= 2 && (
                <div
                  className="rounded-lg border p-3 animate-slide-in"
                  style={{
                    borderColor: "rgba(74,222,128,0.35)",
                    background: "rgba(74,222,128,0.05)",
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="font-mono text-[0.6rem] font-semibold"
                      style={{ color: "#4ade80" }}
                    >
                      Curator Patches
                    </span>
                    <span
                      className="font-mono text-[0.5rem] px-1.5 py-0.5 rounded"
                      style={{ background: "rgba(74,222,128,0.2)", color: "#4ade80" }}
                    >
                      ACTIVE
                    </span>
                  </div>
                  <div className="space-y-1.5">
                    {activeSnapshot.curation.map((op, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded border p-2"
                        style={{
                          borderColor: `${opColor(op.op)}30`,
                          background: `${opColor(op.op)}08`,
                        }}
                      >
                        <span
                          className="font-mono text-[0.55rem] px-1.5 py-0.5 rounded flex-shrink-0 uppercase"
                          style={{
                            background: `${opColor(op.op)}20`,
                            color: opColor(op.op),
                          }}
                        >
                          {op.op}
                        </span>
                        <div className="flex-1">
                          <div className="font-mono text-[0.6rem] text-muted-foreground/50">
                            {op.strategy_id}
                          </div>
                          <div className="font-mono text-[0.7rem] text-muted-foreground">
                            {op.rule}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Completion state */}
          {isComplete && (
            <div className="animate-fade-in">
              <div className="rounded-lg border border-ch6/20 bg-yellow-500/5 p-4 text-center mb-4">
                <div className="font-mono text-sm text-ch6 font-semibold mb-1">
                  Loop Complete
                </div>
                <div className="font-mono text-[0.7rem] text-muted-foreground">
                  {maxLoops} loops executed. The playbook now contains {totalStrategies} strategies.
                </div>
              </div>

              {/* Assessment timeline */}
              <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground mb-2">
                Assessment History
              </div>
              <div className="flex gap-2 mb-4">
                {assessmentHistory.map((a, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-lg border p-2 text-center cursor-pointer transition-all"
                    style={{
                      borderColor: expandedLoop === i
                        ? ASSESSMENT_COLORS[a].border
                        : `${ASSESSMENT_COLORS[a].border}80`,
                      background: expandedLoop === i
                        ? ASSESSMENT_COLORS[a].bg
                        : `${ASSESSMENT_COLORS[a].bg}60`,
                    }}
                    onClick={() =>
                      setExpandedLoop(expandedLoop === i ? null : i)
                    }
                  >
                    <div className="font-mono text-[0.55rem] text-muted-foreground">
                      Loop {i + 1}
                    </div>
                    <div
                      className="font-mono text-[0.65rem] font-semibold"
                      style={{ color: ASSESSMENT_COLORS[a].text }}
                    >
                      {a.replace("_", " ")}
                    </div>
                  </div>
                ))}
              </div>

              {/* Expanded loop detail */}
              {expandedLoop !== null && snapshots[expandedLoop] && (
                <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 animate-fade-in">
                  <div className="font-mono text-[0.65rem] text-ch6 font-semibold mb-2">
                    Loop {expandedLoop + 1} Detail
                  </div>
                  <div className="space-y-2">
                    <div>
                      <span className="font-mono text-[0.6rem] text-muted-foreground">Q: </span>
                      <span className="font-mono text-[0.7rem] text-foreground">
                        {snapshots[expandedLoop].question}
                      </span>
                    </div>
                    <div>
                      <span className="font-mono text-[0.6rem] text-muted-foreground">A: </span>
                      <span className="font-mono text-[0.65rem] text-muted-foreground/80">
                        {snapshots[expandedLoop].answer.slice(0, 120)}...
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {snapshots[expandedLoop].curation.map((op, i) => (
                        <span
                          key={i}
                          className="font-mono text-[0.55rem] px-1.5 py-0.5 rounded"
                          style={{
                            background: `${opColor(op.op)}15`,
                            color: opColor(op.op),
                            border: `1px solid ${opColor(op.op)}30`,
                          }}
                        >
                          {op.op}: {op.rule.slice(0, 40)}...
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ---- Show JSON toggle ---- */}
      <button
        className={`btn-mono ${showJSON ? "active" : ""}`}
        onClick={() => setShowJSON(!showJSON)}
        style={{
          borderColor: showJSON ? "#facc15" : undefined,
          color: showJSON ? "#facc15" : undefined,
        }}
      >
        {showJSON ? "Hide" : "Show"} How It Works
      </button>
      {showJSON && (
        <div className="mt-3 code-block text-xs animate-fade-in">
          <div className="text-muted-foreground mb-2">
            ACE (Agentic Context Engineering) Loop:
          </div>
          <div>
            <span className="text-ch6">1.</span>{" "}
            <span className="text-ch6">Generator</span> receives question + current playbook strategies
          </div>
          <div>
            <span className="text-ch6">2.</span> LLM produces an answer using playbook guidance
          </div>
          <div>
            <span className="text-ch6">3.</span>{" "}
            <span style={{ color: "#fb923c" }}>Reflector</span> (separate LLM call) critiques the answer
            against ground truth
          </div>
          <div>
            <span className="text-ch6">4.</span> Produces structured assessment: correct / partially_correct / incorrect
          </div>
          <div>
            <span className="text-ch6">5.</span>{" "}
            <span style={{ color: "#4ade80" }}>Curator</span> patches the playbook based on reflection
          </div>
          <div>
            <span className="text-ch6">6.</span> Operations: add new strategies, remove outdated ones, update existing rules
          </div>
          <div className="mt-2 text-muted-foreground">
            Key insight: the model improves through better prompting strategies, not weight
            updates. The playbook is a list of natural language rules that guide future
            generation. Powered by Ollama with a 7B+ model for meta-learning calls.
          </div>
        </div>
      )}

      {/* ---- Stats row ---- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
        <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-ch6">Playbook Size</div>
          <div className="font-mono text-lg font-bold text-ch6">
            {isRunning && activeSnapshot
              ? activeSnapshot.playbook_size
              : totalStrategies}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Loops Done
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            {snapshots.length}
          </div>
        </div>
        <div className="rounded-lg border border-white/[0.06] bg-black/20 p-3 text-center">
          <div className="font-mono text-[0.6rem] text-muted-foreground">
            Added / Removed
          </div>
          <div className="font-mono text-lg font-bold text-foreground">
            <span style={{ color: "#4ade80" }}>{stats.added}</span>
            <span className="text-muted-foreground/40"> / </span>
            <span style={{ color: "#f87171" }}>{stats.removed}</span>
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
        The ACE loop enables LLM self-improvement without weight updates. A Generator produces answers
        guided by a playbook of strategies, a Reflector critiques correctness, and a Curator patches the
        playbook with add/remove/update operations. Each loop iteration refines the strategy set, improving
        future answers. Requires Ollama running locally with a 7B+ model for meta-learning calls.
      </p>
    </div>
  );
}
