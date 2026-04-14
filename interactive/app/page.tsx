"use client";

import { useEffect } from "react";
import {
  DualMLPWidget,
  AlphaDecayWidget,
  TFIDFGateWidget,
  TTTEngineWidget,
  ForgettingWidget,
  RetrievalWidget,
  LogitBiasWidget,
  KnowledgeStoreWidget,
  RewardModulationWidget,
  ACELoopWidget,
  PlaybookWidget,
  HypernetworkWidget,
  StrategyComparisonWidget,
} from "./components/widgets";

const chapters = [
  { id: "ch1", num: "01", title: "DualMLP", color: "text-ch1" },
  { id: "ch2", num: "02", title: "TF-IDF Gate", color: "text-ch2" },
  { id: "ch3", num: "03", title: "TTT-E2E", color: "text-ch3" },
  { id: "ch4", num: "04", title: "JitRL MVP", color: "text-ch4" },
  { id: "ch5", num: "05", title: "JitRL Full", color: "text-ch5" },
  { id: "ch6", num: "06", title: "ACE", color: "text-ch6" },
  { id: "ch7", num: "07", title: "Doc-to-LoRA", color: "text-ch7" },
];

export default function Home() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) e.target.classList.add("visible");
        }),
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );
    document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen">
      {/* ---- Skip to content (a11y) ---- */}
      <a href="#main-content" className="skip-to-content">
        Skip to content
      </a>

      {/* ---- Nav ---- */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-background/80 border-b border-border" role="navigation" aria-label="Chapter navigation">
        <div className="max-w-6xl mx-auto px-5 py-2.5 flex items-center gap-1 overflow-x-auto">
          <a href="#top" className="nav-link font-bold text-foreground mr-3 tracking-tight">
            CL<span className="text-muted-foreground font-normal opacity-60">.play</span>
          </a>
          <div className="w-px h-4 bg-border mx-1 flex-shrink-0" />
          {chapters.map((ch) => (
            <a key={ch.id} href={`#${ch.id}`} className="nav-link">
              <span className={ch.color}>{ch.num}</span>
              <span className="hidden md:inline ml-1">{ch.title}</span>
            </a>
          ))}
        </div>
      </nav>

      {/* ---- Hero ---- */}
      <header id="top" className="relative overflow-hidden py-28 md:py-36">
        <div className="hero-glow bg-orange-500 top-[-200px] left-[8%]" />
        <div className="hero-glow bg-cyan-500 top-[-80px] right-[12%]" />
        <div className="hero-glow bg-green-500 bottom-[-200px] left-[35%]" />
        <div className="max-w-6xl mx-auto px-5 relative z-10">
          <p className="font-mono text-[0.7rem] text-muted-foreground tracking-[0.2em] uppercase mb-5 opacity-70">
            Interactive Explorer
          </p>
          <h1 className="text-4xl md:text-[3.5rem] lg:text-[4rem] font-bold tracking-[-0.035em] leading-[1.1] mb-7" style={{ textWrap: "balance" as never }}>
            Teach a{" "}
            <span className="text-ch1">1.5B Model</span>{" "}
            New Knowledge<br className="hidden md:block" /> Without{" "}
            <span className="text-ch3">Forgetting</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-[58ch] leading-relaxed">
            Seven chapters. Thirteen interactive widgets. Five strategies for continual learning
            in small language models: from{" "}
            <span className="text-ch1">DualMLP injection</span> and{" "}
            <span className="text-ch2">TF-IDF gating</span> through{" "}
            <span className="text-ch3">test-time training</span>,{" "}
            <span className="text-ch4">retrieval</span>,{" "}
            <span className="text-ch5">reward modulation</span>,{" "}
            <span className="text-ch6">agentic context</span>, and{" "}
            <span className="text-ch7">hypernetwork LoRA</span>.
          </p>
          <p className="text-sm text-muted-foreground/70 mt-5 font-mono tracking-wide">
            Every concept below is interactive. Click neurons, drag sliders, step through training loops.
          </p>
        </div>
      </header>

      {/* ---- Table of Contents ---- */}
      <div className="max-w-6xl mx-auto px-5 mb-20">
        <div className="bg-card border border-border rounded-2xl p-7" style={{ boxShadow: "0 4px 24px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.03)" }}>
          <h2 className="text-[0.7rem] font-mono text-muted-foreground/70 uppercase tracking-[0.15em] mb-5">
            Chapters
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { id: "ch1", num: "01", title: "DualMLP Architecture", desc: "Frozen + trainable MLPs, SwiGLU, alpha blending", color: "text-ch1", border: "border-orange-500/15 hover:border-orange-500/35" },
              { id: "ch2", num: "02", title: "TF-IDF Gradient Gating", desc: "Neuron-level gradient masking, calibration", color: "text-ch2", border: "border-cyan-500/15 hover:border-cyan-500/35" },
              { id: "ch3", num: "03", title: "TTT-E2E Engine", desc: "Mini-batch gradient descent, alpha decay", color: "text-ch3", border: "border-green-500/15 hover:border-green-500/35" },
              { id: "ch4", num: "04", title: "JitRL MVP", desc: "TF-IDF retrieval, logit biasing", color: "text-ch4", border: "border-purple-500/15 hover:border-purple-500/35" },
              { id: "ch5", num: "05", title: "JitRL Full", desc: "Knowledge store, reward modulation", color: "text-ch5", border: "border-pink-500/15 hover:border-pink-500/35" },
              { id: "ch6", num: "06", title: "ACE", desc: "Generate-Reflect-Curate loops", color: "text-ch6", border: "border-yellow-500/15 hover:border-yellow-500/35" },
              { id: "ch7", num: "07", title: "Doc-to-LoRA", desc: "Perceiver hypernetwork, rank concat", color: "text-ch7", border: "border-orange-500/15 hover:border-orange-500/35" },
            ].map((ch) => (
              <a
                key={ch.id}
                href={`#${ch.id}`}
                className={`group block p-4 rounded-xl border ${ch.border} bg-card transition-all duration-200 hover:bg-white/[0.02] active:scale-[0.98]`}
              >
                <span className={`font-mono text-[0.65rem] ${ch.color} opacity-70 group-hover:opacity-100 transition-opacity`}>{ch.num}</span>
                <h3 className="font-semibold text-sm mt-1 tracking-tight">{ch.title}</h3>
                <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">{ch.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </div>

      <main id="main-content" className="max-w-6xl mx-auto px-5 prose-dark">
        {/* ================================================================
            THE PROBLEM: CATASTROPHIC FORGETTING
            ================================================================ */}
        <section className="reveal mb-20">
          <h2 className="text-2xl md:text-3xl font-bold mb-4">The Problem: Catastrophic Forgetting</h2>
          <p>
            When a neural network learns new information through gradient updates, it tends to overwrite
            the weights that encoded prior knowledge. This is <strong>catastrophic forgetting</strong> &mdash;
            the fundamental challenge of continual learning. A model that scores 85% on general knowledge
            might drop to 40% after learning a single domain-specific document, because the same neurons
            that stored general facts got repurposed for the new content.
          </p>
          <p>
            This project implements <strong>five complementary strategies</strong> for teaching
            Qwen2.5-1.5B new knowledge at inference time without destroying what it already knows.
            Two strategies modify weights carefully (TTT-E2E, Doc-to-LoRA), while three avoid weight
            changes entirely (JitRL MVP, JitRL Full, ACE). Each trades off speed, quality, and resource
            requirements differently.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 1: DUALMPLP ARCHITECTURE
            ================================================================ */}
        <section id="ch1" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch1">01</span>
            <div>
              <h2 className="mb-0">DualMLP Architecture</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Surgical injection of trainable capacity into a frozen model
              </p>
            </div>
          </div>

          <p>
            The core architectural innovation is the <strong>DualMLP</strong>. Instead of finetuning
            the entire model (which would destroy existing knowledge) or freezing it entirely (which
            would prevent learning), we replace the MLP in layers 21&ndash;27 of Qwen2.5-1.5B with a
            dual structure: the <strong>original frozen MLP</strong> (weights locked, unchanged) plus a
            <strong> trainable copy</strong>{" "}initialized with tiny random values (std=0.001, not zeros &mdash;
            SwiGLU has dead gradient issues with zero initialization).
          </p>

          <DualMLPWidget />

          <p>
            The outputs are blended with the formula: <strong>frozen_out + (1 - &alpha;) &times; trainable_out</strong>.
            Alpha starts at 1.0, meaning the trainable MLP has zero influence initially. As documents
            are learned, alpha decays toward 0.3, gradually increasing the trainable influence. The frozen
            MLP always provides a stable baseline &mdash; this is the core mechanism preventing catastrophic
            forgetting.
          </p>

          <h3>Alpha Decay: Graduating from Student to Expert</h3>
          <p>
            The alpha parameter controls how much the model trusts its new knowledge versus its original
            training. Starting at 1.0, the model relies entirely on frozen weights.
            As it processes more mini-batches, alpha decays by 0.01 per batch, allowing the trainable
            MLP to contribute more. The floor at 0.3 ensures 70% trainable influence at most.
          </p>

          <AlphaDecayWidget />

          <p>
            Try adjusting the decay rate. A faster decay means the model trusts new knowledge sooner
            but risks destabilization. A slower decay is safer but requires more training batches
            before new knowledge has significant impact. The default of 0.01 per mini-batch reaches
            the floor of 0.3 after about 70 batches.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 2: TF-IDF GRADIENT GATING
            ================================================================ */}
        <section id="ch2" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch2">02</span>
            <div>
              <h2 className="mb-0">TF-IDF Gradient Gating</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Protecting general-purpose neurons from overwriting
              </p>
            </div>
          </div>

          <p>
            Even with the DualMLP structure, naive gradient descent on the trainable MLP can still
            damage important neurons. The insight: not all neurons are equal. Some activate broadly
            across many different inputs (general-purpose), while others are specialized to specific
            content types. <strong>TF-IDF gating</strong> identifies which neurons are general-purpose
            and zeroes out their gradients during backpropagation.
          </p>

          <TFIDFGateWidget />

          <p>
            The gate operates in two phases. First, a <strong>calibration phase</strong> runs diverse
            text through the model and measures each neuron&apos;s activation pattern. Neurons that
            activate frequently across many documents get low IDF scores (they&apos;re general).
            During learning, the gate computes TF-IDF for each neuron given the current document. Neurons
            below the threshold have their gradients masked to zero &mdash; they&apos;re protected from
            modification. Only document-specific neurons receive gradient updates.
          </p>
          <p>
            The threshold is the key hyperparameter. Lower values protect fewer neurons (faster learning,
            more forgetting risk). Higher values protect more neurons (slower learning, better preservation).
            The default of 0.3 provides a good balance for most documents.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 3: TTT-E2E ENGINE
            ================================================================ */}
        <section id="ch3" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch3">03</span>
            <div>
              <h2 className="mb-0">Test-Time Training (TTT-E2E)</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Mini-batch gradient descent at inference time
              </p>
            </div>
          </div>

          <p>
            The TTT-E2E engine puts the DualMLP and TF-IDF gate together into a complete learning
            pipeline. When a new document arrives, it&apos;s tokenized and split into 32-token
            mini-batches. For each batch, the engine performs a full forward-backward pass, applies
            the TF-IDF gradient mask, and takes an Adam optimizer step &mdash; updating only the
            trainable MLP parameters. Alpha decays after each batch.
          </p>

          <TTTEngineWidget />

          <p>
            This is the most powerful learning strategy: the model actually modifies its weights
            to encode new knowledge. The trade-off is resource cost &mdash; about 12 seconds and
            24GB of VRAM per document on an A10 GPU. The loss should decrease steadily as the model
            internalizes the document content. A flat or increasing loss suggests the learning rate
            is too high or the TF-IDF threshold is too aggressive.
          </p>

          <h3>Measuring Catastrophic Forgetting</h3>
          <p>
            After learning, we need to verify the model hasn&apos;t forgotten what it knew before.
            The <strong>forgetting ratio</strong> compares accuracy on a held-out QA set before and
            after learning. A ratio of 0 means no forgetting; 1 means total knowledge loss. The target
            is below 0.15 (15%) &mdash; the TF-IDF gate and alpha blending should keep it there.
          </p>

          <ForgettingWidget />

          <p>
            Notice that non-weight-modifying strategies (JitRL MVP, JitRL Full, ACE) have
            zero forgetting by definition &mdash; they never touch the model&apos;s weights.
            TTT-E2E and Doc-to-LoRA modify weights but use protective mechanisms to minimize damage.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 4: JITRL MVP
            ================================================================ */}
        <section id="ch4" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch4">04</span>
            <div>
              <h2 className="mb-0">JitRL MVP: Retrieval + Logit Bias</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Zero-weight-modification learning in 2 milliseconds
              </p>
            </div>
          </div>

          <p>
            What if we could teach the model without changing any weights at all? The JitRL MVP
            (Just-in-Time Retrieval Learning, Minimum Viable Product) does exactly this. Instead
            of gradient descent, it splits documents into ~300-word chunks and builds a TF-IDF
            index over them. At query time, it retrieves the most relevant chunks and uses two
            mechanisms to inject that knowledge: <strong>context prepending</strong> (RAG-style)
            and <strong>logit biasing</strong>.
          </p>

          <RetrievalWidget />

          <p>
            The retrieval step is remarkably fast &mdash; about 2 milliseconds to &quot;learn&quot;
            a document (just indexing, no gradient computation). The top-3 chunks are retrieved
            via cosine similarity against the TF-IDF vectors, then prepended to the prompt as
            context. This gives the model relevant information to draw from during generation.
          </p>

          <h3>Logit Biasing: Nudging the Output Distribution</h3>
          <p>
            Beyond context prepending, JitRL MVP also biases the model&apos;s output logits
            toward tokens that appear in the retrieved chunks. For each token in the chunks,
            its logit gets boosted proportionally to how frequently it appears. This gently steers
            the model toward using domain-specific vocabulary without overriding its general
            language modeling ability.
          </p>

          <LogitBiasWidget />

          <p>
            The bias strength is the key control. Too low and the bias has no effect. Too high
            and the model over-relies on chunk tokens, producing repetitive or nonsensical output.
            The default of 2.0 provides a good balance for most documents.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 5: JITRL FULL
            ================================================================ */}
        <section id="ch5" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch5">05</span>
            <div>
              <h2 className="mb-0">JitRL Full: Reward-Guided Modulation</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Semantic embeddings and continuous reward signals
              </p>
            </div>
          </div>

          <p>
            While JitRL MVP uses simple TF-IDF vectors, the Full variant encodes documents through
            the model itself, capturing semantic meaning in <strong>hidden-state embeddings</strong>.
            Each document is passed through the frozen model in a single forward pass, and the
            last hidden state is mean-pooled into a dense embedding vector. These are stored in a
            <strong> knowledge store</strong> for later retrieval.
          </p>

          <KnowledgeStoreWidget />

          <p>
            At query time, the query is similarly encoded and compared against all stored document
            embeddings via cosine similarity. The top-3 most relevant documents are retrieved &mdash;
            but instead of simple context prepending, JitRL Full uses a <strong>reward signal</strong>
            to modulate the model&apos;s logits continuously.
          </p>

          <h3>Reward-Guided Logit Modulation</h3>
          <p>
            The reward computation is elegant: the query and knowledge embeddings are both
            L2-normalized, then element-wise multiplied to produce a reward vector that captures
            per-dimension semantic alignment. This vector is projected through the language model
            head (lm_head.weight) to produce per-token reward logits, which are added to the
            original logits scaled by a temperature parameter.
            The result: the model&apos;s output distribution is smoothly warped toward tokens relevant
            to the retrieved knowledge.
          </p>

          <RewardModulationWidget />

          <p>
            JitRL Full is more expressive than MVP (hidden-state embeddings capture semantic
            nuance that TF-IDF misses) but currently slower (~33s generation time vs ~2.4s).
            The temperature parameter controls reward strength: too high risks over-modulation,
            too low makes the reward ineffective.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 6: ACE
            ================================================================ */}
        <section id="ch6" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch6">06</span>
            <div>
              <h2 className="mb-0">ACE: Agentic Context Engineering</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Meta-learning through LLM self-improvement loops
              </p>
            </div>
          </div>

          <p>
            ACE takes a radically different approach: instead of modifying weights or biasing logits,
            it <strong>evolves a playbook of reasoning strategies</strong> through iterative
            self-critique. Powered by a local Ollama model (7B+), it runs Generate&ndash;Reflect&ndash;Curate
            loops that progressively improve the model&apos;s approach to answering questions about
            learned documents.
          </p>

          <ACELoopWidget />

          <p>
            Each loop has three stages. The <strong>Generator</strong> answers a question using the
            document and current playbook. The <strong>Reflector</strong> critiques the answer,
            identifying strengths, weaknesses, and suggesting new strategies. The <strong>Curator</strong>
            patches the playbook: adding new strategies, removing failed ones, and updating existing rules.
            Over 3 loops, the playbook converges toward effective reasoning patterns.
          </p>

          <h3>Playbook Evolution</h3>
          <p>
            The playbook is a living list of strategies, each with an ID, a rule, and a source tag.
            It grows through successful interactions and shrinks through failures. A FIFO eviction
            policy caps the playbook at 50 strategies (configurable), preventing unbounded growth.
            Playbooks can be saved to JSON and loaded across sessions, providing persistent
            meta-learned knowledge.
          </p>

          <PlaybookWidget />

          <p>
            ACE requires no GPU (runs on Ollama&apos;s CPU/GPU) and has zero forgetting risk since
            it never touches model weights. The trade-off is speed: each loop takes ~30 seconds
            (Ollama latency), so 3 loops means ~90 seconds of learning time. The quality depends
            entirely on the Ollama model&apos;s reasoning ability.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            CHAPTER 7: DOC-TO-LORA
            ================================================================ */}
        <section id="ch7" className="reveal mb-20 scroll-mt-20">
          <div className="flex items-center gap-3 mb-6">
            <span className="font-mono text-3xl font-bold text-ch7">07</span>
            <div>
              <h2 className="mb-0">Doc-to-LoRA: Hypernetwork Generation</h2>
              <p className="text-sm text-muted-foreground mt-0 mb-0">
                Single-pass document-to-adapter conversion
              </p>
            </div>
          </div>

          <p>
            The most production-friendly strategy: a <strong>Perceiver-based hypernetwork</strong>
            reads document text and generates complete LoRA adapter matrices in a single forward pass.
            No gradient descent, no iterative optimization &mdash; just one pass through the hypernetwork
            to produce rank-8 LoRA matrices for each target MLP layer. The generated adapters are then
            injected into the base model as weight deltas: W&apos; = W + (A @ B).T.
          </p>

          <HypernetworkWidget />

          <p>
            The pipeline has five stages: chunk the document (~1024 words each), hash each chunk
            for deterministic seeding, generate LoRA A and B matrices via the hypernetwork, concatenate
            matrices from multiple chunks along the rank axis (effective rank = base_rank &times; num_chunks),
            and inject the resulting deltas into the model&apos;s MLP layers. The entire process takes
            about 0.8 seconds.
          </p>
          <p>
            Doc-to-LoRA supports two modes: <strong>doc mode</strong> for document internalization
            (splits text into chunks) and <strong>text mode</strong> for task specialization (treats
            the entire input as a single chunk, e.g., &quot;answer questions about quantum physics&quot;).
            Adapters can be removed to restore original weights, making this fully reversible.
          </p>
        </section>

        <div className="section-divider" />

        {/* ================================================================
            COMPARISON
            ================================================================ */}
        <section className="reveal mb-20">
          <h2 className="text-2xl md:text-3xl font-bold mb-4">Choosing a Strategy</h2>
          <p>
            Each strategy makes different trade-offs. The right choice depends on your constraints:
            available GPU memory, acceptable latency, forgetting tolerance, and whether you need
            deep internalization or surface-level retrieval.
          </p>

          <StrategyComparisonWidget />

          <p>
            A practical approach: start with <strong>JitRL MVP</strong> for fast prototyping (zero
            setup, instant learning). Graduate to <strong>TTT-E2E</strong> when you need deep
            knowledge internalization and have GPU resources. Use <strong>ACE</strong> to evolve
            reasoning strategies for complex QA tasks. Deploy <strong>Doc-to-LoRA</strong> in
            production for its single-pass efficiency.
          </p>
        </section>

        <div className="section-divider" />

        {/* ---- Conclusion ---- */}
        <section className="reveal mb-20">
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-8 tracking-tight">The Full Picture</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
            {[
              { num: "01", label: "Inject", desc: "DualMLP into layers 21-27", color: "text-ch1", bg: "bg-orange-500/5 border-orange-500/15 hover:border-orange-500/30" },
              { num: "02", label: "Gate", desc: "TF-IDF neuron protection", color: "text-ch2", bg: "bg-cyan-500/5 border-cyan-500/15 hover:border-cyan-500/30" },
              { num: "03", label: "Train", desc: "TTT-E2E gradient descent", color: "text-ch3", bg: "bg-green-500/5 border-green-500/15 hover:border-green-500/30" },
              { num: "04", label: "Retrieve", desc: "TF-IDF chunks + logit bias", color: "text-ch4", bg: "bg-purple-500/5 border-purple-500/15 hover:border-purple-500/30" },
              { num: "05", label: "Modulate", desc: "Reward-guided logits", color: "text-ch5", bg: "bg-pink-500/5 border-pink-500/15 hover:border-pink-500/30" },
              { num: "06", label: "Evolve", desc: "Agentic playbook loops", color: "text-ch6", bg: "bg-yellow-500/5 border-yellow-500/15 hover:border-yellow-500/30" },
              { num: "07", label: "Generate", desc: "Hypernetwork LoRA", color: "text-ch7", bg: "bg-orange-500/5 border-orange-500/15 hover:border-orange-500/30" },
              { num: "\u2713", label: "Measure", desc: "Forgetting ratio < 15%", color: "text-ch3", bg: "bg-green-500/5 border-green-500/15 hover:border-green-500/30" },
            ].map((step, i) => (
              <div key={i} className={`p-4 rounded-xl border ${step.bg} transition-all duration-200`}>
                <span className={`font-mono text-2xl font-bold ${step.color}`}>{step.num}</span>
                <h3 className="font-semibold text-sm mt-1.5 tracking-tight">{step.label}</h3>
                <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
          <p className="text-center text-muted-foreground max-w-[52ch] mx-auto leading-relaxed" style={{ textWrap: "balance" as never }}>
            Five strategies, one goal: learn without forgetting. Each approach attacks the problem
            from a different angle: weight modification, retrieval augmentation, self-critique,
            or hypernetwork generation.{" "}
            <strong className="text-foreground">
              The architecture prevents forgetting. The strategies enable learning.
            </strong>
          </p>
        </section>
      </main>

      {/* ---- Footer ---- */}
      <footer className="border-t border-border py-10 mt-20">
        <div className="max-w-6xl mx-auto px-5 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-bold text-sm tracking-tight text-foreground/80">CL<span className="text-muted-foreground/50 font-normal">.play</span></span>
            <span className="w-px h-3 bg-border" />
            <p className="text-xs text-muted-foreground/60 font-mono">
              Continual Learning for SLMs
            </p>
          </div>
          <p className="text-xs text-muted-foreground/50 font-mono">
            Next.js + React + Tailwind CSS
          </p>
        </div>
      </footer>
    </div>
  );
}
