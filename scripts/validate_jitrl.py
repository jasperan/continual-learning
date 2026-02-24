#!/usr/bin/env python3
"""GPU validation for JitRL MVP and Full engines."""

import sys
import time
import torch

sys.path.insert(0, "src")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print_section("JITRL COMPARISON VALIDATION")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from continual_learning.jitrl.mvp.engine import JitRLMVPEngine
    from continual_learning.jitrl.full.engine import JitRLFullEngine
    from continual_learning.jitrl.comparison import ComparisonHarness

    print("  Loading Qwen2.5-1.5B (unmodified for JitRL)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", dtype=torch.float32, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    print("  Model loaded.")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {allocated:.1f}GB / {total:.1f}GB")

    mvp = JitRLMVPEngine(model=model, tokenizer=tokenizer, top_k=3, bias_strength=2.0)
    full = JitRLFullEngine(model=model, tokenizer=tokenizer, modulation_temperature=0.5)

    doc = """Oracle AI Vector Search enables storing and searching vector
    embeddings alongside business data in Oracle Database. It supports
    approximate nearest neighbor search using HNSW and IVF indexes.
    The feature is available in Oracle Database 23ai and later versions.
    It integrates with popular AI frameworks and supports multiple
    distance metrics including cosine similarity and dot product."""

    qa_items = [
        {"question": "What is Oracle AI Vector Search?", "answer": "vector"},
        {"question": "What indexes does Oracle AI Vector Search support?", "answer": "HNSW"},
        {"question": "Which database version supports AI Vector Search?", "answer": "23"},
    ]

    print_section("INDIVIDUAL ENGINE TESTS")

    # Test MVP
    print("  Testing JitRL MVP...")
    t0 = time.time()
    mvp.learn(doc)
    mvp_learn_time = time.time() - t0
    print(f"  MVP learn time: {mvp_learn_time:.3f}s")

    t0 = time.time()
    mvp_response = mvp.generate("What is Oracle AI Vector Search?", max_new_tokens=100)
    mvp_gen_time = time.time() - t0
    print(f"  MVP generate time: {mvp_gen_time:.3f}s")
    print(f"  MVP response: {mvp_response[:200]}")

    # Test Full
    print("\n  Testing JitRL Full...")
    t0 = time.time()
    full.learn(doc)
    full_learn_time = time.time() - t0
    print(f"  Full learn time: {full_learn_time:.3f}s")

    t0 = time.time()
    full_response = full.generate("What is Oracle AI Vector Search?", max_new_tokens=100)
    full_gen_time = time.time() - t0
    print(f"  Full generate time: {full_gen_time:.3f}s")
    print(f"  Full response: {full_response[:200]}")

    print_section("COMPARISON HARNESS")

    mvp.clear()
    full.clear()

    harness = ComparisonHarness(engines={"JitRL MVP": mvp, "JitRL Full": full})
    results = harness.run_comparison([doc], qa_items, max_new_tokens=100)

    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    Accuracy: {r['accuracy']:.2%} ({r['correct']}/{r['num_evaluated']})")
        print(f"    Learn time: {r['learn_time_s']:.3f}s")
        print(f"    Eval time: {r['eval_time_s']:.3f}s")
        print(f"    Tokens learned: {r['tokens_learned']}")

    print_section("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
