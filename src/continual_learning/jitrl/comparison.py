import time
from continual_learning.jitrl.base import BaseJitRLEngine


class ComparisonHarness:
    """Runs identical benchmarks across multiple engines for A/B comparison."""

    def __init__(self, engines: dict[str, BaseJitRLEngine]):
        self.engines = engines

    def run_comparison(self, learn_texts: list[str], qa_items: list[dict],
                       max_new_tokens: int = 50) -> dict[str, dict]:
        results = {}

        for name, engine in self.engines.items():
            engine.clear()

            t0 = time.time()
            total_tokens = 0
            for text in learn_texts:
                metrics = engine.learn(text)
                total_tokens += metrics.get("tokens_processed", 0)
            learn_time = time.time() - t0

            t0 = time.time()
            correct = 0
            total = 0
            for item in qa_items:
                if not item.get("answer"):
                    continue
                response = engine.generate(
                    f"Answer concisely: {item['question']}",
                    max_new_tokens=max_new_tokens,
                )
                if item["answer"].lower() in response.lower():
                    correct += 1
                total += 1
            eval_time = time.time() - t0

            results[name] = {
                "accuracy": correct / total if total > 0 else 0.0,
                "num_evaluated": total,
                "correct": correct,
                "learn_time_s": learn_time,
                "eval_time_s": eval_time,
                "tokens_learned": total_tokens,
                "method": engine.__class__.__name__,
            }

        return results
