from continual_learning.text_utils import answer_matches


class Doc2LoRAEvaluator:
    """Evaluation suite for Doc-to-LoRA hypernetwork quality.

    Supports three evaluation modes:
    1. QA accuracy: Does the LoRA-adapted model answer questions correctly?
    2. Needle-in-haystack: Can it find specific facts in long documents?
    3. Forgetting test: Does LoRA injection preserve base model capabilities?
    """

    def __init__(self, engine):
        """
        Args:
            engine: A Doc2LoRAEngine instance (or any object with learn/generate/clear methods).
        """
        self.engine = engine

    def evaluate_qa(
        self,
        qa_items: list[dict],
        verbose: bool = False,
    ) -> dict:
        """Evaluate QA accuracy on a list of items.

        Args:
            qa_items: List of {"context": str, "question": str, "answer": str}.
            verbose: Print per-item results.

        Returns:
            Dict with accuracy, correct_count, total, and per-item results.
        """
        results = []
        correct = 0

        for i, item in enumerate(qa_items):
            # Learn the context
            self.engine.clear()
            self.engine.learn(item["context"])

            # Generate answer
            response = self.engine.generate(item["question"])

            # Check if expected answer appears in response (substring match)
            is_correct = answer_matches(item["answer"], response)
            if is_correct:
                correct += 1

            result = {
                "question": item["question"],
                "expected": item["answer"],
                "generated": response,
                "correct": is_correct,
            }
            results.append(result)

            if verbose:
                status = "PASS" if is_correct else "FAIL"
                print(f"  [{status}] Q: {item['question'][:60]}... A: {item['answer'][:30]}")

        accuracy = correct / len(qa_items) if qa_items else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(qa_items),
            "results": results,
        }

    def evaluate_needle_in_haystack(
        self,
        needle_fact: str,
        haystack_texts: list[str],
        question: str,
        expected_answer: str,
        verbose: bool = False,
    ) -> dict:
        """Needle-in-a-haystack test: can the model find a specific fact
        embedded in a larger document?

        Args:
            needle_fact: The specific fact to find (e.g., "The secret code is 42.")
            haystack_texts: List of distractor paragraphs.
            question: Question about the needle fact.
            expected_answer: Expected answer substring.
            verbose: Print details.

        Returns:
            Dict with found (bool), response, document_length_words.
        """
        # Insert needle at the middle position in the haystack
        all_texts = list(haystack_texts)
        insert_pos = len(all_texts) // 2
        all_texts.insert(insert_pos, needle_fact)

        full_document = "\n\n".join(all_texts)
        doc_words = len(full_document.split())

        self.engine.clear()
        self.engine.learn(full_document)
        response = self.engine.generate(question)

        found = expected_answer.lower() in response.lower()

        if verbose:
            status = "FOUND" if found else "NOT FOUND"
            print(f"  [{status}] Doc: {doc_words} words, Q: {question[:50]}...")

        return {
            "found": found,
            "response": response,
            "document_length_words": doc_words,
            "needle_position": insert_pos,
            "total_paragraphs": len(all_texts),
        }

    def evaluate_forgetting(
        self,
        baseline_prompts: list[dict],
        document: str,
        threshold: float = 0.15,
        verbose: bool = False,
    ) -> dict:
        """Test if LoRA injection causes catastrophic forgetting.

        Args:
            baseline_prompts: List of {"prompt": str, "expected": str} for general knowledge.
            document: Document to learn (which might cause forgetting).
            threshold: Maximum acceptable forgetting ratio.
            verbose: Print details.

        Returns:
            Dict with forgetting_ratio, passed, before_accuracy, after_accuracy.
        """
        # Measure baseline accuracy (no LoRA)
        self.engine.clear()
        before_correct = 0
        for item in baseline_prompts:
            response = self.engine.generate(item["prompt"])
            if answer_matches(item["expected"], response):
                before_correct += 1
        before_acc = before_correct / len(baseline_prompts) if baseline_prompts else 0.0

        # Learn document
        self.engine.learn(document)

        # Measure post-learning accuracy
        after_correct = 0
        for item in baseline_prompts:
            response = self.engine.generate(item["prompt"])
            if answer_matches(item["expected"], response):
                after_correct += 1
        after_acc = after_correct / len(baseline_prompts) if baseline_prompts else 0.0

        # Compute forgetting ratio
        if before_acc > 0:
            forgetting_ratio = (before_acc - after_acc) / before_acc
        else:
            forgetting_ratio = 0.0

        passed = forgetting_ratio <= threshold

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] Forgetting: {forgetting_ratio:.4f} (threshold: {threshold})")
            print(f"  Before: {before_acc:.2%}, After: {after_acc:.2%}")

        return {
            "forgetting_ratio": forgetting_ratio,
            "passed": passed,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "threshold": threshold,
        }
