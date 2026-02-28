import json
import random
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


class MetaTrainingDataset(Dataset):
    """SQuAD-based dataset for hypernetwork meta-training.

    Each item yields a triple:
    - context: document passage (truncated to max_context_words)
    - question: a question about the passage
    - answer: the expected answer

    Used for teacher-student training where:
    - Teacher sees context + question (full context)
    - Student sees only question (with hypernetwork-generated LoRA from context)
    """

    def __init__(
        self,
        split: str = "train",
        max_context_words: int = 256,
        min_context_words: int = 32,
        max_items: Optional[int] = None,
        seed: int = 42,
    ):
        self.max_context_words = max_context_words
        self.min_context_words = min_context_words
        self._items: list[dict] = []
        self._load_squad(split, max_items, seed)

    def _load_squad(self, split, max_items, seed):
        """Load SQuAD v2 from HuggingFace datasets."""
        from datasets import load_dataset

        squad = load_dataset("rajpurkar/squad_v2", split=split)

        rng = random.Random(seed)
        items = []

        for item in squad:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]["text"]

            # Skip items without answers (SQuAD v2 has unanswerable questions)
            if not answers:
                continue

            # Skip contexts that are too short
            words = context.split()
            if len(words) < self.min_context_words:
                continue

            # Truncate long contexts
            if len(words) > self.max_context_words:
                context = " ".join(words[:self.max_context_words])

            items.append({
                "context": context,
                "question": question,
                "answer": answers[0],  # Take first answer
            })

        rng.shuffle(items)

        if max_items is not None:
            items = items[:max_items]

        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    @property
    def contexts(self) -> list[str]:
        """All unique contexts."""
        return list(set(item["context"] for item in self._items))


class SyntheticMetaDataset(Dataset):
    """Synthetic dataset for testing without downloading SQuAD.

    Generates simple factual passages with corresponding QA pairs.
    """

    def __init__(self, num_items: int = 100, seed: int = 42):
        rng = random.Random(seed)
        self._items = []

        templates = [
            {
                "context": "The {entity} was founded in {year} by {founder}. It is headquartered in {city}.",
                "question": "When was {entity} founded?",
                "answer": "{year}",
            },
            {
                "context": "{entity} is a {type} that specializes in {field}. It has {count} employees worldwide.",
                "question": "What does {entity} specialize in?",
                "answer": "{field}",
            },
            {
                "context": "The capital of {country} is {capital}. It has a population of {pop} million people.",
                "question": "What is the capital of {country}?",
                "answer": "{capital}",
            },
        ]

        entities = ["Oracle", "Google", "Microsoft", "Apple", "Amazon", "Meta", "Tesla", "Intel", "NVIDIA", "IBM"]
        years = ["1977", "1998", "1975", "1976", "1994", "2004", "2003", "1968", "1993", "1911"]
        founders = ["Larry Ellison", "Larry Page", "Bill Gates", "Steve Jobs", "Jeff Bezos", "Mark Zuckerberg", "Elon Musk", "Robert Noyce", "Jensen Huang", "Charles Flint"]
        cities = ["Austin", "Mountain View", "Redmond", "Cupertino", "Seattle", "Menlo Park", "Palo Alto", "Santa Clara", "Santa Clara", "Armonk"]
        types = ["company", "corporation", "enterprise", "organization", "firm"]
        fields = ["databases", "search", "software", "hardware", "cloud", "social media", "electric vehicles", "chips", "GPUs", "computing"]
        countries = ["France", "Japan", "Germany", "Brazil", "India", "Australia", "Canada", "Mexico", "Spain", "Italy"]
        capitals = ["Paris", "Tokyo", "Berlin", "Brasilia", "New Delhi", "Canberra", "Ottawa", "Mexico City", "Madrid", "Rome"]

        for i in range(num_items):
            template = rng.choice(templates)
            idx = i % len(entities)

            item = {
                "context": template["context"].format(
                    entity=entities[idx], year=years[idx], founder=founders[idx],
                    city=cities[idx], type=rng.choice(types), field=fields[idx],
                    count=rng.randint(1000, 500000), country=countries[idx % len(countries)],
                    capital=capitals[idx % len(capitals)], pop=rng.randint(1, 100),
                ),
                "question": template["question"].format(
                    entity=entities[idx], country=countries[idx % len(countries)],
                ),
                "answer": template["answer"].format(
                    year=years[idx], field=fields[idx],
                    capital=capitals[idx % len(capitals)],
                ),
            }
            self._items.append(item)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
