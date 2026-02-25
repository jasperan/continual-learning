import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Playbook:
    """Evolving context playbook for ACE. Stores strategies as a list of rules."""

    def __init__(self, max_strategies: int = 50):
        self.max_strategies = max_strategies
        self.strategies: list[dict] = []
        self.stats = {
            "total_loops": 0,
            "documents_processed": 0,
        }
        self.created = datetime.now(timezone.utc).isoformat()
        self.updated = self.created

    def add_strategy(self, rule: str, source: str) -> str:
        sid = f"s{uuid.uuid4().hex[:8]}"
        self.strategies.append({"id": sid, "rule": rule, "source": source})
        while len(self.strategies) > self.max_strategies:
            self.strategies.pop(0)
        self._touch()
        return sid

    def update_strategy(self, strategy_id: str, new_rule: str) -> None:
        for s in self.strategies:
            if s["id"] == strategy_id:
                s["rule"] = new_rule
                self._touch()
                return
        raise KeyError(f"Strategy {strategy_id} not found")

    def remove_strategy(self, strategy_id: str) -> None:
        self.strategies = [s for s in self.strategies if s["id"] != strategy_id]
        self._touch()

    def render(self) -> str:
        if not self.strategies:
            return "No strategies yet. Answer questions based on the provided document context."
        lines = ["Strategies for answering questions:"]
        for i, s in enumerate(self.strategies, 1):
            lines.append(f"{i}. {s['rule']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "created": self.created,
            "updated": self.updated,
            "strategies": self.strategies,
            "stats": self.stats,
            "max_strategies": self.max_strategies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Playbook":
        pb = cls(max_strategies=data.get("max_strategies", 50))
        pb.strategies = data.get("strategies", [])
        pb.stats = data.get("stats", {"total_loops": 0, "documents_processed": 0})
        pb.created = data.get("created", pb.created)
        pb.updated = data.get("updated", pb.updated)
        return pb

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Playbook":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def _touch(self):
        self.updated = datetime.now(timezone.utc).isoformat()
