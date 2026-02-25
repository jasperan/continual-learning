import json
from continual_learning.ace.ollama_client import OllamaClient
from continual_learning.ace.playbook import Playbook


class Curator:
    """Curator role: patch-updates the playbook based on reflector feedback. Never rewrites fully."""

    SYSTEM_PROMPT = (
        "You are a curation agent. Given a playbook's current strategies and reflection feedback, "
        "produce a JSON patch with these fields:\n"
        '- "add": list of new strategy strings to add\n'
        '- "remove": list of strategy IDs to remove\n'
        '- "update": list of objects {"id": "strategy_id", "rule": "new rule text"}\n'
        "Only make minimal, targeted changes. Do NOT rewrite everything. "
        "Respond with ONLY valid JSON, no other text."
    )

    def __init__(self, client: OllamaClient):
        self.client = client

    def curate(self, playbook: Playbook, feedback: dict, loop_num: int) -> None:
        prompt = (
            f"Current playbook strategies:\n{playbook.render()}\n\n"
            f"Reflection feedback:\n{json.dumps(feedback, indent=2)}\n\n"
            f"Produce a JSON patch to improve the playbook:"
        )

        raw = self.client.generate(prompt=prompt, system=self.SYSTEM_PROMPT, temperature=0.3)
        patch = self._parse_patch(raw)

        if patch:
            self._apply_patch(playbook, patch, loop_num)
        else:
            for rule in feedback.get("suggested_strategies", []):
                playbook.add_strategy(rule, source=f"loop_{loop_num}")

        playbook.stats["total_loops"] = loop_num

    def _parse_patch(self, raw: str) -> dict | None:
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _apply_patch(self, playbook: Playbook, patch: dict, loop_num: int) -> None:
        for sid in patch.get("remove", []):
            try:
                playbook.remove_strategy(sid)
            except KeyError:
                pass

        for update in patch.get("update", []):
            try:
                playbook.update_strategy(update["id"], update["rule"])
            except KeyError:
                pass

        for rule in patch.get("add", []):
            playbook.add_strategy(rule, source=f"loop_{loop_num}")
