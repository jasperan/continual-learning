import torch
from typing import Optional
from transformers import LogitsProcessor, LogitsProcessorList

from continual_learning.jitrl.base import BaseJitRLEngine
from continual_learning.jitrl.full.knowledge_store import KnowledgeStore
from continual_learning.jitrl.full.reward import RewardComputer, LogitModulator


class _RewardLogitsProcessor(LogitsProcessor):
    def __init__(self, reward: torch.Tensor, projection: torch.Tensor, temperature: float):
        self.modulator = LogitModulator(temperature=temperature)
        self.reward = reward
        self.projection = projection

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.modulator.modulate(scores, self.reward, self.projection)


class JitRLFullEngine(BaseJitRLEngine):
    """JitRL Full: Reward-guided logit modulation."""

    def __init__(self, model, tokenizer, modulation_temperature: float = 0.5, max_tokens: int = 4096):
        super().__init__(model, tokenizer)
        self.modulation_temperature = modulation_temperature
        self.max_tokens = max_tokens

        hidden_size = getattr(model.config, "hidden_size", 768)
        self._knowledge_store = KnowledgeStore(hidden_size=hidden_size)
        self._reward_computer = RewardComputer(hidden_size=hidden_size)

    def learn(self, text: str, callback: Optional[callable] = None) -> dict:
        self._documents.append(text)

        device = next(self.model.parameters()).device
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        input_ids = encoded["input_ids"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        token_embeddings = last_hidden.squeeze(0)

        doc_name = f"doc_{self._knowledge_store.num_entries}"
        self._knowledge_store.add(doc_name, token_embeddings)

        tokens_processed = input_ids.shape[1]
        if callback:
            callback(batch_idx=0, loss=0.0, tokens_so_far=tokens_processed)

        return {"tokens_processed": tokens_processed, "method": "jitrl_full"}

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        processors = LogitsProcessorList()

        if self._knowledge_store.num_entries > 0:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            query_hidden = outputs.hidden_states[-1]
            query_mean = query_hidden.mean(dim=(0, 1))

            scores, knowledge_embeddings = self._knowledge_store.query(query_mean, top_k=3)

            reward = self._reward_computer.compute(query_hidden, knowledge_embeddings)

            if hasattr(self.model, "lm_head"):
                projection = self.model.lm_head.weight.data.T.cpu()
            else:
                vocab_size = getattr(self.model.config, "vocab_size", 32000)
                hidden_size = reward.shape[0]
                projection = torch.randn(hidden_size, vocab_size) * 0.01

            processors.append(_RewardLogitsProcessor(reward, projection, self.modulation_temperature))

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                logits_processor=processors if processors else None,
            )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def clear(self) -> None:
        super().clear()
        self._knowledge_store.clear()
