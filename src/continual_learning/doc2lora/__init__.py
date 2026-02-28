# Doc-to-LoRA / Text-to-LoRA — Hypernetwork-generated LoRA pipeline
from continual_learning.doc2lora.engine import Doc2LoRAEngine
from continual_learning.doc2lora.chunker import DocumentChunker
from continual_learning.doc2lora.hypernetwork import SimulatedHypernetwork, PerceiverHypernetwork, PerceiverCrossAttentionBlock
from continual_learning.doc2lora.lora_injector import LoRAInjector
from continual_learning.doc2lora.meta_dataset import MetaTrainingDataset, SyntheticMetaDataset
from continual_learning.doc2lora.trainer import HypernetworkTrainer
from continual_learning.doc2lora.evaluation import Doc2LoRAEvaluator

__all__ = [
    "Doc2LoRAEngine",
    "DocumentChunker",
    "SimulatedHypernetwork",
    "PerceiverHypernetwork",
    "PerceiverCrossAttentionBlock",
    "LoRAInjector",
    "MetaTrainingDataset",
    "SyntheticMetaDataset",
    "HypernetworkTrainer",
    "Doc2LoRAEvaluator",
]
