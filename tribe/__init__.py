"""Tribe: Self-organizing expert lifecycle for continual learning.

A graph of experts that bond, specialize, shed, recycle, and settle —
knowledge naturally migrates to its correct location over generations.
"""

from tribe.core import State, TribeMember, Tribe
from tribe.expert import make_expert, forward, forward_batch, loss_on, train, clone, blend_weights
from tribe.distill import distill
from tribe.patterns import pattern_id, patterns_match, make_clustered_patterns
from tribe.metrics import (measure_knowledge_precision, measure_system_loss,
                           measure_redundancy)
from tribe.cnn import make_cnn_expert, cnn_forward_batch
from tribe.mnist import load_mnist, make_mnist_patterns
from tribe.router import SwitchRouter
from tribe.lora import make_lora_expert, lora_param_count, vit_lora_forward

__all__ = [
    'State', 'TribeMember', 'Tribe',
    'make_expert', 'forward', 'forward_batch', 'loss_on', 'train', 'clone', 'blend_weights',
    'distill',
    'pattern_id', 'patterns_match', 'make_clustered_patterns',
    'measure_knowledge_precision', 'measure_system_loss', 'measure_redundancy',
    'make_cnn_expert', 'cnn_forward_batch',
    'load_mnist', 'make_mnist_patterns',
    'SwitchRouter',
    'make_lora_expert', 'lora_param_count', 'vit_lora_forward',
]
