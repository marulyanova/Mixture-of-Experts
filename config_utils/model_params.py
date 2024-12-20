from dataclasses import dataclass
import torch


@dataclass
class ModelParams:
    vocab_size: int
    embedding_dim: int
    seq_len: int
    num_head: int
    n_experts: int
    n_gates: int
    top_k_experts: int
    n_encoder_blocks: int
    head_dim: int
    moe_dim: int
    device: str
    random_seed: int
