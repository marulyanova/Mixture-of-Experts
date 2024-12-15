import torch
from torch import nn
from utils.positional_encoding import InputEmbedding
from utils.multi_head_attention import MultiHeadAttention
from utils.layer_norm import LayerNorm


class MoETransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        seq_len: int,
        num_head: int,
        device: torch.device,
    ):
        """
        vocab_size: vocabulary size of tokenizer
        input_size: embedding size obtained by the MoE layer
        seq_len: max sequence length
        num_head: number of heads in multi head attention
        device: device of model
        """
        super().__init__()

        self.input_emb = InputEmbedding(
            vocab_size=vocab_size,
            d_model=input_size,
            max_len=seq_len,
            device=device,
        )

        self.multi_head_attention = MultiHeadAttention(
            d_model=input_size, n_head=num_head
        )

        self.layer_norm = LayerNorm(d_model=input_size)

    def forward(self, x):
        input = self.input_emb(x)
        attention = self.multi_head_attention(input)
        normed = self.layer_norm(attention + input)
