import torch
from torch import nn
from utils.positional_encoding import InputEmbedding
from utils.multi_head_attention import MultiHeadAttention
from utils.layer_norm import LayerNorm
from utils.MOE import MoELayer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        seq_len: int,
        num_head: int,
        n_experts: int,
        n_gates: int,
        device: torch.device,
    ):
        """
        vocab_size: vocabulary size of tokenizer
        input_size: embedding size obtained by the MoE layer
        seq_len: max sequence length
        num_head: number of heads in multi head attention
        n_experts: number of experts for each gate
        n_gates: nubmer of gates on MOE-block
        device: device of model
        """
        super(EncoderBlock).__init__()

        self.multi_head_attention = MultiHeadAttention(
            d_model=input_size, n_head=num_head
        )

        self.layer_norm1 = LayerNorm(d_model=input_size)
        self.layer_norm2 = LayerNorm(d_model=input_size)

        self.moe_block = MoELayer(
            n_experts=n_experts,
            n_gates=n_gates,
            input_size=input_size,
            vocab_size=vocab_size,
        )

    def forward(self, input):
        attention = self.multi_head_attention(input)
        normed = self.layer_norm1(attention + input)
        moe_output, gates_respond = self.moe_block(
            normed
        )  # return gates_respond for further analytics. TODO write extra logic for process it
        normed = self.layer_norm2(moe_output + normed)
        return normed


class MoETransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        seq_len: int,
        num_head: int,
        n_experts: int,
        n_gates: int,
        n_encoder_blocks: int,
        device: torch.device,
    ):
        """
        n_encoder_blocks: number of encoder blocks in the architecture
        """
        super(MoETransformerEncoder).__init__()

        self.input_emb = InputEmbedding(
            vocab_size=vocab_size,
            d_model=input_size,
            max_len=seq_len,
            device=device,
        )

        self.moe_transformer = nn.ModuleList(
            [
                EncoderBlock(
                    vocab_size=vocab_size,
                    input_size=input_size,
                    seq_len=seq_len,
                    num_head=num_head,
                    n_experts=n_experts,
                    n_gates=n_gates,
                    device=device,
                )
            ]
            for _ in range(n_encoder_blocks)
        )

    def forward(self, x):
        input = self.input_emb(x)
        transformer_output = self.moe_transformer(input)
        return transformer_output
