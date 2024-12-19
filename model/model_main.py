import torch
from torch import nn
from model.model_utils.positional_encoding import InputEmbedding
from model.model_utils.multi_head_attention import MultiHeadAttention_Parallel
from model.model_utils.layer_norm import LayerNorm
from model.model_utils.MOE import MoELayer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seq_len: int,
        num_head: int,
        n_experts: int,
        n_gates: int,
        head_dim: int,
        moe_dim: int,
        device: torch.device,
    ):
        """
        vocab_size: vocabulary size of tokenizer
        embedding_dim: embedding size obtained by the MoE layer
        seq_len: max sequence length
        num_head: number of heads in multi head attention
        n_experts: number of experts for each gate
        n_gates: nubmer of gates on MOE-block
        head_dim: dimensional of Q, K, W
        moe_dim: dimensional of one expert's FFN layer
        device: device of model
        """
        super(EncoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention_Parallel(
            embed_size=embedding_dim,
            num_heads=num_head,
            head_size=head_dim,
            block_size=seq_len,
        )

        self.layer_norm1 = LayerNorm(embedding_dim=embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim=embedding_dim)

        self.moe_block = MoELayer(
            n_experts=n_experts,
            n_gates=n_gates,
            embedding_dim=embedding_dim,
            moe_hidden=moe_dim,
        )

    def forward(self, input):

        # input: [batch_size, seq_len, embedding_dim]

        attention = self.multi_head_attention(input)

        # attention: [batch_size, seq_len, embedding_dim]

        normed = self.layer_norm1(attention + input)

        # normed: [batch_size, seq_len, embedding_dim]

        moe_output, gates_respond = self.moe_block(
            normed
        )  # return gates_respond for further analytics. TODO write extra logic for process it

        # moe_output: [batch_size, seq_len, embedding_dim]
        # gates_respond: [n_gates, batch_size, n_experts]

        normed = self.layer_norm2(moe_output + normed)
        # normed: [batch_size, seq_len, embedding_dim]

        return normed


class MoETransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seq_len: int,
        num_head: int,
        n_experts: int,
        n_gates: int,
        n_encoder_blocks: int,
        head_dim: int,
        moe_dim: int,
        device: str,
    ):
        """
        n_encoder_blocks: number of encoder blocks in the architecture
        """
        super(MoETransformerEncoder, self).__init__()

        self.input_emb = InputEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=seq_len,
            device=device,
        )

        self.moe_transformer = nn.ModuleList(
            [
                EncoderBlock(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    seq_len=seq_len,
                    num_head=num_head,
                    n_experts=n_experts,
                    n_gates=n_gates,
                    head_dim=head_dim,
                    moe_dim=moe_dim,
                    device=device,
                )
            ]
            for _ in range(n_encoder_blocks)
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):

        # x: [batch_size, seq_len]

        input = self.input_emb(x)

        # input: [batch_size, seq_len, embedding_dim]

        transformer_output = self.moe_transformer(input)

        # transformer_output: [batch_size, seq_len, embedding_dim]

        preds = self.lm_head(transformer_output)

        # preds: [batch_size, seq_len, vocab_size]

        return preds
