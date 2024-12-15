import torch
from torch import nn, Tensor
from utils.positional_encoding import PositionalEncoding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        device: torch.device,
    ):
        super(InputEmbedding, self).__init__()
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_len, device=device
        )

    def forward(self, x):
        # x : [batch_size, seq_len]
        embedded = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        pos_enc = self.positional_encoding(
            embedded
        )  # [seq_len, d_model] -> it will be broadcasted for each batch
        return embedded + pos_enc


class MoETransformerEncoder(nn.Module):
    def __init__(self, input_size: int, device: torch.device):
        super().__init__()

        pass
        # self.positional_encoding = PositionalEncoding(d_model = , max_len = , device = device)
