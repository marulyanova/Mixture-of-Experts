import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, embedding_dim: int, max_len: int, device: torch.device):
        """
        constructor of sinusoid encoding class

        :param embedding_dim: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, embedding_dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, embedding_dim, step=2, device=device).float()
        # 'i' means index of embedding_dim (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embedding_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embedding_dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, embedding_dim = 512]

        # print(f"input pos enc shape: {x.shape}")  # TODO: remove

        _, seq_len, _ = x.size()
        # [batch_size (_) = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, embedding_dim = 512]
        # it will add with tok_emb : [128, 30, 512]


# input embedding + positional encoding
class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_len: int,
        device: torch.device,
    ):
        super(InputEmbedding, self).__init__()
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim, max_len=max_len, device=device
        )

    def forward(self, x):
        # x : [batch_size, seq_len]

        # print(f"input shape: {x.shape}")  # TODO: remove

        embedded = self.input_embedding(x)  # [batch_size, seq_len, embedding_dim]

        # print(f"embedded shape: {x.shape}")  # TODO: remove

        pos_enc = self.positional_encoding(
            embedded
        )  # [seq_len, embedding_dim] -> it will be broadcasted for each batch
        return embedded + pos_enc
