import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiHeadAttention_Parallel(nn.Module):

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size * num_heads, bias=False)
        self.query = nn.Linear(embed_size, head_size * num_heads, bias=False)
        self.value = nn.Linear(embed_size, head_size * num_heads, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_size = head_size

        self.proj = nn.Linear(head_size * num_heads, embed_size)

    def forward(self, x):
        # input of size (B, T, C)
        # output of size (B, T, D * num_heads)

        B, T, C = x.shape
        k = (
            self.key(x).reshape(B, T, self.num_heads, self.head_size).transpose(1, 2)
        )  # (B, head_size, T, num_heads)
        q = (
            self.query(x).reshape(B, T, self.num_heads, self.head_size).transpose(1, 2)
        )  # (B, head_size, T, num_heads)
        v = (
            self.value(x).reshape(B, T, self.num_heads, self.head_size).transpose(1, 2)
        )  # (B, head_size, T, num_heads)

        masks = self.tril[:T, :T].unsqueeze(0).unsqueeze(0)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v

        out = (
            out.transpose(1, 2).contiguous().view(B, T, self.head_size * self.num_heads)
        )
        out = self.dropout(self.proj(out))

        return out  # [batch_size, seq_len, embedding_dim]
