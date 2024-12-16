import torch.nn as nn
import torch.nn.functional as F
import torch


# class ScaleDotProductAttention(nn.Module):
#     """
#     compute scale dot product attention

#     Query : given sentence that we focused on (decoder)
#     Key : every sentence to check relationship with Qeury(encoder)
#     Value : every sentence same with Key (encoder)
#     """

#     def __init__(self):
#         super(ScaleDotProductAttention, self).__init__()
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, q, k, v, mask=None, e=1e-12):
#         # input is 4 dimension tensor
#         # [batch_size, head, length, d_tensor]
#         batch_size, head, length, d_tensor = k.size()

#         # 1. dot product Query with Key^T to compute similarity
#         k_t = k.transpose(2, 3)  # transpose
#         score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

#         # 2. apply masking (opt)
#         if mask is not None:
#             score = score.masked_fill(mask == 0, -10000)

#         # 3. pass them softmax to make [0, 1] range
#         score = self.softmax(score)

#         # 4. multiply with Value
#         v = score @ v

#         return v, score


# class MultiHeadAttention(nn.Module):

#     def __init__(self, embedding_dim: int, head_dim: int, n_head: int):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.attention = ScaleDotProductAttention()
#         self.w_q = nn.Linear(embedding_dim, head_dim)
#         self.w_k = nn.Linear(embedding_dim, head_dim)
#         self.w_v = nn.Linear(embedding_dim, head_dim)
#         self.w_concat = nn.Linear(embedding_dim, embedding_dim)

#     def forward(self, q, k, v, mask=None):
#         # 1. dot product with weight matrices
#         q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

#         # 2. split tensor by number of heads
#         q, k, v = self.split(q), self.split(k), self.split(v)

#         # 3. do scale dot product to compute similarity
#         out, attention = self.attention(q, k, v, mask=mask)

#         # 4. concat and pass to linear layer
#         out = self.concat(out)
#         out = self.w_concat(out)

#         return out

#     def split(self, tensor):
#         """
#         split tensor by number of head

#         :param tensor: [batch_size, length, d_model]
#         :return: [batch_size, head, length, d_tensor]
#         """
#         batch_size, length, d_model = tensor.size()

#         d_tensor = d_model // self.n_head
#         tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
#         # it is similar with group convolution (split by number of heads)

#         return tensor

#     def concat(self, tensor):
#         """
#         inverse function of self.split(tensor : torch.Tensor)

#         :param tensor: [batch_size, head, length, d_tensor]
#         :return: [batch_size, length, d_model]
#         """
#         batch_size, head, length, d_tensor = tensor.size()
#         d_model = head * d_tensor

#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
#         return tensor


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
