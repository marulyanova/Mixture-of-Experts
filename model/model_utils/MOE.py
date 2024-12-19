import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, embedding_dim: int, hidden: int, drop_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden)
        self.linear2 = nn.Linear(hidden, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoELayer(nn.Module):
    def __init__(
        self,
        n_experts: int,
        n_gates: int,
        embedding_dim: int,
        moe_hidden: int,
    ):
        super(MoELayer, self).__init__()

        self.gates = nn.ModuleList(
            nn.Linear(embedding_dim, n_experts) for _ in range(n_gates)
        )
        self.experts = nn.ModuleList(
            PositionwiseFeedForward(embedding_dim=embedding_dim, hidden=moe_hidden)
            for _ in range(n_experts)
        )

        self.n_gates = n_gates
        self.n_experts = n_experts

    def forward(self, x):

        gates_respond = [F.softmax(gate(x), dim=-1) for gate in self.gates]
        expert_preds = torch.stack([expert(x) for expert in self.experts])

        # gates_respond [n_gates, batch_size, n_experts]
        # expert_preds [n_experts, batch_size, embedding_dim]

        # Для каждого гейта, мы берем его отклики и умножаем на предсказания экспертов
        # Мы используем torch.einsum для более эффективного вычисления
        moe_prediction = torch.einsum("gbe,ebv->bv", gates_respond, expert_preds)

        return (
            moe_prediction,
            gates_respond,
        )  # block preds + info from gates about the selected experts
