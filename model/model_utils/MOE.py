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

        self.gates = nn.Sequential(
            *nn.ModuleList(nn.Linear(embedding_dim, n_experts) for _ in range(n_gates))
        )
        self.experts = nn.Sequential(
            *nn.ModuleList(
                PositionwiseFeedForward(embedding_dim=embedding_dim, hidden=moe_hidden)
                for _ in range(n_experts * n_gates)
            )
        )

        self.n_gates = n_gates
        self.n_experts = n_experts

    def forward(self, x):

        batch_size, max_len, embbeding_size = x.size()

        gates_respond = torch.stack([F.softmax(gate(x), dim=-1) for gate in self.gates])
        print(gates_respond.shape, "gate respond")  # TODO: remove
        expert_preds = torch.stack([expert(x) for expert in self.experts]).reshape(
            self.n_gates, self.n_experts, batch_size, max_len, embbeding_size
        )
        print(expert_preds.shape, "expert preds")  # TODO: remove

        # gates_respond [n_gates, batch_size, seq_len, n_experts]
        # expert_preds  [n_experts, batch_size, seq_len, embedding_dim]

        # Для каждого гейта, мы берем его отклики и умножаем на предсказания экспертов
        # Мы используем torch.einsum для более эффективного вычисления
        moe_prediction = torch.einsum("gbs,esv->bv", gates_respond, expert_preds)

        return (
            moe_prediction,
            gates_respond,
        )  # block preds + info from gates about the selected experts
