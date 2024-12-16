import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
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
        input_size: int,
        vocab_size: int,
    ):
        super(MoELayer, self).__init__()

        self.gates = nn.ModuleList(
            [nn.Linear(input_size, n_experts) for _ in range(n_gates)]
        )
        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(d_model=input_size, hidden=vocab_size)
                for _ in range(n_experts)
            ]
        )  # predict probabilities of tokens in vocabulary

        self.n_gates = n_gates
        self.n_experts = n_experts

    def forward(self, x):

        gates_respond = [
            F.softmax(gate(x), dim=-1) for gate in self.gates
        ]  # [n_gates, batch_size, n_experts]
        expert_preds = torch.stack(
            [expert(x) for expert in self.experts]
        )  # [n_experts, batch_size, vocab_size]

        gates_respond = [F.softmax(self.gates[i](x)) for i in range(self.n_gates)]
        expert_preds = [F.softmax(self.experts[i](x)) for i in range(self.n_experts)]

        # gates_respond [n_gates, batch_size, n_experts]
        # expert_preds [n_experts, batch_size, vocab_size]

        # Для каждого гейта, мы берем его отклики и умножаем на предсказания экспертов
        # Мы используем torch.einsum для более эффективного вычисления
        moe_prediction = torch.einsum("gbe,ebv->bv", gates_respond, expert_preds)

        return (
            moe_prediction,
            gates_respond,
        )  # block preds + info from gates about the selected experts
