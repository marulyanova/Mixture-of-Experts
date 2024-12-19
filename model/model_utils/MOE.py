import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden: int,
        drop_prob: float = 0.1,
    ):
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


class Gate(nn.Module):
    def __init__(self, embedding_dim, n_experts, top_k):
        super(Gate, self).__init__()
        self.top_k = top_k
        self.gate = nn.Linear(embedding_dim, n_experts)

    def forward(self, x):
        gate_response = self.gate(x)
        top_k_logits, indices = gate_response.topk(self.top_k, dim=-1)
        zeros = torch.full_like(gate_response, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        gate_output = F.softmax(sparse_logits, dim=-1)
        return gate_output, indices


class MoELayer(nn.Module):
    def __init__(
        self,
        n_experts: int,
        embedding_dim: int,
        moe_hidden: int,
        n_gates: int = 1,
        top_k_experts: int = 1,
    ):
        super(MoELayer, self).__init__()

        self.gate = Gate(
            embedding_dim=embedding_dim, n_experts=n_experts, top_k=top_k_experts
        )

        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(embedding_dim=embedding_dim, hidden=moe_hidden)
                for _ in range(n_experts)
            ]
        )

        self.tok_k = top_k_experts

    def forward(self, x):

        gate_output, indices = self.gate(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gate_output.view(-1, gate_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, indices

        # # gates_respond = torch.stack([F.softmax(gate(x), dim=-1) for gate in self.gates])

        # # [batch_size, max_len, n_experts] - probabilities for each expert for esch token
        # gate_respond = self.gate(x)

        # print(gate_respond.shape, "gate respond")  # TODO: remove

        # top_1_expert = F.softmax(gate_respond, dim=-1)

        # print(top_1_expert.shape, "top_1_expert")  # TODO: remove

        # top_1_expert_preds = self.experts[top_1_expert](x)

        # # expert_preds = torch.stack([expert(x) for expert in self.experts]).reshape(
        # #     self.n_gates, self.n_experts, batch_size, max_len, embbeding_size
        # # )

        # print(top_1_expert_preds.shape, "top_1_expert_preds")  # TODO: remove

        # # gates_respond [n_gates, batch_size, seq_len, n_experts]
        # # expert_preds  [n_experts, batch_size, seq_len, embedding_dim]

        # # Для каждого гейта, мы берем его отклики и умножаем на предсказания экспертов
        # # Мы используем torch.einsum для более эффективного вычисления
        # moe_prediction = torch.einsum("gbs,esv->bv", gates_respond, expert_preds)

        # return (
        #     moe_prediction,
        #     gates_respond,
        # )  # block preds + info from gates about the selected experts
