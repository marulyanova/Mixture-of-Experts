import torch
import torch.nn.functional as F
from config_utils.train_params import TrainParams
from config_utils.model_params import ModelParams
from typing import Tuple


def process_gate_response(
    epoch_gates_stats: torch.tensor,
    gate_respond: torch.tensor,
    train_params: TrainParams,
    model_params: ModelParams,
) -> torch.tensor:
    if epoch_gates_stats.size(0) == 0:
        epoch_gates_stats = F.pad(
            gate_respond.flatten().reshape(
                model_params.n_encoder_blocks,
                -1,
                model_params.seq_len,
            ),
            (
                0,
                0,
                0,
                train_params.batch_size
                - gate_respond.size(0) // model_params.n_encoder_blocks,
            ),
            "constant",
            -1,
        )
    else:
        epoch_gates_stats = torch.cat(
            (
                epoch_gates_stats,
                F.pad(
                    gate_respond.flatten().reshape(
                        model_params.n_encoder_blocks,
                        -1,
                        model_params.seq_len,
                    ),
                    (
                        0,
                        0,
                        0,
                        train_params.batch_size
                        - gate_respond.size(0) // model_params.n_encoder_blocks,
                    ),
                    "constant",
                    -1,
                ),
            ),
            dim=0,
        )

    return epoch_gates_stats


def calculate_loss_accuracy(
    input_ids: torch.tensor,
    output: torch.tensor,
    labels: torch.tensor,
    train_params: TrainParams,
) -> Tuple[torch.tensor, float]:
    mask = input_ids == train_params.tokenizer_mask_id
    masked_output = output[mask]
    masked_labels = labels[mask]

    loss = F.cross_entropy(masked_output, masked_labels)
    _, predicted = torch.max(masked_output, dim=-1)
    correct_predictions = (predicted == masked_labels).sum().item()
    total_predictions = masked_labels.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return loss, accuracy


def epoch_gates_cat(
    gates_stats: torch.tensor, epoch_gates_stats: torch.tensor
) -> torch.tensor:
    if gates_stats.size(0) == 0:
        gates_stats = epoch_gates_stats
    else:
        gates_stats = torch.cat(
            (
                gates_stats,
                epoch_gates_stats,
            ),
            dim=0,
        )
    return gates_stats
