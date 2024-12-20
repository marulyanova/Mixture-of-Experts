import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import json
from pathlib import Path
from typing import Dict

import click
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AdamW

from config_utils.load_config import (
    DataParamsSchema,
    ModelParamsSchema,
    TrainParamsSchema,
    load_params_from_yaml,
)
from model.model_main import MoETransformerEncoder
from train.train_utils.data import PrepareDataloader, PrepareDataset


@click.command()
@click.option("--config-name", type=Path, required=True)
def main(config_name):

    # LOAD PARAMS

    model_params = load_params_from_yaml(config_name, ModelParamsSchema)
    loaded_params = load_params_from_yaml("dataset_params.yaml", DataParamsSchema)
    train_params = load_params_from_yaml("train_params.yaml", TrainParamsSchema)

    # ВОСПРОИЗВОДИМОСТЬ ЭКСПЕРИМЕНТОВ

    set_seed(model_params.random_seed)
    torch.cuda.manual_seed(model_params.random_seed)
    np.random.seed(model_params.random_seed)
    torch.manual_seed(model_params.random_seed)

    # DATASET

    train_loaded, test_loaded = torch.load(loaded_params.data_params.masked_data_path)

    dataset = PrepareDataset(
        train_loaded,
        test_loaded,
        loaded_params.load_params.valid_len / loaded_params.load_params.train_len,
    )

    train_dataloader, val_dataloader, test_dataloader = PrepareDataloader(
        dataset, train_params
    )

    # ACCELERATOR

    accelerator = Accelerator(
        gradient_accumulation_steps=train_params.gradient_accumulation_steps,
        project_dir=".",
        log_with="aim",
    )
    accelerator.init_trackers(
        train_params.experiment_name, config=json.loads(train_params.model_dump_json())
    )

    model = MoETransformerEncoder(**model_params.__dict__)
    optimizer = AdamW(
        model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )
    total_steps = train_params.n_epochs * len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_params.warmup_proportion),
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, val_dataloader, test_dataloader, scheduler = (
        accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            scheduler,
        )
    )

    # PROCESS OF TRAINING

    train_gates_stats = torch.tensor([])  # [n_epochs, n_layers, batch_size, max_len]
    val_gates_stats = torch.tensor([])  # [хз что, n_layers, batch_size, max_len]

    with tqdm(desc="Training", total=total_steps) as pbar:
        for epoch in range(train_params.epochs):
            for batch_i, batch in enumerate(train_dataloader):
                epoch_gates_stats = torch.tensor([])  # [n_layers, batch_size, max_len]
                current_step = batch_i + epoch * len(train_dataloader)
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                with accelerator.accumulate(model):
                    output, gate_respond = model(input_ids)

                    # extend and reshape to nessesary form [n_layers, batch_size, max_len]
                    epoch_gates_stats.extend(gate_respond.flatten()).reshape(
                        model_params.n_encoder_blocks,
                        train_params.batch_size,
                        model_params.seq_len,
                    )
                    mask = input_ids == train_params["tokenizer_mask_id"]
                    masked_output = output[mask]
                    masked_labels = labels[mask]

                    loss = F.cross_entropy(masked_output, masked_labels)
                    _, predicted = torch.max(masked_output, dim=-1)
                    correct_predictions = (predicted == masked_labels).sum().item()
                    total_predictions = masked_labels.size(0)
                    accuracy = (
                        correct_predictions / total_predictions
                        if total_predictions > 0
                        else 0.0
                    )

                    accelerator.log(
                        {"train_batch_loss": loss.item()}, step=current_step + 1
                    )
                    accelerator.log(
                        {"train_batch_accuracy": accuracy}, step=current_step + 1
                    )
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if (
                    current_step + 1
                ) % train_params.eval_steps == 0 or current_step + 1 == total_steps:
                    model.eval()
                    with tqdm(desc="Eval", total=len(val_dataloader)) as eval_pbar:
                        epoch_gates_stats_val = torch.tensor([])
                        with torch.no_grad():
                            for eval_batch in val_dataloader:
                                input_ids, attention_mask, labels = (
                                    eval_batch["input_ids"],
                                    eval_batch["attention_mask"],
                                    eval_batch["labels"],
                                )
                                output, gates_respond_val = model(input_ids)

                                epoch_gates_stats_val.extend(
                                    gates_respond_val.flatten()
                                ).reshape(
                                    model_params.n_encoder_blocks,
                                    train_params.batch_size,
                                    model_params.seq_len,
                                )

                                mask = input_ids == train_params["tokenizer_mask_id"]
                                masked_output = output[mask]
                                masked_labels = labels[mask]

                                loss = F.cross_entropy(masked_output, masked_labels)
                                _, predicted = torch.max(masked_output, dim=-1)
                                correct_predictions = (
                                    (predicted == masked_labels).sum().item()
                                )
                                total_predictions = masked_labels.size(0)
                                accuracy = (
                                    correct_predictions / total_predictions
                                    if total_predictions > 0
                                    else 0.0
                                )
                                eval_pbar.update(1)
                                accelerator.log(
                                    {"accuracy eval": accuracy}, step=current_step + 1
                                )
                    model.train()

                if (
                    current_step + 1
                ) % model_params.save_steps == 0 or current_step + 1 == total_steps:
                    accelerator.wait_for_everyone()
                    accelerator.save_model(
                        model, train_params.save_path / f"step_{current_step + 1}"
                    )

            train_gates_stats.extend(epoch_gates_stats)
            val_gates_stats.extend(epoch_gates_stats_val)

    accelerator.end_training()

    # TODO: add the processing of gates_respond


if __name__ == "__main__":
    main()
