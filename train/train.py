import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from typing import Dict
import yaml
import json
import torch
import numpy as np
from accelerate.utils import set_seed, tqdm
import datasets
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW
import transformers
import torch.nn.functional as F
from torchmetrics import Accuracy


import click
from pathlib import Path

from config_utils.load_config import (
    load_params_from_yaml,
    ModelParamsSchema,
    DataParamsSchema,
)
from model.model_main import MoETransformerEncoder
from train.train_utils.data import PrepareDataset, PrepareDataloader


@click.command()
@click.option("--config-name", type=Path, required=True)
def main(config_name):
    model_params = load_params_from_yaml(config_name, ModelParamsSchema)
    loaded_params = load_params_from_yaml("dataset_params.yaml", DataParamsSchema)
    with open("../configs/train_params.yaml", "r") as f:
        train_params = yaml.safe_load(f)

    set_seed(model_params["random_seed"])
    torch.cuda.manual_seed(model_params["random_seed"])
    np.random.seed(model_params["random_seed"])
    torch.manual_seed(model_params["random_seed"])

    train_loaded, test_loaded = torch.load(
        loaded_params["data_params"]["masked_data_path"]
    )

    dataset = PrepareDataset(
        train_loaded,
        test_loaded,
        loaded_params["load_params"]["valid_len"]
        / loaded_params["load_params"]["train_len"],
    )

    train_dataloader, val_dataloader, test_dataloader = PrepareDataloader(
        dataset, train_params
    )

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

    model, optimizer, train_dataloader, val_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, test_dataloader
        )
    )

    total_steps = train_params.n_epochs * len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_params.warmup_proportion),
        num_training_steps=total_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    # PROCESS OF TRAINING

    gates_stats = []

    with tqdm(desc="Training", total=total_steps) as pbar:
        for epoch in range(train_params.epochs):
            for batch_i, batch in enumerate(train_dataloader):

                current_step = batch_i + epoch * len(train_dataloader)
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                with accelerator.accumulate(model):
                    output, gate_respond = model(input_ids)

                    gates_stats.extend(gate_respond.flatten().tolist())
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

                    accelerator.log({"train_loss": loss.item()}, step=current_step + 1)
                    accelerator.log({"train_accuracy": accuracy}, step=current_step + 1)
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
                        with torch.no_grad():
                            for eval_batch in val_dataloader:
                                input_ids, attention_mask, labels = (
                                    eval_batch["input_ids"],
                                    eval_batch["attention_mask"],
                                    eval_batch["labels"],
                                )
                                output, gates_respond = model(input_ids)
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

    accelerator.end_training()

    # TODO: add the processing of gates_respond


if __name__ == "__main__":
    main()
