import os
import sys

import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import json
from pathlib import Path
from typing import Dict
import random

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
from transformers import AdamW
from loguru import logger

from config_utils.load_config import (
    DataParamsSchema,
    ModelParamsSchema,
    TrainParamsSchema,
    load_params_from_yaml,
)
from model.model_main import MoETransformerEncoder
from train_utils.data import PrepareDataloader, PrepareDataset
from train_utils.gate import (
    process_gate_response,
    calculate_loss_accuracy,
    epoch_gates_cat,
)


@click.command()
@click.option(
    "--config-model",
    type=Path,
    default="model_params.yaml",
    show_default=True,
    help="Path to the model configuration file.",
)
@click.option(
    "--config-dataset",
    type=Path,
    default="dataset_params.yaml",
    show_default=True,
    help="Path to the data configuration file.",
)
@click.option(
    "--config-train",
    type=Path,
    default="train_params.yaml",
    show_default=True,
    help="Path to the train configuration file.",
)
@click.option("--tag", type=str, required=True, help="One tag to mark experiment")
def main(config_model, config_dataset, config_train, tag):

    # LOAD PARAMS

    model_params = load_params_from_yaml(config_model, ModelParamsSchema)
    loaded_params = load_params_from_yaml(config_dataset, DataParamsSchema)
    train_params = load_params_from_yaml(config_train, TrainParamsSchema)

    os.makedirs(train_params.save_path, exist_ok=True)

    # ВОСПРОИЗВОДИМОСТЬ ЭКСПЕРИМЕНТОВ

    set_seed(train_params.random_seed)
    torch.cuda.manual_seed(train_params.random_seed)
    np.random.seed(train_params.random_seed)
    torch.manual_seed(train_params.random_seed)
    random.seed(train_params.random_seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(train_params.random_seed)
    torch.cuda.manual_seed_all(train_params.random_seed)
    torch.backends.cudnn.benchmark = False

    # DATASET
    subreddit1_loaded = torch.load(
        loaded_params.data_params.masked_data_path
        + loaded_params.data_params.subreddit1
        + ".pt"
    )
    subreddit2_loaded = torch.load(
        loaded_params.data_params.masked_data_path
        + loaded_params.data_params.subreddit2
        + ".pt"
    )
    train_loaded = torch.load(
        loaded_params.data_params.masked_data_path
        + os.path.splitext(os.path.basename(loaded_params.data_params.train_data_path))[
            0
        ]
        + ".pt"
    )
    val_loaded = torch.load(
        loaded_params.data_params.masked_data_path
        + os.path.splitext(os.path.basename(loaded_params.data_params.test_data_path))[
            0
        ]
        + ".pt"
    )

    dataset = PrepareDataset(
        train_loaded,
        val_loaded,
    )

    train_dataloader, val_dataloader = PrepareDataloader(dataset, train_params)

    # ACCELERATOR

    accelerator = Accelerator(
        gradient_accumulation_steps=train_params.gradient_accumulation_steps,
        project_dir=".",
        log_with="aim",
    )
    accelerator.init_trackers(
        train_params.experiment_name, config=json.loads(train_params.model_dump_json())
    )
    accelerator.get_tracker("aim").writer.add_tag(tag)

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

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
    )

    # PROCESS OF TRAINING

    # [n_epochs, n_layers, batch_size, max_len]
    train_gates_stats = torch.tensor([]).to(model_params.device)

    # [хз что, n_layers, batch_size, max_len]
    val_gates_stats = torch.tensor([]).to(model_params.device)

    with tqdm(desc="Training", total=total_steps, dynamic_ncols=True) as pbar:
        for epoch in range(train_params.n_epochs):

            # [n_layers, batch_size, max_len]
            epoch_gates_stats = torch.tensor([]).to(model_params.device)

            for batch_i, batch in enumerate(train_dataloader):

                current_step = batch_i + epoch * len(train_dataloader)
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                with accelerator.accumulate(model):
                    output, gate_respond = model(input_ids)

                    # print("GATE_RESPOND_SHAPE", gate_respond.shape)

                    # extend and reshape to nessesary form [n_layers, batch_size, max_len]
                    epoch_gates_stats = process_gate_response(
                        epoch_gates_stats,
                        gate_respond,
                        train_params,
                        model_params,
                    )

                    # print("EPOCH_GATE_RESPOND_SHAPE", epoch_gates_stats.shape)

                    loss, accuracy = calculate_loss_accuracy(
                        input_ids, output, labels, train_params
                    )

                    accelerator.log(
                        {"train_batch_loss": loss.item()}, step=current_step + 1
                    )
                    accelerator.log(
                        {"train_batch_accuracy": accuracy}, step=current_step + 1
                    )
                    logger.info(f"train_batch_loss: {loss.item()}")
                    logger.info(f"train_batch_accuracy: {accuracy}")
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if (
                    current_step + 1
                ) % train_params.eval_steps == 0 or current_step + 1 == total_steps:
                    model.eval()
                    with tqdm(
                        desc="Eval", total=len(val_dataloader), dynamic_ncols=True
                    ) as eval_pbar:
                        epoch_gates_stats_val = torch.tensor([]).to(model_params.device)
                        with torch.no_grad():
                            for eval_batch in val_dataloader:
                                input_ids, attention_mask, labels = (
                                    eval_batch["input_ids"],
                                    eval_batch["attention_mask"],
                                    eval_batch["labels"],
                                )
                                output, gate_respond_val = model(input_ids)

                                epoch_gates_stats_val = process_gate_response(
                                    epoch_gates_stats_val,
                                    gate_respond_val,
                                    train_params,
                                    model_params,
                                )

                                # print(
                                #     "EPOCH_GATE_RESPOND_SHAPE VAL",
                                #     epoch_gates_stats_val.shape,
                                # )

                                loss, accuracy = calculate_loss_accuracy(
                                    input_ids, output, labels, train_params
                                )

                                eval_pbar.update(1)
                                accelerator.log(
                                    {"accuracy_batch_eval": accuracy},
                                    step=current_step + 1,
                                )

                                accelerator.log(
                                    {"loss_batch_eval": loss}, step=current_step + 1
                                )
                                logger.info(f"accuracy_batch_eval: {accuracy}")
                                logger.info(f"loss_batch_eval: {loss}")

                    model.train()

                if (
                    current_step + 1
                ) % train_params.save_steps == 0 or current_step + 1 == total_steps:
                    accelerator.wait_for_everyone()
                    accelerator.save_model(
                        model,  Path(train_params.save_path) / f"step_{current_step + 1}"
                    )

            train_gates_stats = epoch_gates_cat(train_gates_stats, epoch_gates_stats)
            val_gates_stats = epoch_gates_cat(val_gates_stats, epoch_gates_stats_val)

            # print(train_gates_stats.shape)
            # print(val_gates_stats.shape)
 
            torch.save(
                train_gates_stats, 
                Path(train_params.save_path) / train_params.train_gatestats_filename
            )
            torch.save(
                val_gates_stats, 
                Path(train_params.save_path) / train_params.val_gatestats_filename
            )

    accelerator.end_training()

    torch.save(
        model.state_dict(), 
        Path(train_params.save_path) / train_params.model_filename
    )


if __name__ == "__main__":
    main()
