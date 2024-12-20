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

    # process of training

    return


if __name__ == "__main__":
    main()
