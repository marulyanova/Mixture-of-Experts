import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import random
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import torch
from accelerate.utils import set_seed, tqdm
from torch.utils.data import DataLoader

from config_utils.load_config import (
    DataParamsSchema,
    ModelParamsSchema,
    TrainParamsSchema,
    load_params_from_yaml,
)
from config_utils.model_params import ModelParams
from config_utils.train_params import TrainParams
from model.model_main import MoETransformerEncoder
from train.train_utils.data import PrepareDataloader, PrepareDataset
from train.train_utils.gate import calculate_loss_accuracy, process_gate_response


def eval_loop(
    dataloader: DataLoader,
    train_params: TrainParams,
    model_params: ModelParams,
    model: MoETransformerEncoder,
) -> Tuple[List[float], List[float], torch.tensor]:

    LOSS, ACCURACY = [], []

    with tqdm(desc="Eval", total=len(dataloader), dynamic_ncols=True) as eval_pbar:
        epoch_gates_stats_val = torch.tensor([]).to(model_params.device)
        with torch.no_grad():
            for eval_batch in dataloader:
                input_ids, attention_mask, labels = (
                    eval_batch["input_ids"].to(model_params.device),
                    eval_batch["attention_mask"].to(model_params.device),
                    eval_batch["labels"].to(model_params.device),
                )
                output, gate_respond_val = model(input_ids)

                epoch_gates_stats_val = process_gate_response(
                    epoch_gates_stats_val,
                    gate_respond_val,
                    train_params,
                    model_params,
                )

                loss, accuracy = calculate_loss_accuracy(
                    input_ids, output, labels, train_params
                )
                LOSS.append(loss.item())
                ACCURACY.append(accuracy)

                eval_pbar.update(1)

    return LOSS, ACCURACY, epoch_gates_stats_val


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
def main(config_model, config_dataset, config_train):

    # LOAD PARAMS

    model_params = load_params_from_yaml(config_model, ModelParamsSchema)
    loaded_params = load_params_from_yaml(config_dataset, DataParamsSchema)
    train_params = load_params_from_yaml(config_train, TrainParamsSchema)

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

    dataset_1, dataset_2 = PrepareDataset(
        subreddit1_loaded, subreddit1_loaded
    ), PrepareDataset(subreddit2_loaded, subreddit2_loaded)

    # programming
    _, loader_1 = PrepareDataloader(dataset_1, train_params)
    # gaming
    _, loader_2 = PrepareDataloader(dataset_2, train_params)

    # MODEL

    model = MoETransformerEncoder(**model_params.__dict__)
    weights = torch.load(
        train_params.save_path + "/" + train_params.model_filename,
        map_location=torch.device(model_params.device),
    )
    model.load_state_dict(weights)
    model = model.to(torch.device(model_params.device))

    # EVAL PROCESS

    model.eval()

    # programming
    loss_1, accuracy_1, epoch_gates_stats_val_1 = eval_loop(
        loader_1, train_params, model_params, model
    )

    # gaming
    loss_2, accuracy_2, epoch_gates_stats_val_2 = eval_loop(
        loader_2, train_params, model_params, model
    )

    print(epoch_gates_stats_val_1.shape, len(loss_1), len(accuracy_1))
    print(epoch_gates_stats_val_2.shape, len(loss_2), len(accuracy_2))

    print("SUCCESS!")

    # TODO: do smth with results


if __name__ == "__main__":
    main()
