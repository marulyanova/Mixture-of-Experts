import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import click
from pathlib import Path
from datasets import Dataset
from transformers import BertTokenizerFast
import pandas as pd

from loguru import logger

from config_utils.load_config import load_params_from_yaml, DataParamsSchema, ModelParamsSchema

from data_processors.mask_dataset import make_mask

def load_as_hf_dataset(datapath: Path) -> Dataset:
    try:
        df = pd.read_csv(datapath)
        dataset = Dataset.from_pandas(df[["title", "body", "subreddit"]])
    except:
        logger.error(f"Cannot load file: {datapath}")
        exit()
    return dataset


@click.command()
@click.option('--config-dataset', type=Path, default="dataset_params.yaml", 
              show_default=True, help="Path to the data configuration file.")
@click.option('--config-model', type=Path, default="model_params.yaml", 
              show_default=True, help="Path to the model configuration file.")
def main(config_dataset,  config_model):
    dataset_params = load_params_from_yaml(config_dataset, DataParamsSchema)
    model_params = load_params_from_yaml(config_model, ModelParamsSchema)
    
    paths = [ dataset_params.data_params.train_data_path, 
             dataset_params.data_params.test_data_path,
             dataset_params.data_params.subset1_path+dataset_params.data_params.subreddit1+".csv", 
             dataset_params.data_params.subset2_path+dataset_params.data_params.subreddit2+".csv"]
    for i in paths:
        dataset = load_as_hf_dataset(i)
        make_mask(model_params.seq_len, dataset, dataset_params, os.path.splitext(os.path.basename(i))[0])

if __name__=="__main__":
    main()