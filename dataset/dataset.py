import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import click
from pathlib import Path
from datasets import Dataset
from transformers import BertTokenizerFast
import pandas as pd

from config_utils.load_config import load_params_from_yaml, DataParamsSchema

from dataset.load_dataset import load_dataset

def load_as_hf_dataset(datapath: Path) -> Dataset:
    #TODO if file does not exist load with load_dataset func
    df = pd.read_csv(datapath)
    dataset = Dataset.from_pandas(df[["title", "body", "subreddit"]])
    return dataset


@click.command()
@click.option('--config-name', type=Path, required=True)
def main(config_name):
    dataset_params = load_params_from_yaml(config_name, DataParamsSchema)
    dataset = load_as_hf_dataset(dataset_params.data_params.train_data_path)
    print(dataset[0])
    
    return 

if __name__=="__main__":
    main()