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


@click.command()
@click.option('--config-name', type=Path, required=True)
def main(config_name):
    data_params = load_params_from_yaml(config_name, DataParamsSchema)

    df = pd.read_csv(data_params.dataset_params.data_path)

    dataset = Dataset.from_pandas(df[["title", "body", "subreddit"]])
    print(dataset[0])
    
    tokenizer = BertTokenizerFast.from_pretrained(data_params.dataset_params.tokenizer_name)
    return 

if __name__=="__main__":
    main()