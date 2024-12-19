from datasets import load_dataset
import pandas as pd

from config_utils.dataset_params import LoadParams

from loguru import logger

def load_dataset(load_params: LoadParams):
    dataset = load_dataset(load_params.dataset_url, split="train", streaming=True)
    subset = []
    for i, record in enumerate(dataset):
        subset.append(record)
        if i >= load_params.train_len: 
            break

    df = pd.DataFrame(subset)
    logger.info(f"Loaded dataset with size {len(df)}")
    df.to_csv(load_params.raw_data_path, index=False)
    logger.info(f"Saved dataset to path {load_params.raw_data_path}")


