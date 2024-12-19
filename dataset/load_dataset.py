from datasets import load_dataset
import pandas as pd

from loguru import logger

def load_dataset(params_path: str):
    dataset = load_dataset(params.load_params.dataset_url, split="train", streaming=True)
    subset = []
    for i, record in enumerate(dataset):
        subset.append(record)
        if i >= params.load_params.data_len: 
            break

    df = pd.DataFrame(subset)
    logger.info(f"Loaded dataset with size {len(df)}")
    df.to_csv(params.load_params.raw_data_path, index=False)
    logger.info(f"Saved dataset to path {params.load_params.raw_data_path}")


