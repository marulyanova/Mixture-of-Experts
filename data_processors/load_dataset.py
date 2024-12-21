import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from datasets import load_dataset
import pandas as pd

from config_utils.load_config import load_params_from_yaml, DataParamsSchema
import click
from pathlib import Path
from loguru import logger
from collections import defaultdict
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--config-name', type=Path, default="dataset_params.yaml", 
              show_default=True, help="Path to the data configuration file.")
def main(config_name):
    loaded_params = load_params_from_yaml(config_name, DataParamsSchema)
    subreddits=loaded_params.data_params.subreddits
    dataset = load_dataset(loaded_params.load_params.dataset_url, split="train", streaming=True)
    subset = []
    subset1 = []
    subset2 = []
    if len(subreddits)==0:
        for i, record in enumerate(dataset):
            if record['subreddit']==loaded_params.data_params.subreddit1 and len(subset1)<loaded_params.load_params.valid_len:
                subset1.append(record)
            elif record['subreddit']==loaded_params.data_params.subreddit2 and len(subset2)<loaded_params.load_params.valid_len:
                subset2.append(record)
            else:
                subset.append(record)
                if len(subset)>= loaded_params.load_params.train_len+loaded_params.load_params.test_len: 
                    break
    else:
        subreddits_set = set(subreddits) 
        target_count = (loaded_params.load_params.train_len+loaded_params.load_params.test_len) // len(subreddits) 
        subreddit_counts = defaultdict(int)
        for record in dataset:
            if record['subreddit']==loaded_params.data_params.subreddit1 and len(subset1)<loaded_params.load_params.valid_len:
                subset1.append(record)
            elif record['subreddit']==loaded_params.data_params.subreddit2 and len(subset2)<loaded_params.load_params.valid_len:
                subset2.append(record)
            else:
                subreddit = record['subreddit']
                if subreddit in subreddits_set:
                    if subreddit_counts[subreddit] < target_count:
                        subset.append(record)
                        subreddit_counts[subreddit] += 1
                        print(len(subset))
            if (len(subset) >= loaded_params.load_params.train_len+loaded_params.load_params.test_len 
                and len(subset2)>=loaded_params.load_params.valid_len 
                and len(subset1)>=loaded_params.load_params.valid_len):
                    break
    labels = [record['subreddit'] for record in subset]
    if len(subreddits)==0:
        train_indices, test_indices = train_test_split(
        range(len(subset)), 
        test_size=loaded_params.load_params.test_len/(loaded_params.load_params.test_len+loaded_params.load_params.train_len), 
        random_state=loaded_params.random_state
    )
    else:
        train_indices, test_indices = train_test_split(
            range(len(subset)), 
            test_size=loaded_params.load_params.test_len/(loaded_params.load_params.test_len+loaded_params.load_params.train_len), 
            stratify=labels, 
            random_state=loaded_params.random_state
        )
    train_set = [subset[i] for i in train_indices]
    test_set = [subset[i] for i in test_indices]

    subset1_df= pd.DataFrame(subset1)
    subset2_df= pd.DataFrame(subset2)
    train_df = pd.DataFrame(train_set)
    test_df = pd.DataFrame(test_set)

    # Сохраняем DataFrame в CSV файлы
    train_df.to_csv(loaded_params.data_params.train_data_path, index=False)
    logger.info(f"Saved train dataset with size: {len(train_df)}")
    test_df.to_csv(loaded_params.data_params.test_data_path, index=False)
    logger.info(f"Saved test dataset with size: {len(test_df)}")
    subset1_df.to_csv(loaded_params.data_params.subset1_path+loaded_params.data_params.subreddit1+".csv", index=False)
    subset2_df.to_csv(loaded_params.data_params.subset2_path+loaded_params.data_params.subreddit2+".csv", index=False)

if __name__ == "__main__":
    main()


            






