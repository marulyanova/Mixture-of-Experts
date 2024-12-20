from typing import Dict, Any, Tuple
import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader


def get_data_dict(loaded) -> Dict[str, torch.tensor]:
    return Dataset.from_dict(
        {
            "input_ids": torch.tensor(loaded["input_ids"]),
            "attention_mask": torch.tensor(loaded["attention_mask"]),
            "labels": torch.tensor(loaded["labels"]),
        }
    )


def PrepareDataset(train_loaded, val_loaded) -> DatasetDict:
    dataset = DatasetDict(
        {
            "train": get_data_dict(train_loaded),
            "val": get_data_dict(val_loaded),
        }
    )
    dataset.set_format(type="torch")
    return dataset


def PrepareDataloader(
    dataset: DatasetDict, train_params: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        dataset["train"], batch_size=train_params.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset["val"], batch_size=train_params.batch_size, shuffle=False
    )
    return train_dataloader, val_dataloader
