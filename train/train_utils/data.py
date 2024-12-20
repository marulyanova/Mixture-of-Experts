from typing import Dict, Any, Tuple
import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader


def split_val_data(loaded, p: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_idx = int(len(loaded["input_ids"]) * p)
    return {
        "input_ids": loaded["input_ids"][:train_idx],
        "attention_mask": loaded["attention_mask"][:train_idx],
        "labels": loaded["labels"][:train_idx],
    }, {
        "input_ids": loaded["input_ids"][train_idx:],
        "attention_mask": loaded["attention_mask"][train_idx:],
        "labels": loaded["labels"][train_idx:],
    }


def get_data_dict(loaded) -> Dict[str, torch.tensor]:
    return Dataset.from_dict(
        {
            "input_ids": torch.tensor(loaded["input_ids"]),
            "attention_mask": torch.tensor(loaded["attention_mask"]),
            "labels": torch.tensor(loaded["labels"]),
        }
    )


def PrepareDataset(train_loaded, test_loaded, p: float) -> DatasetDict:
    train_loaded, val_loaded = split_val_data(train_loaded, p)
    dataset = DatasetDict(
        {
            "train": get_data_dict(train_loaded),
            "val": get_data_dict(val_loaded),
            "test": get_data_dict(test_loaded),
        }
    )
    dataset.set_format(type="torch")
    return dataset


def PrepareDataloader(
    dataset: DatasetDict, train_params: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        dataset["train"], batch_size=train_params["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        dataset["val"], batch_size=train_params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=train_params["batch_size"], shuffle=False
    )
    return train_dataloader, val_dataloader, test_dataloader
