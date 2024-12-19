from dataclasses import dataclass, field
import marshmallow

@dataclass
class DataParams:
    tokenizer_name: str
    train_data_path: str
    validate_data_path: str
    test_data_path: str
    subreddit1: str
    subreddit2: str
    mask_prob: float = field(
        default=0.10, metadata={"validate": marshmallow.validate.Range(max=1)}
    )

@dataclass
class LoadParams:
    raw_data_path: str
    dataset_url: str
    train_len: int = field(
        default=200000, metadata={"validate": marshmallow.validate.Range(min=1000)}
    )
    valid_len: int
    test_len: int

@dataclass
class DataParams:
    load_params: LoadParams
    data_params: DataParams
    random_state: int
    seed: int