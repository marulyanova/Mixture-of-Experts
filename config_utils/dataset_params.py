from dataclasses import dataclass, field
import marshmallow

@dataclass
class DatasetParams:
    n_samples: int
    shuffle: bool
    test_data_size: float
    tokenizer_name: str
    data_path: str

@dataclass
class LoadParams:
    raw_data_path: str
    dataset_url: str
    data_len: int = field(
        default=200000, metadata={"validate": marshmallow.validate.Range(min=1000)}
    )

@dataclass
class DataParams:
    load_params: LoadParams
    dataset_params: DatasetParams