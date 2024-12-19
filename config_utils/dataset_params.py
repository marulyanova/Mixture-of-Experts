from dataclasses import dataclass

@dataclass
class DatasetParams:
    n_samples: int
    shuffle: bool
    test_data_size: float