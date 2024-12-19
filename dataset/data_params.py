import sys
from dataclasses import dataclass, field

import marshmallow.validate
import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class DataParams:
    train_data_path: str
    test_data_path: str
    subreddit1: str
    subreddit2: str
    mask_prob: float = field(
        default=0.10, metadata={"validate": marshmallow.validate.Range(max=1)}
    )

@dataclass()
class LoadParams:
    raw_data_path: str
    dataset_url: str
    data_len: int = field(
        default=200000, metadata={"validate": marshmallow.validate.Range(min=1000)}
    )


@dataclass()
class PipelineParams:
    data_params: DataParams
    load_params: LoadParams
    random_state: int
    seed: int


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(path: str) -> PipelineParams:
    with open(path, "r") as input_stream:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))



