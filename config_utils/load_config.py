import typing as t
from pathlib import Path

import yaml

from marshmallow import Schema
from marshmallow_dataclass import class_schema

from config_utils.model_params import ModelParams
from config_utils.dataset_params import DatasetParams

project_root = Path(__file__).resolve().parents[1]
config_root = project_root / "configs"

ModelParamsSchema = class_schema(ModelParams)
DatasetParamsSchema = class_schema(DatasetParams)

# main load yaml function
def load_params_from_yaml( 
    path: Path, schema: t.Type[Schema]
) -> t.Union[
    ModelParams, DatasetParams
]:
    full_path = config_root / path
    with open(full_path, "r") as f:
        schema = schema()
        return schema.load(yaml.safe_load(f))