import typing as t
from pathlib import Path

import yaml

from marshmallow import Schema
from marshmallow_dataclass import class_schema

from config_utils.model_params import ModelParams
from config_utils.dataset_params import DataParams

project_root = Path(__file__).resolve().parents[1]
config_root = project_root / "configs"

ModelParamsSchema = class_schema(ModelParams)
DataParamsSchema = class_schema(DataParams)

# main load yaml function
def load_params_from_yaml( 
    path: Path, schema: t.Type[Schema]
) -> t.Union[
    ModelParams, DataParams
]:
    full_path = config_root / path
    with open(full_path, "r") as f:
        schema = schema()
        return schema.load(yaml.safe_load(f))