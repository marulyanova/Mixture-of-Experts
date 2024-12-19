import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import click
from pathlib import Path

from config_utils.load_config import load_params_from_yaml, ModelParamsSchema


@click.command()
@click.option('--config-name', type=Path, required=True)
def main(config_name):
    model_params = load_params_from_yaml(config_name, ModelParamsSchema)
    print(model_params)

    return 


if __name__ == "__main__":
    main()