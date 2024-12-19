from datasets import load_dataset
import pandas as pd
from data_params import PipelineParams, read_pipeline_params
import typer
from loguru import logger

app = typer.Typer()
# Потоковая загрузка датасета

@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)
    dataset = load_dataset(params.load_params.dataset_url, split="train", streaming=True)
    subset = []
    for i, record in enumerate(dataset):
        subset.append(record)
        if i >= params.load_params.data_len: 
            break

    df = pd.DataFrame(subset)
    logger.info(f"Loaded dataset with size {len(df)}")
    df.to_csv(params.load_params.raw_data_path, index=False)
    logger.info(f"Saved dataset to path {params.load_params.raw_data_path}")


if __name__ == "__main__":
    app()