import sys
import os
sys.path.append("../")
from dataset import load_as_hf_dataset
from config_utils.load_config import load_params_from_yaml, DataParamsSchema, ModelParamsSchema
import torch
import click
from pathlib import Path
from datasets import Dataset
from transformers import BertTokenizerFast

@click.command()
@click.option('--config-name', type=Path, required=True)
def main(config_name):
    loaded_params = load_params_from_yaml("dataset_params.yaml", DataParamsSchema)
    root_dir = Path().resolve().parents[0]
    paths = [ loaded_params.data_params.train_data_path, 
             loaded_params.data_params.test_data_path,
             loaded_params.data_params.subset1_path+loaded_params.data_params.subreddit1+".csv", 
             loaded_params.data_params.subset2_path+loaded_params.data_params.subreddit2+".csv"]
    for i in paths:
        dataset = load_as_hf_dataset(i)
        make_mask(dataset, loaded_params, os.path.splitext(os.path.basename(i))[0])

def make_mask(dataset, loaded_params, file_name):
    model_params = load_params_from_yaml("model_params.yaml", ModelParamsSchema)
    tokenizer = BertTokenizerFast.from_pretrained(loaded_params.data_params.tokenizer_name)
    text = dataset["body"]
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        max_length=model_params.seq_len, 
        truncation=True, 
        padding='max_length')

    inputs['labels'] = inputs.input_ids.detach().clone()

    # create random array of floats in equal dimension to input_ids
    rand = torch.rand(inputs.input_ids.shape)

    # create mask, without CLS, SEP and padding (0)
    mask_arr = (rand < loaded_params.data_params.mask_prob) * \
                (inputs.input_ids != tokenizer.cls_token_id) * \
                (inputs.input_ids != tokenizer.sep_token_id) * \
                (inputs.input_ids != 0)  

    selection = torch.flatten((mask_arr[0]).nonzero()).tolist()

    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten((mask_arr[i]).nonzero()).tolist()
        inputs.input_ids[i, selection] = tokenizer.mask_token_id
    torch.save(inputs, f'{loaded_params.data_params.masked_data_path}/{file_name}.pt')

if __name__ == "__main__":
    main()