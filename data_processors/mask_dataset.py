import torch
from transformers import BertTokenizerFast

from loguru import logger

def make_mask(seq_len, dataset, loaded_params, file_name):
    logger.info(f"Start masking dataset: {file_name}")

    tokenizer = BertTokenizerFast.from_pretrained(loaded_params.data_params.tokenizer_name)
    text = dataset["body"]
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        max_length=seq_len, 
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

    save_path = f'{loaded_params.data_params.masked_data_path}{file_name}.pt'
    torch.save(inputs, save_path)

    logger.info(f"Masking dataset save to: {save_path}")
