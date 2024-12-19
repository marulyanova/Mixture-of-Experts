from datasets import Dataset
from transformers import BertTokenizerFast

from config_utils.dataset_params import DataParams
def mask_dataset(dataset: Dataset, data_params: DataParams):

    tokenizer = BertTokenizerFast.from_pretrained(data_params.data_params.tokenizer_name)
    return