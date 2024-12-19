from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Путь для сохранения модели и токенизатора
save_directory = "dataset/model"

# Сохранение модели и токенизатора
tokenizer.save_pretrained(save_directory)