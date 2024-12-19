from datasets import Dataset
from transformers import BertTokenizerFast
import pandas as pd

# Задаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Загружаем данные
data_path = "dataset/data/raw/data.csv"
df = pd.read_csv(data_path)

# Преобразуем в формат Hugging Face Dataset
dataset = Dataset.from_pandas(df[["title", "body", "subreddit"]])