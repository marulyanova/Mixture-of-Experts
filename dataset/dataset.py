import collections
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from datasets import Dataset

# Задаем токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Загружаем данные
data_path = "dataset/data/raw/data.csv"
df = pd.read_csv(data_path)

# Преобразуем в формат Hugging Face Dataset
dataset = Dataset.from_pandas(df[["title", "body", "subreddit"]])

# Коллатор для Whole Word Masking
wwm_probability = 0.2  # Вероятность маскировки слова

def whole_word_masking_data_collator(features):
    # features — это список словарей, так как мы используем batched=True
    for feature in features:
        # Проверяем тип, чтобы убедиться, что feature — это словарь
        if isinstance(feature, dict):
            word_ids = feature.pop("word_ids", None)  # Извлекаем word_ids

            if word_ids is None:
                continue  # Если word_ids нет, пропускаем этот пример

            # Создаем маппинг между словами и их индексами в токенах
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Маскируем случайные слова
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels
        else:
            print("Ошибка: элемент features не является словарем", feature)
    return features

# Токенизация с учетом Whole Word Masking
def tokenize_function(examples):
    encoding = tokenizer(
        examples["body"],  # Токенизируем только поле "body"
        truncation=True,
        padding="max_length",  # Для одинаковой длины
        max_length=128,  # Можно задать максимальную длину
        return_tensors="pt"
    )

    # Создаем word_ids вручную, используя токенизатор
    word_ids = []
    for text in examples["body"]:
        word_ids.append(tokenizer.encode(text, add_special_tokens=False))
    
    encoding["word_ids"] = word_ids
    return encoding

# Применяем токенизацию
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Проверяем, что данные в правильном формате
print(tokenized_dataset[0])

# Применяем коллатор для маскировки целых слов
tokenized_dataset = tokenized_dataset.map(whole_word_masking_data_collator, batched=True)

# Проверка, как выглядит токенизованный датасет
print(tokenized_dataset)
