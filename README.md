# Mixture-of-Experts
Проектная работа по курсу Advanced NLP. Transformer encoder с заменой MLP-блока на MOE для задачи MLM


# Запуск эскпериментов

## 1. Запуск формирования датасета
```bash
python3 dataset/dataset.py  --config-name train_dataset_params.yaml
```

## 2. Запуск обучения
```bash
python3 train/train.py  --config-name model_params.yaml
```