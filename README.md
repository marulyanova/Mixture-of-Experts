# Mixture-of-Experts
Проектная работа по курсу Advanced NLP. Transformer encoder с заменой MLP-блока на MOE для задачи MLM


# Запуск эскпериментов
* для получения пояснений к параметрам введите опцию --help

## 1. Запуск формирования датасета
```bash
python data_processors/load_dataset.py --config-name dataset_params.yaml
```

## 2. Запуск процесса маскирования датасетов сформированных на предыдущем шаге
```bash
python data_processors/dataset.py --data-config dataset_params.yaml --model-config model_params.yaml
```

## 2. Запуск обучения
```bash
python train/train.py --config-name model_params.yaml
```