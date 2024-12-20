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
python data_processors/dataset.py --config-dataset dataset_params.yaml --config-model model_params.yaml
```

## 3. Запуск обучения
```bash
python train/train.py --config-model model_params.yaml --config-dataset dataset_params.yaml --config-train train_params.yaml