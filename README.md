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
```

### 4. Запуск инференса
```bash
python inference/inference.py --config-model model_params.yaml --config-dataset dataset_params.yaml --config-train train_params.yaml
```
или с параметрами по умолчанию 
```bash
python inference/inference.py
```

## Aim для отслеживания экспериментов

Установить, если не установлен
```bash
pip install aim
```

Поднять UI
```bash
aim up
```

Запуск обучения с логированием эксперимента в Aim
```bash
python train/train.py --tag name
```