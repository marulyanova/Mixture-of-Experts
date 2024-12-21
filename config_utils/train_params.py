from dataclasses import dataclass, asdict
import json


@dataclass
class TrainParams:
    batch_size: int
    n_epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    experiment_name: str
    weight_decay: float
    warmup_proportion: float
    tokenizer_mask_id: int  # TODO move to data params
    eval_steps: int
    save_steps: int
    save_path: str
    random_seed: int
    model_filename: str
    train_gatestats_filename: str
    val_gatestats_filename: str

    def model_dump_json(self) -> str:
        """Сериализует объект в JSON строку."""
        return json.dumps(asdict(self), indent=4)
