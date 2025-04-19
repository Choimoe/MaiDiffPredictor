import json
from dataclasses import dataclass, field

config_path = './config/train.json'

@dataclass
class Config:
    use_augmentation: bool = field(default=False)
    diff_dim: int = 6
    batch_size: int = 32
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    lr: float = 1e-4
    epochs: int = 50
    max_len: int = 1280

config = ''

with open(config_path, "r", encoding="utf-8") as config_file:
    config_dict = json.load(config_file)
    config = Config(**config_dict)