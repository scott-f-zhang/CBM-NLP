from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    mode: str = "standard"
    model_name: str = "bert-base-uncased"
    max_len: int = 128
    batch_size: int = 8
    num_epochs: int = 1
    optimizer_lr: float = 1e-5
    data_type: str = "pure_cebab"  # default for standard; joint/mix may override


def make_run_config(
    mode: Optional[str] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    data_type: Optional[str] = None,
    optimizer_lr: Optional[float] = None,
    default_data_type: Optional[str] = None,
) -> RunConfig:
    cfg = RunConfig()
    if default_data_type is not None:
        cfg.data_type = default_data_type
    if mode is not None:
        cfg.mode = mode
    if max_len is not None:
        cfg.max_len = max_len
    if batch_size is not None:
        cfg.batch_size = batch_size
    if model_name is not None:
        cfg.model_name = model_name
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    if data_type is not None:
        cfg.data_type = data_type
    if optimizer_lr is not None:
        cfg.optimizer_lr = optimizer_lr
    return cfg
