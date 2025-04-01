import os.path
import numpy as onp
import pandas as pd
import wandb
from time import time

from typing import Dict, List

Scalar = float | int


class Logger:
    aggregated_data: Dict[str, pd.DataFrame]
    write_csv: bool = False
    epoch_start_time: float| None = None
    # backend: str = 'wandb'  # TODO: add support for Tensorboard?

    def __init__(self, project: str, config: Dict, dir: str, name=None) -> None:
        dir = os.path.join(dir, 'wandb')
        wandb.init(project=project, config=config, dir=dir, name=name)  # type: ignore

    def log(self, dict: Dict[str, Scalar]) -> None:
        wandb.log(dict)
        if self.accumulate:
            for key, value in dict.items():
                if key in self.accumulate_keys:
                    self.accumulate_keys[key].append(value)

    def start_mean(self, keys: List[str]) -> None:
        self.accumulate_keys = {key: [] for key in keys}
        self.accumulate = True

    def _evaluate_mean(self, key=None) -> None:
        for key, values in self.accumulate_keys.items():
            label = f'mean/{key}'
            self.log({label: sum(values) / len(values)})

    def get_current_mean(self, key: str, max_nans=5) -> float:
        values = self.accumulate_keys[key]
        array = onp.array(values)
        isnan = onp.isnan(array)
        out = array[~isnan].mean()
        if isnan.sum() > max_nans:
            out = onp.nan
        return out

    def stop_mean(self) -> None:
        self._evaluate_mean()
        self.accumulate = False
        self.accumulate_keys = {}

    def start_epoch(self, e: int) -> None:
        print('#' * 20, f'Epoch {e}:', flush=True)
        current_time = time()
        if self.epoch_start_time is not None:
            time_diff = current_time - self.epoch_start_time
            self.log({'debug/Epoch run duration [min]:': time_diff / 60})
        self.epoch_start_time = current_time

    def log_epoch_training_duration(self) -> None:
        current_time = time()
        time_diff = current_time - self.epoch_start_time  # type: ignore
        self.log({'debug/Epoch train duration [min]:': time_diff / 60})