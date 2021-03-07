from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from mlconfig.config import Config
from torch.nn import Module
from tqdm import tqdm

from ..utils.mlflow_logger import MLFlowLogger


class Trainer(metaclass=ABCMeta):
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        self.meta = meta
        self.device = device
        self.exp_logger = exp_logger

        self.cfg_trainer = self.config_trainer(config.trainer)

        self.step = 1

        self.train_dataset = None
        self.valid_dataset = None

        self.optimizers = []
        self.schedulers = []

        self.msg_dict = {}

    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("niters", 100000)
        config.setdefault("niters_print", 100)
        config.setdefault("niters_display", 10000)
        config.setdefault("niters_valid", 10000)
        config.setdefault("niters_save", 20000)
        config.setdefault("niters_save_latest", 5000)
        config.set_immutable(False)
        return config

    def fit(self):
        self.on_training_begin()

        try:
            pbar = tqdm(self.train_dataset, total=len(self.train_dataset), ascii=True)
            pbar.update(self.step)
            for data in pbar:
                data_dict = self.prepare_data(data)

                self.on_iteration_begin(data_dict)

                self.training_step(data_dict)

                self.on_iteration_end(data_dict)

                if self.step % self.cfg_trainer.niters_valid == 0:
                    self.validation_step()

                self.step += 1

                if self.step > self.cfg_trainer.niters:
                    break

        except Exception as e:
            self.on_training_interrupt()
            raise e

        self.on_training_end()

    def on_training_begin(self):
        logger.info("Training begin")

    def on_training_end(self):
        logger.info("Finish training")

    def on_training_interrupt(self):
        logger.info("Training stop")

    def on_iteration_begin(self, data_dict: dict):
        self.msg_dict = {"lr": self.optimizers[0].param_groups[0]["lr"]}

    def on_iteration_end(self, data_dict: dict):
        for scheduler in self.schedulers:
            scheduler.step()

    @abstractmethod
    def prepare_data(self, data: dict) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def training_step(self, data_dict: dict):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, name: Optional[str] = None, training_step=False):
        raise NotImplementedError()

    @abstractmethod
    def resume(self, model_path: str, training_step=False):
        raise NotImplementedError()

    @staticmethod
    def with_state_name(save_path: str) -> str:
        save_path = Path(save_path)
        state_path = save_path.with_name(f"{save_path.stem}_state{save_path.suffix}")
        return state_path

    @staticmethod
    def print_model_size(model: Module, depth: int = 0):
        models = [(model._get_name(), model)]
        for _ in range(depth):
            childrens = []
            for name, m in models:
                childrens += [(name + "." + n, child) for n, child in list(m.named_children())]
            models = childrens

        for name, m in models:
            num_params = 0
            for param in m.parameters():
                num_params += param.numel()
            logger.info("[Network] {} # of parameters: {:.3f} M".format(name, num_params / 1e6))
