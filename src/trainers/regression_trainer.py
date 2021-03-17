from pathlib import Path

import mlconfig
import numpy as np
import torch
from mlconfig.collections import AttrDict
from mlconfig.config import Config
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from src.utils.mlflow_logger import MLFlowLogger


@mlconfig.register()
class RegressionTrainer():
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        self.meta = AttrDict(meta)
        self.device = device
        self.exp_logger = exp_logger

        self.epoch = 1

        self.cfg_trainer = self.config_trainer(config.trainer)
        self.cfg_dataset = self.config_dataset(config.dataset)

        train_dataset = CSVDataset("train", self.cfg_dataset.train_dataset, self.cfg_dataset.select_column)
        self.train_dataset = DataLoader(train_dataset, batch_size=self.cfg_dataset.batch_size, shuffle=True)
        self.meta.update(**train_dataset.meta)

        valid_dataset = CSVDataset("valid", self.cfg_dataset.train_dataset, self.cfg_dataset.select_column)
        self.valid_dataset = DataLoader(valid_dataset, batch_size=self.cfg_dataset.batch_size, shuffle=False)

        test_dataset = CSVDataset("test", self.cfg_dataset.test_dataset, self.cfg_dataset.select_column)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.cfg_dataset.batch_size, shuffle=False)

        self.criterion = nn.MSELoss().to(self.device)

        self.model = MLP(**config.model).to(self.device)
        self.optimizer = config.optimizer(self.model.parameters())
        self.scheduler = config.scheduler(self.optimizer)

        self.gt_mean = self.meta.gt_mean.to(self.device)
        self.gt_std = self.meta.gt_std.to(self.device)

    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("epochs", 500)
        config.setdefault("early_stop", 200)
        config.set_immutable(False)
        return config

    @classmethod
    def config_dataset(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("batch_size", 500)
        config.setdefault("select_column", None)
        config.set_immutable(False)
        return config

    def prepare_data(self, data: dict) -> dict:
        data_dict = AttrDict()
        data_dict.x_state = data["state"].to(self.device)
        data_dict.x_feat = ((data["feat"] - self.meta.feat_mean) / self.meta.feat_std).to(self.device)
        data_dict.y = (data["gt"].to(self.device) - self.gt_mean) / self.gt_std
        return data_dict

    def fit(self):
        pbar = trange(self.cfg_trainer.epochs, ascii=True)
        pbar.update(self.epoch)

        best_valid = np.finfo(float).max
        early_stop_count = 0

        for self.epoch in pbar:
            for data in self.train_dataset:
                data_dict = self.prepare_data(data)

                self.optimizer.zero_grad()
                y_pred = self.model(torch.cat([data_dict.x_state, data_dict.x_feat], dim=-1))
                loss = self.criterion(data_dict.y, y_pred)
                loss.backward()
                self.optimizer.step()

            train_loss, train_metric = self.evaluate(self.train_dataset)
            valid_loss, valid_metric = self.evaluate(self.valid_dataset)

            self.scheduler.step()

            self.exp_logger.log_metric("train_loss", train_loss, self.epoch)
            self.exp_logger.log_metric("valid_loss", valid_loss, self.epoch)

            self.exp_logger.log_metric("train_metric", train_metric, self.epoch)
            self.exp_logger.log_metric("valid_metric", valid_metric, self.epoch)

            if valid_loss < best_valid:
                tqdm.write(
                    f"Epoch {self.epoch:0>5d}:" + \
                    f"lr: {self.optimizer.param_groups[0]['lr']:.5f}, " + \
                    f"train_loss: {train_loss:.5f}, " + \
                    f"valid_loss: {valid_loss:.5f}, " + \
                    f"train_metric: {train_metric:.5f}, " + \
                    f"valid_metric: {valid_metric:.5f}"
                )
                best_valid = valid_loss
                self.exp_logger.log_checkpoint(self.model.state_dict(), "best.pth")
                early_stop_count = 0
            else:
                early_stop_count += 1

            self.exp_logger.log_metric("best_valid", best_valid, self.epoch)

            if early_stop_count >= self.cfg_trainer.early_stop:
                break

        self.model.load_state_dict(torch.load(self.exp_logger.tmp_dir / "best.pth"))
        result_file = self.inference()
        self.exp_logger.log_artifact(result_file)


    @torch.no_grad()
    def evaluate(self, dataset):
        losses = []
        matrics = []
        for data in dataset:
            data_dict = self.prepare_data(data)
            y_pred = self.model(torch.cat([data_dict.x_state, data_dict.x_feat], dim=-1))
            loss = self.criterion(data_dict.y, y_pred)
            losses.append(loss.item())

            metric = self.criterion(data_dict.y * self.gt_std + self.gt_mean, y_pred * self.gt_std + self.gt_mean)
            matrics.append(metric.item())

        return np.mean(losses), np.mean(matrics)

    @torch.no_grad()
    def inference(self):
        path = Path("results/pred.csv")
        f = path.open("w+")
        f.write("id,tested_positive")

        count = 0
        for data in self.test_dataset:
            data_dict = self.prepare_data(data)
            y_pred = self.model(torch.cat([data_dict.x_state, data_dict.x_feat], dim=-1))
            y_pred = y_pred.cpu() * self.gt_std + self.gt_mean

            for pred in y_pred:
                f.write(f"\n{count},{pred.item():.5f}")
                count += 1

        f.close()
        return path


class MLP(nn.Sequential):
    def __init__(self, in_nc, out_nc, nd, n_layers=2):
        super().__init__()

        self.add_module("fc1", nn.Linear(in_nc, nd))
        self.add_module("actv1", nn.ReLU())

        for i in range(2, n_layers):
            self.add_module(f"fc{i}", nn.Linear(nd, nd))
            self.add_module(f"actv{i}", nn.ReLU())

        self.add_module("out", nn.Linear(nd, out_nc))


class CSVDataset(Dataset):

    columns = [
        "cli", "ili" , "hh_cmnty_cli", "nohh_cmnty_cli", "wearing_mask",
        "travel_outside_state", "work_outside_home", "shop", "restaurant",
        "spent_time", "large_event", "public_transit", "anxious", "depressed",
        "felt_isolated", "worried_become_ill", "worried_finances", "tested_positive"
    ]

    def __init__(self, split, csv_file, use_columns=None):
        assert split in ["train", "valid", "test"]

        raw = np.loadtxt(csv_file, delimiter=",", skiprows=1)

        if split == "train":
            selected_index = [i for i in range(len(raw)) if i % 10 != 0]
        elif split == "valid":
            selected_index = [i for i in range(len(raw)) if i % 10 == 0]
        else:
            selected_index = range(len(raw))

        raw = raw[selected_index]

        # 2nd ~ 41th columns indicate state
        self.state_code = raw[:, 1:41].astype(np.float)

        if split == "test":
            self.gts = np.zeros_like(raw[:, -1:], dtype=np.float)
        else:
            self.gts = raw[:, -1:].astype(np.float)

        if use_columns is None:
            self.feats = raw[:, 41:-1].astype(np.float)
        else:
            offset = len(self.columns)
            index = []
            for col in use_columns:
                i = self.columns.index(col)
                if col == "tested_positive":
                    index += [41+i, 41+i+offset]
                else:
                    index += [41+i, 41+i+offset, 41+i+offset*2]

            self.feats = raw[:, index]

        self.meta = AttrDict()
        self.meta.feat_mean = torch.FloatTensor(self.feats.mean(0, keepdims=True))
        self.meta.feat_std = torch.FloatTensor(self.feats.std(0, keepdims=True))
        self.meta.gt_mean = torch.FloatTensor(self.gts.mean(0, keepdims=True))
        self.meta.gt_std = torch.FloatTensor(self.gts.std(0, keepdims=True))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        return {
            "state": torch.FloatTensor(self.state_code[index]),
            "feat": torch.FloatTensor(self.feats[index]),
            "gt": torch.FloatTensor(self.gts[index]),
        }
