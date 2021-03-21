from pathlib import Path

import mlconfig
import numpy as np
import torch
from loguru import logger
from mlconfig.collections import AttrDict
from mlconfig.config import Config
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from src.utils.mlflow_logger import MLFlowLogger


@mlconfig.register()
class ClassificationTrainer():
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        self.meta = AttrDict(meta)
        self.device = device
        self.exp_logger = exp_logger

        self.epoch = 1

        self.cfg_trainer = self.config_trainer(config.trainer)
        self.cfg_dataset = self.config_dataset(config.dataset)

        self.train_dataset = DataLoader(
            TIMITDataset(self.cfg_dataset.dataroot, "train", self.cfg_dataset.valid_ratio),
            batch_size=self.cfg_dataset.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg_dataset.num_workers,
        )

        self.valid_dataset = DataLoader(
            TIMITDataset(self.cfg_dataset.dataroot, "valid", self.cfg_dataset.valid_ratio),
            batch_size=self.cfg_dataset.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg_dataset.num_workers,
        )

        self.test_dataset = DataLoader(
            TIMITDataset(self.cfg_dataset.dataroot, "test"),
            batch_size=self.cfg_dataset.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg_dataset.num_workers,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.model = Classifier(**config.model).to(self.device)
        self.optimizer = config.optimizer(self.model.parameters())
        self.scheduler = config.scheduler(self.optimizer)

    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("epochs", 500)
        config.setdefault("early_stop", 200)
        config.set_immutable(False)
        return config

    @classmethod
    def config_dataset(cls, config: Config) -> Config:
        assert "dataroot" in config

        config.set_immutable(True)
        config.setdefault("batch_size", 64)
        config.setdefault("valid_ratio", 0.2)
        config.setdefault("num_workers", 0)
        config.set_immutable(False)
        return config

    def prepare_data(self, data: dict) -> dict:
        data_dict = AttrDict()
        data_dict.x = data["data"].to(self.device)
        data_dict.y = data["label"].to(self.device)

        return data_dict

    def fit(self):
        pbar = trange(self.cfg_trainer.epochs, ascii=True)
        pbar.update(self.epoch)

        best_metric = 0
        early_stop_count = 0

        for self.epoch in pbar:
            self.model.train()

            train_loss = 0
            train_metric = 0
            train_num = 0

            for data in tqdm(self.train_dataset, total=len(self.train_dataset), ascii=True):
                data_dict = self.prepare_data(data)

                B = len(data_dict.x)

                self.optimizer.zero_grad()
                y_logit = self.model(data_dict.x)
                loss = self.criterion(y_logit, data_dict.y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * B
                train_metric += self.calculate_metric(data_dict.y, y_logit) * B
                train_num += B

            train_loss /= train_num
            train_metric /= train_num

            valid_loss, valid_metric = self.evaluate(self.valid_dataset)

            self.scheduler.step()

            self.exp_logger.log_metric("train_loss", train_loss, self.epoch)
            self.exp_logger.log_metric("valid_loss", valid_loss, self.epoch)

            self.exp_logger.log_metric("train_metric", train_metric, self.epoch)
            self.exp_logger.log_metric("valid_metric", valid_metric, self.epoch)

            if valid_metric > best_metric:
                best_metric = valid_metric
                self.exp_logger.log_checkpoint(self.model.state_dict(), "best.pth")
                early_stop_count = 0
            else:
                early_stop_count += 1

            tqdm.write(
                f"Epoch {self.epoch:0>5d}:" + \
                f"lr: {self.optimizer.param_groups[0]['lr']:.5f}, " + \
                f"best_metric: {best_metric:.5f}, " + \
                f"train_loss: {train_loss:.5f}, " + \
                f"valid_loss: {valid_loss:.5f}, " + \
                f"train_metric: {train_metric:.5f}, " + \
                f"valid_metric: {valid_metric:.5f}"
            )

            self.exp_logger.log_metric("best_metric", best_metric, self.epoch)

            if early_stop_count >= self.cfg_trainer.early_stop:
                break

        self.model.load_state_dict(torch.load(self.exp_logger.tmp_dir / "best.pth"))
        result_file = self.inference()
        self.exp_logger.log_artifact(result_file)

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()

        total_loss = 0
        total_metric = 0
        total_num = 0
        for data in tqdm(dataset, total=len(dataset), ascii=True):
            data_dict = self.prepare_data(data)

            B = len(data_dict.x)

            y_logit = self.model(data_dict.x)
            loss = self.criterion(y_logit, data_dict.y)
            metric = self.calculate_metric(data_dict.y, y_logit)

            total_loss += loss.item() * B
            total_metric += metric * B
            total_num += B

        total_loss /= total_num
        total_metric /= total_num

        return total_loss, total_metric

    @torch.no_grad()
    def inference(self):
        self.model.eval()

        path = Path("results/pred.csv")
        f = path.open("w+")
        f.write("Id,Class")

        count = 0
        for data in tqdm(self.test_dataset, total=len(self.test_dataset), ascii=True):
            data_dict = self.prepare_data(data)
            y_logit = self.model(data_dict.x)
            _, y_pred = torch.max(y_logit, dim=-1)

            for pred in y_pred:
                f.write(f"\n{count},{pred.item():d}")
                count += 1

        f.close()
        return path


    @staticmethod
    def calculate_metric(y: torch.Tensor, y_logit: torch.Tensor) -> float:
        with torch.no_grad():
            _, y_pred = torch.max(y_logit, dim=-1)
            accuracy = (y == y_pred).sum() / len(y)
        return accuracy.item()


class TIMITDataset(Dataset):
    def __init__(self, dataroot, split, valid_ratio=None):
        """ Folder structure
        dataroot/
            test_11.npy
            train_11.npy
            train_label_11.npy
        """

        assert split in ["train", "valid", "test"]

        dataroot = Path(dataroot)
        if split == "test":
            data = torch.FloatTensor(np.load(dataroot / "test_11.npy"))
            label = torch.zeros(len(data), 1)
        else:
            data = torch.FloatTensor(np.load(dataroot / "train_11.npy"))
            label = torch.LongTensor(np.load(dataroot / "train_label_11.npy").astype(np.int))

            assert len(data) == len(label)

            percent = int(len(data) * (1-valid_ratio))
            if split == "train":
                data, label = data[:percent], label[:percent]
            else:
                data, label = data[percent:], label[percent:]

        self.data = data
        self.label = label

        logger.info(f"[{split}] Load: # {len(data)} data")

    def __getitem__(self, index):
        return {
            "data": self.data[index],
            "label": self.label[index],
        }

    def __len__(self):
        return len(self.data)


class Classifier(nn.Module):
    def __init__(self, in_nc, out_nc, nd_qk=64, nd_v=64, n_frames=11, nd_mlp=128, n_mlp_layers=1):
        super().__init__()
        self.attention1 = AttentionBlock(in_nc, nd_v, nd_qk, bias=True)
        self.attention2 = AttentionBlock(nd_v, nd_v, nd_qk, bias=True)

        assert n_mlp_layers >= 1

        operations = []
        for i in range(n_mlp_layers):
            nd = nd_v * n_frames if i == 0 else nd_mlp
            operations += [
                nn.Linear(nd, nd_mlp),
                nn.BatchNorm1d(nd_mlp),
                nn.ReLU(),
            ]
        self.mlp = nn.Sequential(*operations)
        self.out = nn.Linear(nd_mlp, out_nc)

    def forward(self, x):
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.mlp(x)
        x = self.out(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_nc, out_nc, nd, bias=False):
        super().__init__()

        self.in_nc = in_nc
        self.Wq = nn.Linear(in_nc, nd, bias=bias)
        self.Wk = nn.Linear(in_nc, nd, bias=bias)
        self.Wv = nn.Linear(in_nc, out_nc, bias=bias)

    def forward(self, x):
        # x: (B, 11*39) -> (B, 11, 39) = (B, N, C)
        B, N = x.shape
        N = N // self.in_nc
        x = x.view(B*N, self.in_nc)

        Q: torch.Tensor = self.Wq(x).view(B, N, -1) # (B, N, nd)
        K: torch.Tensor = self.Wk(x).view(B, N, -1) # (B, N, nd)
        V: torch.Tensor = self.Wv(x).view(B, N, -1) # (B, N, out_nc)

        A = Q @ K.transpose(1, 2) # (B, N, N)
        A = torch.softmax(A, dim=-1)

        out = A @ V # (B, N, out_nc)
        out = out.view(B, -1)

        return out
