from pathlib import Path

import mlconfig
import numpy as np
import torch
from mlconfig.collections import AttrDict
from mlconfig.config import Config
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from tqdm import tqdm, trange

from src.utils.mlflow_logger import MLFlowLogger


@mlconfig.register()
class HW3Trainer():
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        self.meta = AttrDict(meta)
        self.device = device
        self.exp_logger = exp_logger

        self.epoch = 1

        self.cfg_trainer = self.config_trainer(config.trainer)
        self.cfg_dataset = self.config_dataset(config.dataset)

        batch_size = self.cfg_dataset.batch_size
        self.train_dataset = create_dataset(self.cfg_dataset.train_dataset)
        self.unlabel_dataset = create_dataset(self.cfg_dataset.unlabeled_dataset)
        self.valid_dataloader = create_dataloader(create_dataset(self.cfg_dataset.valid_dataset), batch_size)
        self.test_dataloader = create_dataloader(create_dataset(self.cfg_dataset.test_dataset), batch_size)

        self.criterion = nn.CrossEntropyLoss()

        self.model = config.model().to(self.device)
        self.optimizer = config.optimizer(self.model.parameters())
        self.scheduler = config.scheduler(self.optimizer)

    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("epochs", 500)
        config.setdefault("epochs_eval", 10)
        config.setdefault("early_stop", np.iinfo(int).max)
        config.set_immutable(False)
        return config

    @classmethod
    def config_dataset(cls, config: Config) -> Config:
        config.set_immutable(True)
        config.setdefault("batch_size", 128)
        config.set_immutable(False)
        return config

    def prepare_data(self, data: dict) -> dict:
        data_dict = AttrDict()
        data_dict.x = data["image"].to(self.device)
        data_dict.y = data["label"].to(self.device)
        data_dict.index = data["index"]
        return data_dict

    def forward(self, data_dict):
        y_pred = self.model(data_dict.x)
        data_dict.y_pred = y_pred
        return data_dict

    def fit(self):
        pbar = trange(1,  self.cfg_trainer.epochs + 1, ascii=True)
        pbar.update(self.epoch)

        best_valid = np.finfo(float).max
        best_metric = 0
        early_stop_count = 0

        for self.epoch in pbar:
            train_dataloader = create_dataloader(self.train_dataset, batch_size=self.cfg_dataset.batch_size)

            for data in tqdm(train_dataloader, total=len(train_dataloader), ascii=True, desc="train"):
                data_dict = self.prepare_data(data)

                self.optimizer.zero_grad()
                data_dict = self.forward(data_dict)
                loss = self.criterion(data_dict.y_pred, data_dict.y)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            if self.epoch % self.cfg_trainer.epochs_eval != 0:
                continue

            train_loss, train_metric = self.evaluate(train_dataloader)
            valid_loss, valid_metric = self.evaluate(self.valid_dataloader)

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
                best_metric = valid_metric
                self.exp_logger.log_checkpoint(self.model.state_dict(), "best.pth")
                early_stop_count = 0
            else:
                early_stop_count += 1

            self.exp_logger.log_metric("best_valid", best_valid, self.epoch)
            self.exp_logger.log_metric("best_metric", best_metric, self.epoch)

            if early_stop_count >= self.cfg_trainer.early_stop:
                break

        self.model.load_state_dict(torch.load(self.exp_logger.tmp_dir / "best.pth"))
        result_file = self.inference()
        self.exp_logger.log_artifact(result_file)

    @torch.no_grad()
    def evaluate(self, dataloader):
        losses = []
        matrics = []
        for data in tqdm(dataloader, total=len(dataloader), ascii=True, desc="evaluate"):
            data_dict = self.prepare_data(data)
            data_dict = self.forward(data_dict)
            loss = self.criterion(data_dict.y_pred, data_dict.y)
            losses.append(loss.item())

            metric = calculate_accuracy(data_dict.y_pred, data_dict.y)
            matrics.append(metric.item())

        return np.mean(losses), np.mean(matrics)


    @torch.no_grad()
    def inference(self):
        path = Path("results/pred.csv")
        f = path.open("w+")
        f.write("Id,Category")

        count = 0
        for data in tqdm(self.test_dataloader, total=len(self.test_dataloader), ascii=True, desc="inference"):
            data_dict = self.prepare_data(data)
            data_dict = self.forward(data_dict)

            for pred in torch.argmax(data_dict.y_pred, dim=-1):
                f.write(f"\n{count},{pred.item():d}")
                count += 1

        f.close()
        return path

def create_dataset(dataroot):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=255),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return DatasetFolderWithIndex(dataroot, loader=lambda x: Image.open(x), extensions="jpg", transform=transform)


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def calculate_accuracy(pred, label_target):
    label_pred = torch.argmax(pred, dim=-1)
    acc = (label_pred == label_target).float().mean()
    return acc


class DatasetFolderWithIndex(DatasetFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return {"image": image, "label": label, "index": index}


@mlconfig.register()
class HW3MiniClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),

            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 8 * 8, 11),
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


@mlconfig.register()
class HW3BasicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x
