from pathlib import Path

import cv2
import kornia as K
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
        self.cfg_augmentator = self.config_augmentator(config.augmentator)

        batch_size = self.cfg_dataset.batch_size
        num_workers = self.cfg_dataset.num_workers
        self.train_dataset = create_dataset(self.cfg_dataset.train_dataset)
        self.unlabel_dataset = create_dataset(self.cfg_dataset.unlabeled_dataset)
        self.valid_dataloader = create_dataloader(create_dataset(self.cfg_dataset.valid_dataset), batch_size, num_workers)
        self.test_dataloader = create_dataloader(create_dataset(self.cfg_dataset.test_dataset), batch_size, num_workers, shuffle=False)

        self.normalized = nn.Sequential(
            K.augmentation.Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]), std=torch.FloatTensor([0.229, 0.224, 0.225])),
        ).to(self.device)

        self.denormalized = nn.Sequential(
            K.augmentation.Denormalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]), std=torch.FloatTensor([0.229, 0.224, 0.225])),
        ).to(self.device)

        self.augmentator = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomAffine(**self.cfg_augmentator.random_affine),
            K.augmentation.ColorJitter(**self.cfg_augmentator.color_jitter),
            K.augmentation.GaussianBlur(**self.cfg_augmentator.gaussian_blur),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.model = config.model().to(self.device)
        self.optimizer = config.optimizer(self.model.parameters())
        self.scheduler = config.scheduler(self.optimizer)

    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config.set_immutable(False)
        config.setdefault("epochs", 500)
        config.setdefault("epochs_eval", 10)
        config.setdefault("early_stop", np.iinfo(int).max)
        config.set_immutable(True)
        return config

    @classmethod
    def config_dataset(cls, config: Config) -> Config:
        config.set_immutable(False)
        config.setdefault("batch_size", 128)
        config.setdefault("num_worker", 0)
        config.set_immutable(True)
        return config

    @classmethod
    def config_augmentator(cls, config: Config) -> Config:
        config.set_immutable(False)

        random_affine: AttrDict = config.get("rando_affine", AttrDict({}))
        random_affine.setdefault("p", 0.7)
        random_affine.setdefault("degrees", 10)
        random_affine.setdefault("translate", (0.05, 0.05))
        random_affine.setdefault("scale", (0.9, 1.1))
        random_affine.setdefault("shear", 5)
        config.random_affine = random_affine

        color_jitter: AttrDict = config.get("color_jitter", AttrDict({}))
        color_jitter.setdefault("p", 1.0)
        color_jitter.setdefault("brightness", 0)
        color_jitter.setdefault("contrast", 0)
        color_jitter.setdefault("saturation", 0)
        color_jitter.setdefault("hue", 0)
        config.color_jitter = color_jitter

        gaussian_blur: AttrDict = config.get("gaussian_blur", AttrDict({}))
        gaussian_blur.setdefault("p", 1.0)
        gaussian_blur.setdefault("kernel_size", (5, 5))
        gaussian_blur.setdefault("sigma", (3, 3))
        config.gaussian_blur = gaussian_blur

        config.set_immutable(True)
        return config

    def prepare_data(self, data: dict, augment: bool =False) -> dict:
        data_dict = AttrDict()
        data_dict.x = data["image"].to(self.device)
        data_dict.y = data["label"].to(self.device)
        data_dict.index = data["index"]

        if augment:
            data_dict.x = self.augmentator(data_dict.x)
        data_dict.x = self.normalized(data_dict.x)

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
            train_dataloader = create_dataloader(self.train_dataset, self.cfg_dataset.batch_size, self.cfg_dataset.num_workers)

            train_loss = 0
            train_metric = 0
            train_sample = None
            num_data = 0
            tbar = tqdm(train_dataloader, total=len(train_dataloader), ascii=True)
            for data in tbar:
                data_dict = self.prepare_data(data, augment=True)

                self.optimizer.zero_grad()
                data_dict = self.forward(data_dict)
                loss = self.criterion(data_dict.y_pred, data_dict.y)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    if train_sample is None:
                        train_sample = self.show_prediction(data_dict)

                    batch = len(data_dict.x)
                    train_loss+= loss.item() * batch
                    train_metric += calculate_accuracy(data_dict.y_pred, data_dict.y).item() * batch
                    num_data += batch

                    tbar.set_description(f"[{self.epoch:0>4d}] loss: {train_loss / num_data:.4f}")
                    tbar.refresh()

            self.scheduler.step()

            train_loss /= num_data
            train_metric /= num_data
            self.exp_logger.log_metric("train_loss", train_loss, self.epoch)
            self.exp_logger.log_metric("train_metric", train_metric, self.epoch)
            self.exp_logger.log_image(self.epoch, train_sample, "train_sample", "jpg")

            if self.epoch % self.cfg_trainer.epochs_eval != 0:
                continue

            valid_loss, valid_metric, valid_sample = self.evaluate(self.valid_dataloader, "valid_dataset")

            self.exp_logger.log_metric("valid_loss", valid_loss, self.epoch)
            self.exp_logger.log_metric("valid_metric", valid_metric, self.epoch)
            self.exp_logger.log_image(self.epoch, valid_sample, "valid_sample", "jpg")

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
                early_stop_count += self.cfg_trainer.epochs_eval

            self.exp_logger.log_metric("best_valid", best_valid, self.epoch)
            self.exp_logger.log_metric("best_metric", best_metric, self.epoch)

            if early_stop_count >= self.cfg_trainer.early_stop:
                break

        self.model.load_state_dict(torch.load(self.exp_logger.tmp_dir / "best.pth"))
        result_file = self.inference()
        self.exp_logger.log_artifact(result_file)

    @torch.no_grad()
    def show_prediction(self, data_dict, num_data=16):
        num_data = min(num_data, len(data_dict.x))
        x = self.denormalized(data_dict.x[:num_data])
        x = K.resize(x, (128, 128)) * 255
        x = torch.clamp(x, 0, 255)
        x: np.ndarray = K.tensor_to_image(x).astype(np.uint8)
        x = np.ascontiguousarray(x)
        y_pred = torch.argmax(data_dict.y_pred[:num_data], dim=-1).detach().cpu().numpy()
        y = data_dict.y[:num_data].cpu().numpy()

        for img, pred, gt in zip(x, y_pred, y):
            draw_text(img, f"{gt:d}:{pred:d}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        x = x.reshape((4, -1, *x.shape[1:]))
        x = np.concatenate(x, axis=-2)
        x = np.concatenate(x, axis=-3)

        return x

    @torch.no_grad()
    def evaluate(self, dataloader, name):
        losses = []
        matrics = []
        sample = None
        for data in tqdm(dataloader, total=len(dataloader), ascii=True, desc=f"evaluate {name}"):
            data_dict = self.prepare_data(data)
            data_dict = self.forward(data_dict)
            loss = self.criterion(data_dict.y_pred, data_dict.y)
            losses.append(loss.item())

            metric = calculate_accuracy(data_dict.y_pred, data_dict.y)
            matrics.append(metric.item())

            if sample is None:
                sample = self.show_prediction(data_dict)

        return np.mean(losses), np.mean(matrics), sample


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
    ])
    return DatasetFolderWithIndex(dataroot, loader=lambda x: Image.open(x), extensions="jpg", transform=transform)


def create_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)


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


def draw_text(image, text,
        pos=(0, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        font_thickness=2,
        text_color=(255, 255, 255),
        text_color_bg=(0, 0, 0, 0)
    ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y - text_h), (x + text_w, y), text_color_bg, -1)
    cv2.putText(image, text, pos, font, font_scale, text_color, font_thickness, bottomLeftOrigin=False)
