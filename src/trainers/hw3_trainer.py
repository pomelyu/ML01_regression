from pathlib import Path

import cv2
import kornia as K
import mlconfig
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from loguru import logger
from mlconfig.collections import AttrDict
from mlconfig.config import Config
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from tqdm import tqdm, trange

from src.utils.mlflow_logger import MLFlowLogger
from src.utils.pytorch import get_state_dict, set_state_dict


@mlconfig.register()
class HW3Trainer():
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        self.meta = AttrDict(meta)
        self.device = device
        self.exp_logger = exp_logger

        self.train_state = AttrDict({})
        self.train_state.epoch = 1
        self.train_state.step = 1

        self.cfg_trainer = self.config_trainer(config.trainer)
        self.cfg_dataset = self.config_dataset(config.dataset)
        self.cfg_augmentator = self.config_augmentator(config.augmentator)
        self.cfg_mixmatch = self.config_mixmatch(config.mixmatch)

        batch_size = self.cfg_dataset.batch_size
        num_workers = self.cfg_dataset.num_workers
        batch_u = self.cfg_dataset.batch_size // (self.cfg_mixmatch.K + 1)
        batch_x = self.cfg_dataset.batch_size - batch_u * self.cfg_mixmatch.K
        self.train_dataloader = create_dataloader(create_dataset(self.cfg_dataset.train_dataset), batch_x, num_workers)
        self.valid_dataloader = create_dataloader(create_dataset(self.cfg_dataset.valid_dataset), batch_size, num_workers)
        self.test_dataloader = create_dataloader(create_dataset(self.cfg_dataset.test_dataset), batch_size, num_workers, shuffle=False)
        unlabel_dataloader = create_dataloader(create_dataset(self.cfg_dataset.unlabeled_dataset), batch_u, num_workers, drop_last=True)
        self.unlabel_dataloader = InfiniteIteratorWrapper(unlabel_dataloader)

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

        self.criterion_label = nn.CrossEntropyLoss()
        self.criterion_unlabel = nn.MSELoss()

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
    def config_mixmatch(cls, config: Config) -> Config:
        config.set_immutable(False)
        config.setdefault("T", 0.5)
        config.setdefault("K", 2)
        config.setdefault("alpha", 0.5)
        config.setdefault("lambda_unlabel", 100)
        config.setdefault("step_rampup", 16000)
        config.set_immutable(True)
        return config

    @classmethod
    def config_augmentator(cls, config: Config) -> Config:
        config.set_immutable(False)

        random_affine: AttrDict = config.get("random_affine", AttrDict({}))
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

    def prepare_data(self, data: dict, augment: bool = False) -> dict:
        data_dict = AttrDict()
        data_dict.x = data[0].to(self.device)
        data_dict.y = data[1].to(self.device)

        if augment:
            data_dict.x = self.augmentator(data_dict.x)
        data_dict.x = self.normalized(data_dict.x)

        return data_dict

    def forward(self, data_dict):
        y_pred = self.model(data_dict.x)
        data_dict.y_pred = y_pred
        return data_dict


    @torch.no_grad()
    def on_train_begin(self):
        self.train_state.best_valid = np.finfo(float).max
        self.train_state.best_metric = 0
        self.train_state.early_stop_count = 0
        self.train_state.epoch_pbar = trange(1,  self.cfg_trainer.epochs + 1, ascii=True)
        self.train_state.epoch_pbar.update(self.train_state.epoch)


    @torch.no_grad()
    def on_train_end(self):
        self.model.eval()

        self.resume(self.exp_logger.tmp_dir / "best_loss.pth", training_state=False)
        result_file = self.inference(f"pred_best_loss_{self.meta.exp_id[:5]}")
        self.exp_logger.log_artifact(result_file)

        self.resume(self.exp_logger.tmp_dir / "best_metric.pth", training_state=False)
        result_file = self.inference(f"pred_best_metric_{self.meta.exp_id[:5]}")
        self.exp_logger.log_artifact(result_file)


    @torch.no_grad()
    def on_epoch_begin(self):
        self.train_state.loss_x = 0
        self.train_state.loss_u = 0
        self.train_state.num_data = 0
        self.train_state.mixed_sample = None

        self.train_state.iter_pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader), ascii=True)


    @torch.no_grad()
    def on_epoch_end(self):
        self.save("latest", training_state=True)

        loss_x = self.train_state.loss_x / self.train_state.num_data
        loss_u = self.train_state.loss_u / self.train_state.num_data

        self.exp_logger.log_metric("loss_x", loss_x, self.train_state.epoch)
        self.exp_logger.log_metric("loss_u", loss_u, self.train_state.epoch)
        self.exp_logger.log_image(self.train_state.epoch, self.train_state.mixed_sample, "mixed_sample", "jpg")

        if self.train_state.epoch % self.cfg_trainer.epochs_eval != 0:
            return

        train_loss, train_metric, train_sample = self.evaluate(self.train_dataloader, "train_dataset")
        valid_loss, valid_metric, valid_sample = self.evaluate(self.valid_dataloader, "valid_dataset")

        self.exp_logger.log_metric("train_loss", train_loss, self.train_state.epoch)
        self.exp_logger.log_metric("valid_loss", valid_loss, self.train_state.epoch)
        self.exp_logger.log_metric("train_metric", train_metric, self.train_state.epoch)
        self.exp_logger.log_metric("valid_metric", valid_metric, self.train_state.epoch)
        self.exp_logger.log_image(self.train_state.epoch, train_sample, "train_sample", "jpg")
        self.exp_logger.log_image(self.train_state.epoch, valid_sample, "valid_sample", "jpg")

        if valid_loss >= self.train_state.best_valid and valid_metric <= self.train_state.best_metric:
            self.train_state.early_stop_count += self.cfg_trainer.epochs_eval
        else:
            self.train_state.early_stop_count = 0
            if valid_loss < self.train_state.best_valid:
                self.train_state.best_valid = valid_loss
                self.save("best_loss", training_state=True)

            if valid_metric > self.train_state.best_metric:
                self.train_state.best_metric = valid_metric
                self.save("best_metric", training_state=True)

        tqdm.write(
            f"Epoch {self.train_state.epoch:0>5d}:" + \
            f"lr: {self.optimizer.param_groups[0]['lr']:.5f}, " + \
            f"train_loss: {train_loss:.5f}, " + \
            f"valid_loss: {valid_loss:.5f}, " + \
            f"train_metric: {train_metric:.5f}, " + \
            f"valid_metric: {valid_metric:.5f}"
        )

        self.exp_logger.log_metric("best_valid", self.train_state.best_valid, self.train_state.epoch)
        self.exp_logger.log_metric("best_metric", self.train_state.best_metric, self.train_state.epoch)


    @torch.no_grad()
    def on_iteration_begin(self, data):
        data_dict = self.prepare_data(data, augment=True)
        unlabel_data_dict = self.create_unlabel_data()

        all_x = torch.cat([data_dict.x, unlabel_data_dict.x], dim=0)
        all_y = torch.cat([label_to_onehot(data_dict.y, 11), unlabel_data_dict.y], dim=0)

        L = np.random.beta(self.cfg_mixmatch.alpha, self.cfg_mixmatch.alpha)
        L = max(L, 1-L)

        shuffle_index = torch.randperm(len(all_x))
        result = AttrDict({})
        result.x = all_x * L + all_x[shuffle_index] * (1 - L)
        result.y = all_y * L + all_y[shuffle_index] * (1 - L)

        self.model.train()
        return result


    @torch.no_grad()
    def on_iteration_end(self, data_dict):
        self.scheduler.step()
        self.model.eval()

        if self.train_state.mixed_sample is None:
            self.train_state.mixed_sample = self.show_prediction(data_dict)

        batch_size = len(data_dict.x)
        self.train_state.loss_x += data_dict.loss_x.item() * batch_size
        self.train_state.loss_u += data_dict.loss_u.item() * batch_size
        self.train_state.num_data += batch_size

        loss_x = self.train_state.loss_x / self.train_state.num_data
        loss_u = self.train_state.loss_u / self.train_state.num_data
        self.train_state.iter_pbar.set_description(f"[{self.train_state.epoch:0>4d}] loss_x: {loss_x:.4f} loss_u: {loss_u:.4f}")
        self.train_state.iter_pbar.refresh()

        self.train_state.step += 1

        return data_dict


    def train_step(self, data_dict):
        self.optimizer.zero_grad()
        data_dict = self.forward(data_dict)

        batch_u = self.cfg_dataset.batch_size // (self.cfg_mixmatch.K + 1)
        batch_x = self.cfg_dataset.batch_size - batch_u * self.cfg_mixmatch.K
        # First B are labeled
        loss_x = self.criterion_label(data_dict.y_pred[:batch_x], torch.argmax(data_dict.y[:batch_x], dim=-1))
        loss_u = self.criterion_unlabel(data_dict.y_pred[batch_x:], data_dict.y[batch_x:])

        lambda_unlabel = min(self.train_state.step / self.cfg_mixmatch.step_rampup, 1) * self.cfg_mixmatch.lambda_unlabel
        loss = loss_x + lambda_unlabel * loss_u
        loss.backward()
        self.optimizer.step()

        data_dict.loss_x = loss_x.detach()
        data_dict.loss_u = loss_u.detach()
        data_dict.loss = loss.detach()
        data_dict.y = torch.argmax(data_dict.y, dim=-1)
        return data_dict


    def fit(self):
        self.on_train_begin()

        for self.train_state.epoch in self.train_state.epoch_pbar:
            self.on_epoch_begin()

            for data in self.train_state.iter_pbar:
                data_dict = self.on_iteration_begin(data)
                data_dict = self.train_step(data_dict)
                data_dict = self.on_iteration_end(data_dict)

            self.on_epoch_end()

            if self.train_state.early_stop_count >= self.cfg_trainer.early_stop:
                break

        self.on_train_end()


    @torch.no_grad()
    def create_unlabel_data(self):
        K = self.cfg_mixmatch.K
        x = []
        y = 0
        for _ in range(K):
            data = next(self.unlabel_dataloader)
            data_dict = self.prepare_data(data, augment=True)
            x.append(data_dict.x)
            y += torch.softmax(self.model(data_dict.x), dim=-1)

        # x: (K, B, D, H, W)
        # y: (B, C)
        # sharpen
        y /= K
        y = torch.pow(y, 1 / (self.cfg_mixmatch.T + 1e-5))
        y /= y.sum(-1, keepdim=True)
        y = y.detach()

        result = AttrDict({})
        result.x = torch.cat(x, dim=0)
        result.y = y.repeat(K, 1)

        return result


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
            loss = self.criterion_label(data_dict.y_pred, data_dict.y)
            losses.append(loss.item())

            metric = calculate_accuracy(data_dict.y_pred, data_dict.y)
            matrics.append(metric.item())

            if sample is None:
                sample = self.show_prediction(data_dict)

        return np.mean(losses), np.mean(matrics), sample


    @torch.no_grad()
    def inference(self, name):
        path = Path(f"results/{name}.csv")
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


    @torch.no_grad()
    def save(self, name: str = None, training_state: bool = True):
        self.model.eval()
        checkpoint = {
            "model": get_state_dict(self.model),
            "epoch": self.train_state.epoch,
            "step": self.train_state.step,
            "meta": self.meta,
        }

        self.exp_logger.log_checkpoint(checkpoint, f"{name}.pth")

        if not training_state:
            return

        state_checkpoint = {"optimizer": get_state_dict(self.optimizer)}
        self.exp_logger.log_checkpoint(state_checkpoint, f"{name}_optimizer.pth")


    @torch.no_grad()
    def resume(self, model_path: str, training_state: bool = False):
        model_path = Path(model_path)

        checkpoint = torch.load(model_path)
        set_state_dict(self.model, checkpoint["model"])

        next_epoch = checkpoint["epoch"] + 1
        next_step = checkpoint["step"] + 1

        del checkpoint
        torch.cuda.empty_cache()
        logger.info(f"Load model from: {model_path}")
        if not training_state:
            return

        self.train_state.epoch = next_epoch
        self.train_state.step = next_step

        optimizer_path = model_path.with_name(f"{model_path.stem}_optimizer{model_path.suffix}")
        state_checkpoint = torch.load(optimizer_path)
        set_state_dict(self.optimizer, state_checkpoint["optimizer"])

        del state_checkpoint
        torch.cuda.empty_cache()
        logger.info(f"Load training state from: {optimizer_path}")


def create_dataset(dataroot):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return DatasetFolder(dataroot, loader=lambda x: Image.open(x), extensions="jpg", transform=transform)


def create_dataloader(dataset, batch_size, num_workers, shuffle=True, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, **kwargs)


def calculate_accuracy(pred, label_target):
    label_pred = torch.argmax(pred, dim=-1)
    acc = (label_pred == label_target).float().mean()
    return acc


@mlconfig.register()
def TorchVisionModels(model_name):
    func = getattr(tv_models, model_name)
    return func(pretrained=False, num_classes=11)


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


def label_to_onehot(label: torch.LongTensor, num_class: int) -> torch.FloatTensor:
    if len(label.shape) == 1:
        label = label.unsqueeze(-1)
    result = torch.zeros(len(label), num_class, device=label.device, dtype=torch.float32)
    result.scatter_(1, label, 1)
    return result


class InfiniteIteratorWrapper():
    def __init__(self, iterator):
        self.iterator = iterator
        self.the_iter = iter(iterator)

    def __next__(self):
        try:
            item = next(self.the_iter)
        except StopIteration:
            self.the_iter = iter(self.iterator)
            item = next(self.the_iter)
        return item
            