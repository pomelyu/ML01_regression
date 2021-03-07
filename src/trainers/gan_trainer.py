from typing import Optional

import mlconfig
import numpy as np
import torch
from kornia.utils import tensor_to_image
from mlconfig.collections import AttrDict
from mlconfig.config import Config
from torch.nn import Module
from tqdm import tqdm

from ..utils.mlflow_logger import MLFlowLogger
from ..utils.pytorch import get_state_dict, set_requires_grad, set_state_dict
from .trainer import Trainer


@mlconfig.register()
class GANTrainer(Trainer):
    def __init__(self, meta: dict, device: torch.device, exp_logger: MLFlowLogger, config: Config, **kwargs):
        super().__init__(meta, device, exp_logger, config)

        self.train_dataset = config.dataset(train=True, num_iters=self.cfg_trainer.niters)

        self.criterion: Module = config.gan_criterion().to(device)

        config_model = self.patch_config_model(config.model)
        self.gen: Module = config_model.gen().to(device)
        self.dis: Module = config_model.dis().to(device)

        self.optimizer_gen = config.optimizer.gen(self.gen.parameters())
        self.optimizer_dis = config.optimizer.dis(self.dis.parameters())

        self.scheduler_gen = config.scheduler(self.optimizer_gen)
        self.scheduler_dis = config.scheduler(self.optimizer_dis)

        self.optimizers += [self.optimizer_gen, self.optimizer_dis]
        self.schedulers += [self.scheduler_gen, self.scheduler_dis]

        self.latent_size = config.model.gen.latent_size
        self.fixed_sample = torch.randn(self.cfg_trainer.num_fixed_sample, self.latent_size, device=device)

        self.print_model_size(self.gen)
        self.print_model_size(self.dis)


    @classmethod
    def config_trainer(cls, config: Config) -> Config:
        config = super().config_trainer(config)
        config.set_immutable(True)
        config.setdefault("num_fixed_sample", 25)
        config.setdefault("num_image_rows", 5)
        config.set_immutable(False)

        assert config.num_fixed_sample % config.num_image_rows == 0

        return config

    @classmethod
    def patch_config_model(cls, config_model):
        config_model.set_immutable(True)
        config_model.dis.update({"in_nc": config_model.gen.out_nc})
        config_model.set_immutable(False)

        return config_model

    def prepare_data(self, data: dict) -> dict:
        data_dict = AttrDict()
        data_dict.image = data["image"].to(self.device)

        B = data_dict.image.shape[0]
        data_dict.z = torch.randn(B, self.latent_size, device=self.device)

        return data_dict

    def training_step(self, data_dict: dict) -> dict:
        # train generator
        loss_dict = self.update_gen(data_dict)
        self.msg_dict.update(loss_dict)

        # train discriminator
        loss_dict = self.update_dis(data_dict)
        self.msg_dict.update(loss_dict)

    def update_gen(self, data_dict: dict) -> dict:
        self.gen.train()
        set_requires_grad(self.gen, True)
        self.dis.eval()
        set_requires_grad(self.dis, False)

        self.optimizer_gen.zero_grad()
        fake = self.gen(data_dict.z)
        loss_g_real, loss_g_fake = self.criterion.loss_g(self.dis, data_dict.image, fake)
        loss_g = loss_g_real + loss_g_fake
        loss_g.backward()
        self.optimizer_gen.step()

        return {"loss_g_real": loss_g_real.item(), "loss_g_fake": loss_g_fake.item()}

    def update_dis(self, data_dict: dict) -> dict:
        self.gen.eval()
        set_requires_grad(self.gen, False)
        self.dis.train()
        set_requires_grad(self.dis, True)

        self.optimizer_dis.zero_grad()
        fake = self.gen(data_dict.z)
        loss_d_real, loss_d_fake = self.criterion.loss_d(self.dis, data_dict.image, fake)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_dis.step()

        return {"loss_d_real": loss_d_real.item(), "loss_d_fake": loss_d_fake.item()}

    def on_iteration_begin(self, data_dict):
        if self.step == 1:
            image = self.inference(data_dict)
            self.exp_logger.log_image(self.step, image, "generated", ext="jpg")

        super().on_iteration_begin(data_dict)

    def on_iteration_end(self, data_dict):
        if self.exp_logger is None:
            return

        if self.step % self.cfg_trainer.niters_print == 0:
            message = self.exp_logger.log_message(self.step, **self.msg_dict)
            tqdm.write(message)

        if self.step % self.cfg_trainer.niters_display == 0:
            image = self.inference(data_dict)
            self.exp_logger.log_image(self.step, image, "generated", ext="jpg")

        if self.step % self.cfg_trainer.niters_save == 0:
            self.save("latest", training_step=True)

        super().on_iteration_end(data_dict)

    @torch.no_grad()
    def inference(self, data_dict: dict) -> np.ndarray:
        self.gen.eval()
        num_rows = self.cfg_trainer.num_image_rows

        fake_image = tensor_to_image(self.gen(data_dict.z[:num_rows]))
        fake_image = np.expand_dims(fake_image, 1)

        fixed_image = tensor_to_image(self.gen(self.fixed_sample))
        fixed_image = fixed_image.reshape((num_rows, -1, *fake_image.shape[-3:]))

        # fake_image = (num_rows, 1, H, W, D)
        # fixed_image = (num_rows, N, H, W, D)
        result = np.concatenate([fake_image, fixed_image], axis=1)  # (num_rows, N', H, W, D)
        result = np.concatenate(result, axis=1) # (N', H * num_rows, W, D)
        result = np.concatenate(result, axis=1) # (H * num_rows, W * N', D)
        result = (np.ascontiguousarray(result) * 255).astype(np.uint8)

        return result

    def validation_step(self):
        raise NotImplementedError()

    def save(self, name: Optional[str] = None, training_step=False):
        raise NotImplementedError()

    def resume(self, model_path: str, training_step=False):
        raise NotImplementedError()
