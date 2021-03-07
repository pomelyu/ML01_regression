import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import mlflow
import numpy as np
import torch

from .os import rmdir


class MLFlowLogger():
    def __init__(
        self,
        exp_name: str = "defualt",
        tracking_uri: Optional[str] = None,
        run_name: str = "default",
        run_id: Optional[str] = None,
        tmp_dir: Optional[str] = None,
    ):

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_id=run_id, run_name=run_name)

        self.tracking_uri = mlflow.get_tracking_uri()
        self.experiment = mlflow.get_experiment_by_name(exp_name)
        self.exp_id = self.experiment.experiment_id
        self.run_id = mlflow.active_run().info.run_id
        self.log_param("_run_id", self.run_id)

        if tmp_dir:
            self.tmp_dir = Path(tmp_dir / "mlflow")
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.tmp_dir = Path(tempfile.mkdtemp("mlflow"))

    def close(self) -> None:
        rmdir(self.tmp_dir)
        mlflow.end_run()

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        mlflow.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_param(self, key: str, value: float):
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, float]):
        mlflow.log_params(params)

    def log_message(self, step: int, **kwargs) -> str:
        message = f"Step: {step:0>8d}, "
        for k, v in kwargs.items():
            message += f"{k}: {v:.4f}, "
            self.log_metric(k, v, step)

        return message

    def log_image(self, step: int, image: np.ndarray, prefix: str = "res", ext: str = "png"):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        res_path = str(self.tmp_dir / f"{prefix}_{step:0>8d}.{ext}")
        cv2.imwrite(res_path, image)
        self.log_artifact(res_path, prefix)

        res_path = str(self.tmp_dir / f"{prefix}_latest.{ext}")
        cv2.imwrite(res_path, image)
        self.log_artifact(res_path)

    def log_checkpoint(self, checkpoint: dict, name: str = "model", artifact_path: Optional[str] = None):
        model_path = self.tmp_dir / name
        torch.save(checkpoint, model_path)
        self.log_artifact(model_path, artifact_path)

    @torch.no_grad()
    def log_grad_statistics(self, model: torch.nn.Module, prefix: str, step: str):
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0

        for p in model.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        grad_means = grad_means / grad_count

        self.log_metric(f"{prefix}.max", grad_max, step)
        self.log_metric(f"{prefix}.means", grad_means, step)
