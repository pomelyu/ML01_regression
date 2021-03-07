import mlconfig
import numpy as np
import torch
from kornia.utils import tensor_to_image

import context  # pylint: disable=unused-import
import src.datasets.image_dataset  # pylint: disable=unused-import
from src.utils.io import save_image
from src.utils.os import mkdir


def test_image_dataset():
    save_folder = mkdir("results/test_image_dataset")

    NUM_ITERS = 5
    BATCH_SIZE = 4
    IMAGE_SIZE = 64

    config_dataset = mlconfig.config.Config({
        "name": "ImageDataset",
        "dataroot": "assets/celeba",
        "train": True,
        "image_size": IMAGE_SIZE,
        "num_iters": NUM_ITERS,
        "batch_size": BATCH_SIZE,
    })

    dataset = config_dataset()

    assert len(dataset) == NUM_ITERS

    for i, batch in enumerate(dataset):
        image = batch["image"]

        assert image.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert image.min() >= 0 and image.max() <= 1
        assert isinstance(image, torch.FloatTensor)

        image = (tensor_to_image(image) * 255).astype(np.uint8)
        image = np.concatenate(image, 1)

        save_image(save_folder / f"{i:0>4d}.jpg", image)
