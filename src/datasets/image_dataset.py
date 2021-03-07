from typing import List, Union

import cv2
import mlconfig
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ..utils.io import load_image
from ..utils.os import images_in_folder


@mlconfig.register("ImageDataset")
class ImageDataLoader(DataLoader):
    def __init__(self, dataroot: Union[str, List[str]], train: bool, image_size: int, num_iters: int, 
                    pin_memory: bool = True, batch_size: int = 1, **kwargs):

        num_data = num_iters * batch_size
        dataset = ImageDataset(dataroot, train, image_size, num_data)

        super().__init__(dataset=dataset, shuffle=False, drop_last=False, pin_memory=pin_memory, batch_size=batch_size, **kwargs)


class ImageDataset(Dataset):
    def __init__(self, dataroot: Union[str, List[str]], train: bool, image_size: int, num_data: int):
        self.train = train
        self.image_size = image_size
        self.num_data = num_data

        if isinstance(dataroot, str):
            dataroot = [dataroot]

        self.data = np.array([images_in_folder(folder) for folder in dataroot]).ravel()

        if len(self.data) == 0:
            raise RuntimeError("Image not found in the folders")

        logger.info(f"Found {len(self.data)} images in folders")

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        path = np.random.choice(self.data)
        image = load_image(path, grayscale=False)

        H, W = image.shape[:2]
        image_size = self.image_size

        # crop the centeral region and resize to image_size
        m1 = np.eye(3)
        m1[0, 2] = -W/2
        m1[1, 2] = -H/2

        m2 = np.eye(3)
        m2[0, 0] *= (image_size / min(H, W))
        m2[1, 1] *= (image_size / min(H, W))

        m3 = np.eye(3)
        m3[0, 2] = image_size/2
        m3[1, 2] = image_size/2

        m = m3 @ (m2 @ m1)
        image = cv2.warpAffine(image, m[:2], (image_size, image_size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

        image = np.moveaxis(image, -1, 0)
        image = torch.FloatTensor(image / 255.)

        return {"image": image}
