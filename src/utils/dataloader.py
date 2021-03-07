from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .io import load_image, load_video
from .os import images_in_folder


class VideoLoader():
    def __init__(self, video_path: str, begin_frame: int = 0, skip_frame: int = 1):
        assert begin_frame >= 0

        self.video_name = Path(video_path).stem
        self.video = load_video(video_path)
        self.begin_frame_index = begin_frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame_index)
        self.skip_frame = skip_frame

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        for _ in range(self.skip_frame - 1):
            self.video.read()

        ret, frame = self.video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame, f"{self.video_name}_{index+1:0>5d}"

        raise IndexError("Index out of range {}".format(index))

    def __len__(self) -> int:
        num_frame = (self.video.get(cv2.CAP_PROP_FRAME_COUNT) - self.begin_frame_index) // self.skip_frame
        return int(num_frame)


class ImageFolderLoader():

    def __init__(self, image_folder: str, image_ext: Optional[Union[str, List[str]]] = None):
        self.files = images_in_folder(image_folder, image_ext)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        path = self.files[index]
        image = load_image(path)

        return image, path.stem


    def __len__(self) -> int:
        return len(self.files)
