import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: str) -> dict:
    with Path(path).open("r") as f:
        info = json.load(f)
    return info


def write_formatted_json(path: str, data: dict) -> None:
    with Path(path).open("w+") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"{path} not found"

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(path: str, image: np.ndarray):
    folder = Path(path).parent
    if not folder.exists():
        raise FileNotFoundError(f"Parent folder not Found: {folder}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def load_video(video_path: str) -> cv2.VideoCapture:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError("Can not open the Video: {}".format(video_path))

    return video


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
