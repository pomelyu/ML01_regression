from pathlib import Path
from typing import List, Optional, Union


def mkdir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def rmdir(folder: str):
    folder = Path(folder)
    for child in folder.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rmdir(folder)
    folder.rmdir()


def images_in_folder(folder: str, image_ext: Optional[Union[str, List[str]]] = None):
    if image_ext is None:
        image_ext = [".jpg", ".png", ".tif"]

    return files_in_folder(folder, image_ext)


def files_in_folder(folder: str, file_ext: Union[str, List[str]]) -> List[str]:

    if isinstance(file_ext, str):
        file_ext = [file_ext]

    files = [str(path) for path in Path(folder).iterdir() if path.suffix.lower() in file_ext]
    files = list(sorted(files))
    return files
