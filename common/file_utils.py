import os
import numpy as np

def make_if_not_exists(filepath: str) -> None:
    d = os.path.dirname(filepath)
    if not os.path.exists(d):
        os.makedirs(d)


def save_ndarray(ndarray: np.ndarray, filepath: str) -> None:
    make_if_not_exists(filepath)
    np.save(filepath, ndarray)


def load_ndarray(filepath: str) -> np.ndarray:
    ndarray = np.load(filepath)
    return ndarray


def filter_by_image_extensions(image_name: str):
    return image_name.lower().endswith(("jpg", "jpeg", "png"))

def list_files_relative_pathes(dir_path: str, recursive=False) -> list:
    rel_pathes = []
    for dir_, dirs, files in os.walk(dir_path):
        for f in files:
            if dir_==dir_path:
                rel_file=f
                rel_pathes.append(rel_file)
            elif recursive:
                rel_dir = os.path.relpath(dir_, dir_path)
                rel_file = os.path.join(rel_dir, f)
                rel_pathes.append(rel_file)
    return rel_pathes


