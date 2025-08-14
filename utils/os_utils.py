import os
from sys import platform


def is_windows():
    return platform == "win32"


def is_linux():
    return platform == "linux"


def create_dir_if_not_exists(dir_fp: str) -> None:
    if os.path.exists(dir_fp):
        return
    else:
        os.mkdir(dir_fp)
