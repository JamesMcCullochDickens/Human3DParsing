import os
import shutil
import platform
from pathlib import Path
from typing import Optional, List


def delete_all_files_in_folder(dir_path: str) -> None:
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_if_not_exists(dir_path: str, with_delete: bool = False) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        if with_delete:
            delete_all_files_in_folder(dir_path)


def get_parent(path: str) -> str:
    path = Path(path)
    parent = path.parent.absolute()
    return str(parent)


def get_grandparent(path: str) -> str:
    return get_parent(get_parent(path))


def print_os_info() -> None:
    print(platform.platform())
    print(platform.system())
    print(platform.release())
    print()


def join_inner_paths(outer_dir: str, inner_fps: Optional[List[str]] = None) -> List[str]:
    if not inner_fps:
        inner_fps = os.listdir(outer_dir)
    return [os.path.join(outer_dir, inner_fp) for inner_fp in inner_fps]


def get_first_elem_in_fold(outer_dir: str) -> str:
    inner_fps = os.listdir(outer_dir)
    return os.path.join(outer_dir, inner_fps[0])


def get_file_name(fp: str) -> str:
    fn = os.path.basename(fp)
    if "." in fn:
        fn = fn.split(".")[0]
    return fn
