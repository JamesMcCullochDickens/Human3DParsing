import os.path
import os
import zipfile
import rarfile # pip install rarfile
import shutil
from typing import List, Tuple, Union

import utils.path_utils as path_utils


def tup_to_csv_string(tup):
    s = ""
    for i, val in enumerate(tup):
        s += str(val)
        if i != len(tup)-1:
            s += ","
    return s


def write_lines_to_file(fp: str, open_type: str, lines: List[Union[str, Tuple]]):
    with open(fp, open_type) as f:
        for i, line in enumerate(lines):
            if isinstance(line, Tuple):
                s = tup_to_csv_string(line)
            else:
                s = line
            if i != (len(lines)) -1:
                if s[-1:] != "\n":
                    s += "\n"
            f.write(s)


def append_lines_to_file(fp: str, lines: List[Union[str, Tuple]]):
    write_lines_to_file(fp, "a", lines)


def delete_small_files_in_fold(fold_fp: str, min_file_size_in_kb: int):
    fps = path_utils.join_inner_paths(fold_fp)
    for fp in fps:
        size = os.path.getsize(fp)
        file_size_bytes = size/1024
        if file_size_bytes < min_file_size_in_kb:
            if os.path.isdir(fp):
                shutil.rmtree(fp)
            else:
                os.remove(fp)


def delete_empty_folders(outer_fp: str):
    empty_folders = []
    for root, dirs, files in os.walk(outer_fp):
        if not dirs and not files:
            empty_folders.append(root)
    for empty_fold in empty_folders:
        print(f"Deleting {empty_fold}.")
        shutil.rmtree(empty_fold)


def delete_folder_contents(folder_fp: str):
    inner_fps = path_utils.join_inner_paths(folder_fp)
    for inner_fp in inner_fps:
        if os.path.isdir(inner_fp):
            shutil.rmtree(inner_fp)
        else:
            os.remove(inner_fp)


def extract_files(folder: str):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        # If the item is a directory, recurse into it
        if os.path.isdir(item_path):
            extract_files(item_path)

        # If the item is a zip file, unzip it
        elif item.endswith('.zip'):
            print(f"Unzipping {item_path}....")
            with zipfile.ZipFile(item_path, 'r') as zip_ref:
                zip_ref.extractall(folder)
            print(f'Finished unzipped: {item_path}.')

        # If the item is a rar file, extract it
        elif item.endswith('.rar'):
            print(f"Extracting {item_path}...")
            with rarfile.RarFile(item_path) as rar_ref:
                rar_ref.extractall(folder)
            print(f'Finished extracting {item_path}.')


def delete_files_with_ext(outer_fp: str, ext: str) -> None:
    for dirpath, _, filenames in os.walk(outer_fp):
        for file in filenames:
            if file.endswith(ext):
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def read_lines_from_file(fp: str) -> List[str]:
    with open(fp, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.replace("\n", "") for l in lines]
    return lines


def delete_files_from_folder(folder_path: str):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            except Exception as e:
                print(f"Error deleting file {item_path}: {e}")
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
            except Exception as e:
                print(f"Error deleting directory {item_path}: {e}")