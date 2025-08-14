import pickle
import json


def write_cached_dict(file_path: str, d: dict) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(d, f)


def read_cached_dict(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        d = pickle.load(f)
    return d


def write_readable_cached_dict(file_path: str, d: dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def read_readable_cached_dict(file_path: str) -> dict:
    with open(file_path) as f:
        dict = json.load(f)
    return dict
