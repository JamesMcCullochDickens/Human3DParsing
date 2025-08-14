from typing import List

import h5py # pip install h5py
import numpy as np
from filelock import FileLock


lock_path = "mydata.hdf5.lock"


def is_key_in_hdf5(hdf5_fp: str, key: str):
    with h5py.File(hdf5_fp, "r") as f:
        return key in f

def create_hdf5_dataset_from_d(data_d: dict, save_fp: str) -> None:
    with h5py.File(save_fp, "w") as hdf:
        for key, array in data_d.items():
            hdf.create_dataset(str(key), data=array)


def append_hdf5_dataset_from_d(data_d: dict, save_fp: str,
                               with_compression: bool = False,
                               with_overwrite: bool = True) -> None:
    with FileLock(lock_path):
        with h5py.File(save_fp, "a") as hdf:
            for key, array in data_d.items():
                if key in hdf:
                    if with_overwrite:  # Careful!
                        del hdf[key]
                    else:
                        continue
                if with_compression:
                    hdf.create_dataset(str(key), data=array, compression="gzip", compression_opts=9)
                else:
                    hdf.create_dataset(str(key), data=array)


def append_hdf5_dataset_from_d_nested(data_d: dict, save_fp: str,
                                   with_compression: bool = False,
                                   with_overwrite: bool = True) -> None:
    def store_group(hdf_group, data, with_compression, with_overwrite):
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgroup = hdf_group.require_group(str(key))
                store_group(subgroup, value, with_compression, with_overwrite)
            else:
                # Overwrite if the dataset exists
                if key in hdf_group:
                    if with_overwrite:
                        del hdf_group[key]
                    else:
                        continue
                # Store dataset with or without compression
                if with_compression:
                    hdf_group.create_dataset(str(key), data=value, compression="gzip")
                else:
                    hdf_group.create_dataset(str(key), data=value)
    lock_path = save_fp + ".lock"  # Define a lock path to avoid concurrent writes
    with FileLock(lock_path):
        with h5py.File(save_fp, "a") as hdf:
            store_group(hdf, data_d, with_compression, with_overwrite)


def remove_empty_list_keys(fp: str):
    with h5py.File(fp, 'a') as data_d:
        for key, array in data_d.items():
            val = array[()]
            if isinstance(val, list):
                if len(val) == 0:
                    print(f"Deleting for key {key} ...")
                    del data_d[key]
            elif isinstance(val, np.ndarray):
                if val.size == 0:
                    print(f"Deleting for key {key} ...")
                    del data_d[key]



def hdf5_to_dict(file_path: str) -> dict:
    data_dict = {}

    with h5py.File(file_path, 'r') as f:
        # Recursively traverse the HDF5 structure
        def traverse_hdf5(name, obj):
            if isinstance(obj, h5py.Dataset):
                # If obj is a dataset, convert it to a numpy array or list
                data_dict[name] = obj[()]  # Convert dataset to a numpy array
            elif isinstance(obj, h5py.Group):
                # If obj is a group (folder in HDF5 terminology), recursively traverse it
                for key, item in obj.items():
                    traverse_hdf5(name + '/' + key, item)

        # Start traversal from the root
        for key, item in f.items():
            traverse_hdf5(key, item)

    return data_dict


def get_hdf5_keys(fp: str) -> List[str]:
    with h5py.File(fp, 'r') as f:
        keys = list(f.keys())
    return keys


def load_entry_by_key(file_path: str, key_name: str):
    with h5py.File(file_path, 'r') as hdf5_file:
        if key_name in hdf5_file:
            entry = hdf5_file[key_name][()]
            return entry
        else:
            print(f"Key {key_name} not in {file_path}.")
            return None


def load_entry_by_key_nested(file_path: str, key: str) -> dict:
    def recursive_load(hdf_group):
        data = {}
        for k in hdf_group.keys():
            item = hdf_group[k]
            if isinstance(item, h5py.Group):
                # Recursively load subgroups
                data[k] = recursive_load(item)
            elif isinstance(item, h5py.Dataset):
                # Load dataset as numpy array or scalar
                data[k] = item[()]
        return data

    with h5py.File(file_path, "r") as hdf5_file:
        if key not in hdf5_file:
            print(f"Key '{key}' not found in '{file_path}'.")
            return None
        return recursive_load(hdf5_file[key])



def delete_entry_by_key(file_path: str, key: str) -> None:
    with h5py.File(file_path, "r+") as f:
        if key in f:
            print(f"Deleting dataset at key {key}...")
            del f[key]
        else:
            print(f"Key {key} not in the dataset. Returning....")