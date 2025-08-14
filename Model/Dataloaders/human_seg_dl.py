from typing import *
import os
import random

from torch.utils.data import Dataset, DataLoader
import torch

import utils.path_utils as path_utils
import utils.hdf5_utils as hdf5_utils
import utils.pc_utils as pc_utils
import data_fps


def get_collate_fn(collate_fn_name: str, collate_args: Optional[dict]) -> Optional[Callable]:
    # TODO add more here
    return None


cihp_cm_num_to_str = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Gloves",
    4: "Sunglasses",
    5: "UpperClothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Torso-skin",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe",
}

cihp_cm_str_to_num = {v: k for (k, v) in cihp_cm_num_to_str.items()}

sapiens_v1_cm_num_to_str = {
    0: "Background",
    1: "Apparel",
    2: "Face_Neck",
    3: "Hair",
    4: "Left_Foot_Complete",
    5: "Left_Hand",
    6: "Left_Arm",
    7: "Left_Leg",
    8: "Lower_Clothing",
    9: "Right_Foot_Complete",
    10: "Right_Hand",
    11: "Right_Arm",
    12: "Right_Leg",
    13: "Torso",
    14: "Upper_Clothing"
}

sapiens_v1_cm_str_to_num = {v: k for (k, v) in sapiens_v1_cm_num_to_str.items()}

sapiens_v2_cm_num_to_str = {
    0: "Background",
    1: "Apparel",
    2: "Face_Neck",
    3: "Hair",
    4: "Left_Foot_Complete",
    5: "Left_Hand",
    6: "Left_Arm",
    7: "Left_Leg",
    8: "Lower_Clothing",
    9: "Right_Foot_Complete",
    10: "Right_Hand",
    11: "Right_Arm",
    12: "Right_Leg",
    13: "Torso",
    14: "Upper_Clothing",
    15: "Lip",
    16: "Teeth",
    17: "Tongue",
}

sapiens_v2_cm_str_to_num = {v: k for (k, v) in sapiens_v2_cm_num_to_str.items()}



def get_hdf5_file_fp(dataset_name: str, is_sapiens_v2: bool = False) -> str:
    assert dataset_name in ["cihp", "cihp_part", "sapiens", "sapiens_part"], \
        f"The hdf5 file path is not supported {dataset_name}"
    if dataset_name == "cihp":
        hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "cihp_sampled_pcs_data.hdf5")
    elif dataset_name == "cihp_part":
        hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "cihp_part_sampled_pcs_data.hdf5")
    elif dataset_name == "sapiens":
        if not is_sapiens_v2:
            hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiens_sampled_pcs_data.hdf5")
        else:
            hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiensv2_sampled_pcs_data.hdf5")
    else: # sapiens sampled
        if not is_sapiens_v2:
            hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiens_part_sampled_pcs_data.hdf5")
        else:
            hdf5_fp = os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiensv2_part_sampled_pcs_data.hdf5")
    assert os.path.exists(hdf5_fp), f"Path {hdf5_fp} does not exist."
    return hdf5_fp


def get_original_labels_fp(dataset_name: str, is_sapiens_v2: bool = False) -> str:
    if "cihp" in dataset_name:
        return os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "cihp_vertex_labels.hdf5")
    else:
        if is_sapiens_v2:
            return os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiens_vertex_labelsv2.hdf5")
        else:
            return os.path.join(data_fps.data_fps["Human_Seg_PC_Data"], "sapiens_vertex_labels.hdf5")



def get_filtered_keys(split_name: str, keys: List[str], dataset_name: str):
    assert split_name in ["all", "train", "test", "val", "trainval"], f"Split {split_name} not supported."
    split_outer_fp = os.path.join(path_utils.get_parent(__file__), "split_files")
    if split_name == "all":
        return keys
    elif split_name in ["train", "val", "test"]:
        split_fps = [os.path.join(split_outer_fp, dataset_name + "_" + split_name + ".txt")]
    else:  # trainval
        split_fps = [os.path.join(split_outer_fp, dataset_name + "_" + s + ".txt") for s in ["train", "val"]]

    filt_keys = []
    for split_fp in split_fps:
        with open(split_fp, "r") as f:
            key_names = f.readlines()
            key_names = [k.replace("\n", "") for k in key_names]
            filt_keys.extend(key_names)
    return filt_keys


def get_split_files(train_pct: float = 0.7, val_pct: float = 0.15):
    outer_save_fp = os.path.join(path_utils.get_parent(__file__), "split_files")
    os.makedirs(outer_save_fp, exist_ok=True)
    for j, dataset_name in enumerate(["cihp", "cihp_part", "sapiens", "sapiens_part"]):
        if j == 0 or j == 2:
            hdf5_fp = get_hdf5_file_fp(dataset_name)
            keys = hdf5_utils.get_hdf5_keys(hdf5_fp)
            num_keys = len(keys)
            random.shuffle(keys)
        for split_name in ["train", "val", "test"]:
            save_fp = os.path.join(outer_save_fp, dataset_name + "_" + split_name + ".txt")
            num_train_keys = int(train_pct*num_keys)
            num_val_keys = int(val_pct*num_keys)
            if split_name == "train":
                keys_f = keys[:num_train_keys]
            elif split_name == "val":
                keys_f = keys[num_train_keys:num_train_keys+num_val_keys]
            else:
                keys_f = keys[num_train_keys+num_val_keys:]
            with open(save_fp, "w") as f:
                num_keys_f = len(keys_f)
                for i, k in enumerate(keys_f):
                    if i != num_keys_f-1:
                        f.write(k+"\n")
                    else:
                        f.write(k)


def get_preprocess_func(preprocess_func_name: Optional[str]):
    if preprocess_func_name == "point_transformer_preprocess":
        return point_transformer_preprocess
    else:
        return None


def point_transformer_preprocess(pcd: torch.Tensor, gpu_num: int):
    if isinstance(pcd, List):
        pcd = pcd[0]
    batch_size, num_points, num_channels = pcd.shape
    pcd = pcd.reshape(-1, num_channels).float().to(gpu_num)
    xyz = pcd[..., :3]
    o = torch.tensor([(i + 1) * num_points for i in range(batch_size)], dtype=torch.int32).to(gpu_num)
    return xyz, pcd, o


class Human3DSegDataset(Dataset):
    def __init__(self, dataset_name: str, split: str,
                 with_random_rot: bool, with_point_jitter: bool,
                 is_sapiens_v2: bool = False):
        self.dataset_name = dataset_name
        self.hdf5_fp = get_hdf5_file_fp(dataset_name, is_sapiens_v2)
        self.split = split
        self.is_test = "test" in split
        self.data_keys = get_filtered_keys(split, hdf5_utils.get_hdf5_keys(self.hdf5_fp),
                                           dataset_name)
        self.with_random_rot = with_random_rot and not self.is_test
        self.with_point_jitter = with_point_jitter and not self.is_test


    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        model_name = self.data_keys[idx]
        d = hdf5_utils.load_entry_by_key_nested(self.hdf5_fp, model_name)
        vertices = d["sampled_vertices"]
        labels = torch.tensor(d["sampled_labels"]).long()

        if self.with_random_rot and not self.is_test:
            vertices = pc_utils.random_pc_rotate_y(vertices, False)

        if self.with_point_jitter and not self.is_test:
            vertices = pc_utils.random_pc_jitter(vertices)

        vertices = pc_utils.normalize_pcd_unit_sphere(vertices)

        return vertices, labels


def get_human3d_dataloader(dataset_kwargs: dict, dl_kwargs: dict,
                           collate_fn_kwargs: Optional[dict]=None):
    dataset = Human3DSegDataset(**dataset_kwargs)

    if collate_fn_kwargs is not None:
        collate_fn_name = collate_fn_kwargs["collate_fn_name"]
        del collate_fn_kwargs["collate_fn_name"]
        collate_fn = get_collate_fn(collate_fn_name, collate_fn_kwargs)
    else:
        collate_fn = None

    dl = DataLoader(dataset=dataset, **dl_kwargs, collate_fn=collate_fn)
    return dl


if __name__ == "__main__":
    get_split_files()
    dataset_kwargs_ = {"dataset_name": "sapiens_part", "split": "test",
                      "with_random_rot": True, "with_point_jitter": True,
                       "is_sapiens_v2": True}
    dataloader_kwargs_ = {"num_workers": 1, "shuffle": True, "batch_size": 2}
    dl = get_human3d_dataloader(dataset_kwargs_, dataloader_kwargs_)
    lip_sum_total = 0
    for dl_output in dl:
        # processed_vals = point_transformer_preprocess(dl_output[0], 0)
        lip_sum = torch.sum(dl_output[1]==15)
        lip_sum_total += lip_sum
        debug = "debug"
    ...
