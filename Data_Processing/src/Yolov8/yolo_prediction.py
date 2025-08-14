from typing import *
import os

import numpy as np
from ultralytics import YOLO
import torch
import imagesize
import torch.nn.functional as F

import utils.path_utils as path_utils


def get_is_seg_masks(im: Union[str, np.ndarray], model_path='yolov8l-seg.pt',
                     conf: float = 0.40, iou: float = 0.3, with_resize: bool = True):
    model_path = os.path.join(path_utils.get_parent(__file__), model_path)
    model = YOLO(model_path)
    if isinstance(im, np.ndarray):
        h, w = im.shape[:2]
    else:
        w, h = imagesize.get(im)
    results = model(im, conf=conf, iou=iou, verbose=False)
    masks = results[0].masks
    if masks is None:
        return []
    mask_list = [mask.data.cpu().numpy()[0] for mask in masks]
    #conf_vals = results[0].boxes.conf.cpu().numpy()
    if with_resize:
        if mask_list[0].shape[0] != h or mask_list[1].shape[1] != w:
            mask_list = torch.tensor(mask_list).unsqueeze(1)
            mask_list = F.interpolate(mask_list, (h, w), mode="nearest").squeeze(1).numpy()
    return mask_list


def get_kps(im: Union[str, np.ndarray], model_path='reorient_pose.pt',
                     conf: float = 0.30, iou: float = 0.6):
    model_path = os.path.join(path_utils.get_parent(__file__), model_path)
    model = YOLO(model_path)
    results = model(im, conf=conf, iou=iou, verbose=False)
    kps = results[0].keypoints.data
    if kps is None:
        return []
    kps = np.array([kps_p.cpu().numpy() for kps_p in kps])
    kps, kp_conf_vals = kps[:, :, :2], kps[:, :, -1]
    return kps, kp_conf_vals


def get_is_masks_and_kps(im: Union[str, np.ndarray]):
    mask_list = get_is_seg_masks(im)
    kps, kp_conf_vals = get_kps(im)
    return {"masks": mask_list, "kps": kps, "kp_conf_vals": kp_conf_vals}


def print_keypoint_table(kps: np.ndarray, kp_conf_vals: np.ndarray):
    kp_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

    print(f"{'Keypoint':<15}{'Location (x, y)':<25}{'Confidence'}")
    print("-" * 50)
    for i, kp_name in enumerate(kp_names):
        kp_location = f"({kps[0, i, 0]:.2f}, {kps[0, i, 1]:.2f})"

        # Ensure kp_conf_vals[0, i] is a scalar
        kp_conf_value = kp_conf_vals[0, i]
        if isinstance(kp_conf_value, np.ndarray):
            kp_conf_value = kp_conf_value.item()  # Convert to scalar if it's an array

        kp_confidence = f"{kp_conf_value:.2f}"
        print(f"{kp_name:<15}{kp_location:<25}{kp_confidence}")


if __name__ == "__main__":
    sample_im_fp = "Picture of a Person.jpg"
    d = get_is_masks_and_kps(sample_im_fp)
    debug = "debug"
