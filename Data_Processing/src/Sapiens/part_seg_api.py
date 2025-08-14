from typing import Optional, Union, Tuple

import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

import utils.frame_utils as frame_utils
import src.Sapiens.part_seg_info as part_seg_info
import utils.path_utils as path_utils

proj_outer_fp = path_utils.get_parent(__file__)
ckpt_fp = os.path.join(proj_outer_fp, "TS_Files/seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2")
assert os.path.exists(ckpt_fp), f"ts path does not exist"
im_mean=[123.5, 116.5, 103.5]
im_std=[58.5, 57.0, 57.5]
model = None

def load_model(checkpoint_fp: str, use_torchscript: bool=True):
    if use_torchscript:
        return torch.jit.load(checkpoint_fp)
    else:
        return torch.export.load(checkpoint_fp).module()


@torch.no_grad()
def warmup_model(model, batch_size=8,
                 device_id: int = 0, warmup_iterations: int = 3):
    print(f"Warming up the model with a batch size of {batch_size} for {warmup_iterations} iterations ...")
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.float16).to(device_id)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        for i in range(warmup_iterations):
            output = model(imgs)
    output = output.detach().cpu().float().numpy()
    del imgs
    print("Finished warming up.")
    return


def preprocess_input(im: Union[str, np.ndarray], im_resize_res: Tuple[int, int]) -> torch.Tensor:
    if isinstance(im, str):
        im_pil = Image.open(im)
    else:
        im_pil = Image.fromarray(im)
    if im.shape[0] != im_resize_res[0] or im.shape[1] != im_resize_res[1]:
        im_pil = im_pil.resize(size=(im_resize_res[1], im_resize_res[0]), resample=Image.BILINEAR)
        im = np.asarray(im_pil)
    im = torch.tensor(im, dtype=torch.float)
    im = torch.unsqueeze(im, dim=0)
    im = torch.permute(im, (0, -1, 1, 2))
    for i in range(3):
        im[:, i, :, :] = (im[:, i, :, :]-im_mean[i])/(im_std[i])
    return im


@ torch.no_grad()
def get_part_seg_on_im(im_: [str, np.ndarray], checkpoint_fp: Optional[str]=None,
                       device_id: Optional[int] = 0, save_fp: Optional[str]=None):
    if checkpoint_fp is None:
        global ckpt_fp
        checkpoint_fp = ckpt_fp
    global model
    if model is None:
        model = load_model(checkpoint_fp, True)
        model = model.to(device_id)
        #warmup_model(model=model, device_id=device_id)
    if isinstance(im_, str):
        im = np.asarray(Image.open(im_))
    else:
        im = im_.copy()
    h, w, _ = im.shape
    im = preprocess_input(im, im_resize_res=(1024, 768))
    if device_id is not None:
        im = im.to(device_id)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        results = model(im)
    results = F.interpolate(results, size=(h, w), mode="bilinear")[0]
    seg_labels = torch.argmax(results, dim=0).cpu().numpy().astype(np.uint8)
    if save_fp is not None:
        save_seg_mask(seg_labels, save_fp=save_fp)
    return seg_labels


def class_seg_im_to_binary_im(class_seg_im):
    binary_im = np.where(class_seg_im != 0, 1, 0)
    return binary_im


def viz_seg_im(seg_labels: np.ndarray):
    d = part_seg_info.part_d_num_to_str
    num_part_labels = len(d.keys())
    seg_color_im = frame_utils.seg_mask_to_color_im(seg_labels)
    color_d = frame_utils.get_color_map(num_part_labels)
    print(color_d)
    d_= {}
    for i in d.keys():
        d_[d[i]] = color_d[i]/255.0 if i != 0 else np.array([0.0, 0.0, 0.0])
    legend_im = frame_utils.get_legend_im(seg_color_im, color_labels=d_, save_fp="./temp.png")
    frame_utils.show_im(legend_im)
    os.remove("./temp.png")


def save_seg_mask(im: np.ndarray, save_fp: str, save_form:str ="png"):
    if save_form == "png":
        im_pil = Image.fromarray(im)
        im_pil.save(save_fp)
    else:
        np.savez(save_fp, **{"part_seg_mask" : im})


if __name__ == "__main__":
    sample_im_fp = "/home/jamesdickens/Desktop/Code/sapiens/Sample_Images/Person.jpg"
    seg_labels = get_part_seg_on_im(sample_im_fp)
    binary_im = class_seg_im_to_binary_im(seg_labels)
    frame_utils.show_binary_instance_mask(binary_im)
    viz_seg_im(seg_labels)
    ...