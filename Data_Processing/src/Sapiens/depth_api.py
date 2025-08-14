from typing import *
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import Visualization.frame_utils as frame_utils
import Sapiens.part_seg_api as part_seg_api

directory_fp = os.path.dirname(os.getcwd())
ckpt_fp = os.path.join(directory_fp, "Sapiens/TS_Files/depth/sapiens_0.3b_render_people_epoch_100_torchscript.pt2")
assert os.path.exists(ckpt_fp), f"ts path does not exist"
im_mean=[123.5, 116.5, 103.5]
im_std=[58.5, 57.0, 57.5]
model = None

def load_model(checkpoint_fp: str, use_torchscript: bool=True):
    if use_torchscript:
        return torch.jit.load(checkpoint_fp)
    else:
        return torch.export.load(checkpoint_fp).module()


def preprocess_input(im:[str, np.ndarray], im_resize_res: Tuple[int, int]) -> torch.Tensor:
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
    del output
    print("Finished warming up.")
    return

@ torch.no_grad()
def get_depth_on_im(im: [str, np.ndarray], checkpoint_fp: Optional[str]=None,
                       device_id: Optional[int] = 0, save_fp: Optional[str]=None):
    if checkpoint_fp is None:
        global ckpt_fp
        checkpoint_fp = ckpt_fp
    global model
    if model is None:
        model = load_model(checkpoint_fp, True)
        model = model.to(device_id)
        #warmup_model(model=model, device_id=device_id)
    if isinstance(im, str):
        im = np.asarray(Image.open(im))
    h, w, _ = im.shape
    im = preprocess_input(im, (1024, 768))
    if device_id is not None:
        im = im.to(device_id)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(im)
    depth = F.interpolate(output, size=(h, w), mode="bilinear")[0][0]
    depth = depth.cpu().numpy()
    if save_fp is not None: # TODO
        ...
    return depth


@torch.no_grad()
def time_model_inference():
    import time
    model = load_model(ckpt_fp, True)
    model = model.to(0)
    #warmup_model(model=model, device_id=0)
    vid = torch.randn(100, 3, 1024, 768).to(0)
    t1 = time.time()
    for frame in vid:
        frame = torch.unsqueeze(frame, dim=0)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = model(frame)
    t2 = time.time()
    print(f"Vid inference took {t2-t1} seconds.")



def get_depth_and_seg(im: Union[str, np.ndarray]):
    seg_im = part_seg_api.get_part_seg_on_im(im, device_id=0)
    depth_im = get_depth_on_im(im, device_id=1)
    return seg_im, depth_im


if __name__ == "__main__":
    #time_model_inference()
    ...

