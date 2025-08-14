import random
from typing import Tuple, Optional, Union, List
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from scipy.ndimage import binary_fill_holes

def percentile_clamp_image(im_arr: np.ndarray, lower_pctile: float,
                           upper_pctile: float, remove_zero: bool = False) -> np.ndarray:
    assert  (0 <= lower_pctile <= 100) and (0 <= upper_pctile <= 100), (f"Percentile values {lower_pctile} and {upper_pctile}"  
                                                                      f"are invalid, must be in the range [0, 100], not normalized")
    if remove_zero:
        im_arr_ = im_arr.reshape(-1)
        im_arr_ = im_arr_[im_arr_ != 0]
        min_val = np.percentile(im_arr_, lower_pctile)
        max_val = np.percentile(im_arr_, upper_pctile)
    else:
        min_val = np.percentile(im_arr, lower_pctile)
        max_val = np.percentile(im_arr, upper_pctile)
    im_arr_ = np.clip(im_arr, min_val, max_val)
    return im_arr_


def get_im_pct_vals(im_arr: np.ndarray, remove_zero: bool = False) -> dict:
    pct_vals = {}
    if remove_zero:
        im_arr_ = im_arr.reshape(-1)
        im_arr_ = im_arr_[im_arr_ != 0]
    else:
        im_arr_ = im_arr
    for i in range(0, 100, 10):
        pct_vals[i] = np.percentile(im_arr_, i)
    return pct_vals


def min_max_normalize(im: Union[np.ndarray, str], with_clamping: bool = False,
                      clamping_vals: Tuple[float, float] = (10, 90), remove_zero_clamping: bool = False) -> np.ndarray:
    if isinstance(im, str):
        im_arr = im_fp_to_arr(im)
    else:
        im_arr = im.copy()
    if with_clamping:
        im_arr = percentile_clamp_image(im_arr, clamping_vals[0], clamping_vals[1], remove_zero_clamping)
    min_, max_ = np.min(im_arr), np.max(im_arr)
    norm_im = (im_arr-min_)/(max_-min_)
    norm_im = norm_im * 255.0
    norm_im = norm_im.astype(np.uint8)
    return norm_im


def min_max_normalize_16_bit(im: Union[np.ndarray, str], with_clamping: bool = False,
                      clamping_vals: Tuple[float, float] = (10, 90)) -> np.ndarray:
    if isinstance(im, str):
        im_arr = im_fp_to_arr(im)
    else:
        im_arr = im.copy()
    if with_clamping:
        im_arr = percentile_clamp_image(im_arr, clamping_vals[0], clamping_vals[1])
    min_, max_ = np.min(im_arr), np.max(im_arr)
    norm_im = (im_arr-min_)/(max_-min_)
    scale_factor = (1 << 16) - 1
    norm_im = norm_im * scale_factor
    norm_im = norm_im.astype(np.uint16)
    return norm_im


def gs_to_color(im_arr: np.ndarray) -> np.ndarray:
    h, w = im_arr.shape
    color_im = np.zeros((h, w, 3), dtype=np.uint8)
    color_im[:, :, 0] = im_arr
    color_im[:, :, 1] = im_arr
    color_im[:, :, 2] = im_arr
    return color_im


def concatenate_images_horizontally(image_arrays):
    concatenated_image = np.concatenate([*image_arrays], axis=1)
    return concatenated_image


def concatenate_images_vertically(image_arrays):
    concatenated_image = np.concatenate([*image_arrays], axis=0)
    return concatenated_image


def concat_from_fps(im_fps, direction:str, save_fp: Optional[str] = None):
    ims = [im_fp_to_arr(im_fp) if isinstance(im_fp, str) else im_fp for im_fp in im_fps]
    if direction == "horizontal":
        concat_im = concatenate_images_horizontally(ims)
    else:
        concat_im = concatenate_images_vertically(ims)
    if save_fp is not None:
        save_from_arr(concat_im, save_fp)

def gs_to_rgb(gs_frame: np.ndarray) -> np.ndarray:
    gs_frame = np.expand_dims(gs_frame, axis=-1)
    gs_frame = np.tile(gs_frame, (1, 1, 3)).astype(np.uint8)
    return gs_frame


def im_fp_to_arr(im_fp: str) -> np.ndarray:
    im_arr = np.asarray(Image.open(im_fp)).copy()
    if im_arr.ndim == 3:
        im_arr = im_arr[..., :3]
    return im_arr


def im_get_min_max_nonzero(im_arr: np.ndarray) -> Tuple[int, int, int, int]:
    nonzero_indices = np.nonzero(im_arr)
    x_coords, y_coords = nonzero_indices
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return min_x, max_x, min_y, max_y


def min_max_nonzero_crop(im: Union[str, np.ndarray], save_fp: Optional[str] = None) -> np.ndarray:
    if isinstance(im, str):
        im_arr = im_fp_to_arr(im)
    else:
        im_arr = im.copy()
    if im_arr.ndim == 3:
        im_arr_ = im_arr[..., 0]
    else:
        im_arr_ = im_arr
    min_x, max_x, min_y, max_y = im_get_min_max_nonzero(im_arr_)
    cropped_im = im_arr[min_x:max_x+1, min_y:max_y+1]
    if save_fp is not None:
        save_from_arr(cropped_im, save_fp)
    return cropped_im


def im_get_min_max_nonvalue(im_arr: np.ndarray, value: int = 255) -> Tuple[int, int, int, int]:
    nonzero_indices = np.nonzero(im_arr!= value)
    y_coords, x_coords = nonzero_indices
    h, w = im_arr.shape[:2]
    if y_coords.size == 0:
        return 0, w-1, 0, h-1
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return min_x, max_x, min_y, max_y


def min_max_nonvalue_crop(im: Union[str, np.ndarray],
                          save_fp: Optional[str] = None,
                          value: int = 255, padding_px: int = 0,
                          padding_py: int = 0) -> np.ndarray:
    if isinstance(im, str):
        im_arr = im_fp_to_arr(im)
    else:
        im_arr = im.copy()
    h, w, = im_arr.shape[:2]
    if im_arr.ndim == 3:
        im_arr_ = np.mean(im_arr, axis=-1)
    else:
        im_arr_ = im_arr
    min_x, max_x, min_y, max_y = im_get_min_max_nonvalue(im_arr_, value=value)
    min_x = max(0, min_x-padding_px)
    max_x = min(w-1, max_x+padding_px)
    min_y = max(0, min_y - padding_py)
    max_y = min(h - 1, max_y + padding_py)
    cropped_im = im_arr[min_y:max_y+1, min_x:max_x+1, :]
    if save_fp is not None:
        save_from_arr(cropped_im, save_fp)
    return cropped_im



def crop_im(im: Union[str, np.ndarray], crop_pct_x, crop_pct_y, save_fp: str = None) -> np.ndarray:
    if isinstance(im, str):
        im = im_fp_to_arr(im)
    original_h, original_w = im.shape[:2]
    crop_lower_bound_x = int(crop_pct_x*original_w)//2
    crop_upper_bound_x = original_w - crop_lower_bound_x
    crop_lower_bound_y = int(crop_pct_y * original_h) // 2
    crop_upper_bound_y = original_h - crop_lower_bound_y
    if im.ndim == 2:
        cropped_im = im[crop_lower_bound_y:crop_upper_bound_y, crop_lower_bound_x:crop_upper_bound_x]
    else:
        cropped_im = im[crop_lower_bound_y:crop_upper_bound_y, crop_lower_bound_x:crop_upper_bound_x, :]
    if save_fp is not None:
        save_from_arr(cropped_im, save_fp)
    return cropped_im


def crop_and_resize_im(im: Union[str, np.ndarray], crop_pct_x, crop_pct_y, new_res: Tuple[int, int], save_fp: str = None) -> np.ndarray:
    if isinstance(im, str):
        im = im_fp_to_arr(im)
    cropped_im = crop_im(im, crop_pct_x, crop_pct_y)
    resized_im = resize_im_arr(cropped_im, new_res)
    if save_fp is not None:
        save_from_arr(resized_im, save_fp)
    return resized_im


def show_im(im: Union[str, np.ndarray]) -> None:
    if isinstance(im, str):
        im_pil = Image.open(im)
    else:
        im_pil = Image.fromarray(im)
    im_pil.show()


def save_from_arr(im_arr: np.ndarray, save_fp: str) -> None:
    im_pil = Image.fromarray(im_arr)
    im_pil.save(save_fp)


def resize_im_arr(im_arr: Union[np.ndarray, str], new_res: Tuple[int, int],
                  save_fp: Optional[str] = None, mode=Image.BILINEAR) -> np.ndarray:
    if isinstance(im_arr, str):
        im_arr = im_fp_to_arr(im_arr)
    im_arr_pil = Image.fromarray(im_arr)
    # new res should be height, width
    # PIL's resize is width by height
    im_arr_pil = im_arr_pil.resize(resample=mode, size=(new_res[1], new_res[0]))
    if save_fp is not None:
        im_arr_pil.save(save_fp)
    return np.asarray(im_arr_pil)


def is_value_image(im_arr: np.ndarray, value: int) -> bool:
    return np.all(im_arr == value)



def float_to_viridis(value, min_val, max_val):
  normalized_value = (value - min_val) / (max_val - min_val)
  cmap = cm.get_cmap('viridis')
  rgba = cmap(normalized_value)
  rgb = rgba[:3]
  return rgb


def visualize_class_scores(scores, im_block_res: Tuple[int, int] = (40, 15),
                           save_fp: Optional[str]=None) -> np.ndarray:


    # scores = scores/np.linalg.norm(scores)
    max_, min_ = np.max(scores), np.min(scores)
    num_classes = scores.shape[0]
    im = np.zeros(shape=(im_block_res[0], num_classes*im_block_res[1], 3), dtype=np.uint8)
    im.fill(125)
    for c_num in range(num_classes):
        lb = c_num*im_block_res[1]
        ub = (c_num+1)*im_block_res[1]
        score_color = float_to_viridis(scores[c_num], min_, max_)
        im[:, lb:ub + 1, 0] = int(score_color[0]*255)
        im[:, lb:ub + 1, 1] = int(score_color[1]*255)
        im[:, lb:ub + 1, 2] = int(score_color[2]*255)
    show_im(im)
    if save_fp is not None:
        save_from_arr(im, save_fp)
    return im



def image_dots_image(im1: Union[str, np.ndarray], im2: Union[str, np.ndarray],
                     is_vertical: bool = True, save_fp: Optional[str] = None,
                     dot_color: Tuple[int, int, int] = (0, 0, 0), bkg_gs_val: int = 255) -> np.ndarray:
    if isinstance(im1, str):
        im1 = im_fp_to_arr(im1)
    if isinstance(im2, str):
        im2 = im_fp_to_arr(im2)

    if not is_vertical:
        height = im1.shape[1]
        image_size = (height, 100, 3)
        dot_size = 8

        dot_image = np.zeros(image_size, dtype=np.uint8)
        dot_image.fill(bkg_gs_val)
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        for i in range(3):
            x = center_x + (i - 1) * 2 * dot_size
            cv2.circle(dot_image, (x, center_y), dot_size, dot_color, -1)

        ims = [im1, dot_image, im2]
        im_dots_im_arr = concatenate_images_horizontally(ims)
    else:
        width = im1.shape[1]
        image_size = (100, width, 3)
        dot_size = 8

        dot_image = np.zeros(image_size, dtype=np.uint8)
        dot_image.fill(bkg_gs_val)
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        for i in range(3):
            y = center_y + (i - 1) * 2 * dot_size
            cv2.circle(dot_image, (center_x, y), dot_size, dot_color, -1)

        ims = [im1, dot_image, im2]
        im_dots_im_arr = concatenate_images_vertically(ims)
    if save_fp is not None:
        save_from_arr(im_dots_im_arr, save_fp)
    return im_dots_im_arr



def color_mask_to_instance_mask(mask_image: np.ndarray) -> np.ndarray:
    unique_colors = np.unique(mask_image.reshape(-1, mask_image.shape[-1]), axis=0)
    color_to_int = {tuple(color): i for i, color in enumerate(unique_colors)}
    integer_image = np.zeros(mask_image.shape[:2], dtype=np.int32)
    for i in range(mask_image.shape[0]):
        for j in range(mask_image.shape[1]):
            integer_image[i, j] = color_to_int[tuple(mask_image[i, j])]
    return integer_image


def get_color_map(num_vals: int = 100) -> dict:
    keys = list(range(-1, num_vals))
    color_d = {}
    biases, gain, power = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, power)) % 255.0
        g = int(math.pow((key+biases[1])*gain, power)) % 255.0
        b = int(math.pow((key+biases[2])*gain, power)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d


def seg_mask_to_color_im(seg_mask: np.ndarray) -> np.ndarray:
    unique = np.unique(seg_mask)
    num_vals = np.max(unique)
    color_d = get_color_map(num_vals=num_vals+1)
    color_d[0] = np.array([0, 0, 0], dtype=np.uint8)
    color_mask = np.zeros(shape=(*seg_mask.shape, 3), dtype=np.uint8)
    for val in unique:
        if val != 0:
            color_mask[seg_mask==val] = color_d[val]
    return color_mask


def show_seg_mask(seg_mask: np.ndarray) -> None:
    color_mask = seg_mask_to_color_im(seg_mask)
    show_im(color_mask)


def pad_im_horizontally(im: np.ndarray, pad_left: bool, pad_pixels: int) -> np.ndarray:
    h, w = im.shape[:2]
    if im.ndim == 2:
        padded_im = np.zeros(shape=(h, w + pad_pixels), dtype=np.uint8)
        if pad_left:
            padded_im[:, pad_pixels:] = im
        else:
            padded_im[:, :w] = im
    else:
        padded_im = np.zeros(shape=(h, w + pad_pixels, 3), dtype=np.uint8)
        if pad_left:
            padded_im[:, pad_pixels:, :] = im
        else:
            padded_im[:, :w, :] = im
    return padded_im


def draw_number_on_im(im_arr: np.ndarray, number: int, position: Tuple[int, int],
                      color: Tuple = (255, 0, 0)) -> np.ndarray:
    image_pil = Image.fromarray(im_arr)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default(size=40)
    draw.text(position, str(number), fill=color, font=font)
    image_np_modified = np.array(image_pil)
    return image_np_modified


def get_kp_im(im, kps, h, w, point_radius: int = 5):
    if im is None:
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        img_array = im.copy()
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    for x, y in kps:
        x_ = max(0, min(round(x), w - 1))
        y_ = max(0, min(round(y), h - 1))
        left_up_point = (max(0, x_ - point_radius), max(0, y_ - point_radius))
        right_down_point = (min(h-1, x_ + point_radius), min(h-1, y_ + point_radius))
        draw.ellipse([left_up_point, right_down_point], fill=(255, 0, 0))
    return np.asarray(img)


def get_legend_with_im(im_arr, color_labels: dict, save_fp: str, text: Optional[str] = None, legend_scale: float = 1.0) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.imshow(im_arr)
    ax.axis('off')  # Hide axes

    # Calculate legend dimensions
    max_label_width = max(len(label) for label in color_labels)
    legend_width = max_label_width * 55 * legend_scale  # Adjust based on scale

    # Create legend patches
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_labels.items()]

    # Add legend to the right side with scaled font size
    legend = ax.legend(
        handles=patches,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=8 * legend_scale,
        frameon=False,
        handlelength=2 * legend_scale,
        handleheight=1 * legend_scale,
        borderpad=1 * legend_scale,
        labelspacing=0.5 * legend_scale
    )

    # Optional text
    if text is not None:
        fig.text(0.5, 0.27, text, ha='center', va='bottom', fontsize=14 * legend_scale)

    # Adjust figure size
    fig.set_size_inches(
        im_arr.shape[1] / 100 + legend_width / 100,
        im_arr.shape[0] / 100 + legend_width / 100
    )

    plt.tight_layout()
    plt.savefig(save_fp, bbox_inches='tight')
    plt.close()

    im = im_fp_to_arr(save_fp)
    return im


def get_legend_im(color_labels: dict, save_fp: str, text: Optional[str] = None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, len(color_labels) * 0.5))  # Adjust size for legend
    ax.axis('off')  # Hide axes as we're focusing only on the legend

    # Create legend patches
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_labels.items()]

    # Add legend to the center of the figure
    legend = ax.legend(handles=patches, loc='center', fontsize='small', frameon=False)

    # Optionally add text below the legend
    if text is not None:
        fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=14)

    # Save the legend as an image
    plt.savefig(save_fp, bbox_inches='tight', dpi=100)
    plt.close()

    # Load the saved image to return as a NumPy array
    im = im_fp_to_arr(save_fp)
    return im



def get_2d_embedding_vis(embed: np.ndarray, save_fp: Optional[str] = None,
                         square_size: float = 1, spacing: float = 0.2, with_display: bool = False,
                         color_type:str = "jet"):
    num_rows, num_cols = embed.shape[:2]
    fig, ax = plt.subplots()
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * (square_size + spacing)
            y = row * (square_size + spacing)
            if color_type == "viridis":
                color = plt.cm.viridis(embed[row, col])
            else:
                color = plt.cm.jet(embed[row, col])
            rect = FancyBboxPatch((x, y), square_size, square_size,
                                  boxstyle="round,pad=0, rounding_size=0.2",
                                  facecolor=color)
            ax.add_patch(rect)
    ax.set_xlim(0, num_cols * (square_size + spacing) - spacing)
    ax.set_ylim(0, num_rows * (square_size + spacing) - spacing)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if save_fp is not None:
        plt.savefig(save_fp)
    if with_display:
        plt.show()
    plt.close()


def get_rand_temporal_embedding(num_frames: int, num_splits: int,
                                dimension: int, save_fp: Optional[str] = None,
                                with_display: bool = False) -> np.ndarray:
    t_embed = np.zeros((num_frames, dimension))
    div = num_frames//num_splits
    rem = num_frames - (div*num_splits)
    for i in range(0, num_frames, div):
        t_embed[i:(i+1)*div, :] = np.random.rand(1)
    if rem != 0:
        t_embed[-rem:, :] = np.random.rand(1)
    # add very small random noise
    t_embed += np.random.rand(*t_embed.shape)/2.0
    t_embed = t_embed.transpose()
    if save_fp is not None:
        get_2d_embedding_vis(t_embed, save_fp=save_fp, with_display=with_display)
    return t_embed


def get_trapezoid_im_with_text(text: str, is_vertical: bool = True,
                               save_fp: Optional[str] = None,
                               with_display: bool = False):
    # Define trapezoid coordinates
    if not is_vertical:
        coords = [(0, 0), (1, 0), (0.6, 1), (0.4, 1)]
    else:
        coords = [(0, 0), (0, 1), (1, 0.6), (1, 0.4)]
    poly = Polygon(coords, closed=True, facecolor='lightblue', edgecolor='black')
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.add_patch(poly)
    text_x = (poly.xy[:, 0].min() + poly.xy[:, 0].max()) / 2
    text_y = (poly.xy[:, 1].min() + poly.xy[:, 1].max()) / 2
    ax.text(text_x, text_y, text, ha='center', va='center', fontsize=25)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if save_fp is not None:
        plt.savefig(save_fp)
    if with_display:
        plt.show()
    plt.close()



def get_square_im_with_text(text: str, save_fp: Optional[str] = None,
                               with_display: bool = False):
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = Polygon(coords, closed=True, facecolor='lightblue', edgecolor='black')
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    text_x = (poly.xy[:, 0].min() + poly.xy[:, 0].max()) / 2
    text_y = (poly.xy[:, 1].min() + poly.xy[:, 1].max()) / 2
    ax.text(text_x, text_y, text, ha='center', va='center', fontsize=18)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if save_fp is not None:
        plt.savefig(save_fp)
    if with_display:
        plt.show()
    plt.close()


def get_rand_im(height, width, is_gs: bool = False):
    if is_gs:
        rand_im = np.random.randint(low=0, high=256, size=(height, width), dtype=np.uint8)
    else:
        rand_im = np.random.randint(low=0, high=256, size=(height, width, 3), dtype=np.uint8)
    return rand_im


def normalize_depth_im(im_arr: np.ndarray):
    min_, max_ = np.min(im_arr), np.max(im_arr)
    if min_ == max_:
        print(f"Min is equal to max, returning white image.")
        h, w = im_arr.shape[:2]
        im_arr_ = np.zeros((h, w), dtype=im_arr.dtype)
        im_arr_.fill(255.0)
        return im_arr_
    im_arr_ = (im_arr - min_)/(max_-min_)
    im_arr_ *= 255.0
    im_arr_ = im_arr_.astype(np.uint8)
    return im_arr_


def show_depth_im_normalized(depth_im):
    depth_im_ = normalize_depth_im(depth_im)
    show_im(depth_im_)


def time_interval_im(num_intervals: int = 10, length: int = 200, save_fp: Optional[str]=None):
    colors = get_color_map(num_intervals)
    height_per_interval = 8
    im = np.zeros((height_per_interval*num_intervals, length, 3), dtype=np.uint8)
    im.fill(255)
    for i in range(num_intervals):
        rand_lb = random.randint(length//4, length//2)
        rand_up = random.randint(rand_lb, length-5)
        im[i*height_per_interval:(i+1)*height_per_interval, rand_lb:rand_up, :3] = colors[i]
    #show_im(im)
    text_labels = {"Class Label " + str(i): colors[i]/255.0 for i in range(num_intervals)}
    legend_im = get_legend_im(im, text_labels, save_fp=save_fp, text="Frame Intervals")
    min_max_nonvalue_crop(legend_im, save_fp=save_fp)


def denoise_with_non_local_means(current_frame, prev_frame, next_frame, h=10, search_window=21, block_size=7):
    # Convert the frames to float32 for averaging
    current_frame = current_frame.astype(np.float32)
    prev_frame = prev_frame.astype(np.float32)
    next_frame = next_frame.astype(np.float32)

    # Average the neighboring frames
    avg_frame = (prev_frame + current_frame + next_frame) / 3

    # Denoise the averaged frame using Non-Local Means
    denoised_frame = cv2.fastNlMeansDenoisingColored(
        src=avg_frame.astype(np.uint8),
        h=h,
        templateWindowSize=block_size,
        searchWindowSize=search_window
    )

    return denoised_frame


def show_binary_instance_mask(mask: np.ndarray) -> None:
    im = np.where(mask == 1, 255, 0)
    if im.ndim == 2:
        im = np.expand_dims(im, axis=-1)
        im = np.tile(im, (1, 1, 3))
    im = im.astype(np.uint8)
    show_im(im)


def get_binary_mask_contour(binary_mask: np.ndarray):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def fill_binary_mask_contour(binary_mask_: np.ndarray):
    if is_empty_mask(binary_mask_):
        return binary_mask_
    if binary_mask_.dtype != np.uint8:
        binary_mask = binary_mask_.astype(np.uint8)
    else:
        binary_mask = binary_mask_
    contours = get_binary_mask_contour(binary_mask)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, (1), cv2.FILLED)
    return filled_mask


def fill_binary_instance_mask_holes(binary_mask: np.ndarray):
    return binary_fill_holes(binary_mask).astype(np.uint8)


def is_empty_mask(binary_mask: np.ndarray):
    return not np.any(binary_mask)


def extend_crop(x_min, y_min, x_max, y_max, im_h, im_w, target_width, target_height):
    crop_width = x_max-x_min
    crop_height = y_max-y_min

    width_to_add = target_width-crop_width
    width_to_add_left = (target_width-crop_width)//2
    width_to_add_right = width_to_add_left + (width_to_add % 2)
    height_to_add = target_height-crop_height
    height_to_add_up = (target_height-crop_height)//2
    height_to_add_down = (target_height-crop_height)//2 + (height_to_add % 2)

    left_space = x_min
    right_space = (im_w-1)-x_max

    up_space = y_min
    down_space = (im_h-1)-y_max

    # horizontal cropping
    if width_to_add <= 0:
        crop_x_min = x_min
        crop_x_max = x_max
    else:
        if left_space >= width_to_add_left and right_space >= width_to_add_right:
            crop_x_min = x_min - width_to_add_left
            crop_x_max = x_max + width_to_add_right
        elif left_space >= width_to_add_left and right_space < width_to_add_right:
            remainder = width_to_add_right - right_space
            if x_min-width_to_add_left-remainder > 0:
                crop_x_min = x_min - width_to_add_left - remainder
                crop_x_max = x_max + width_to_add_right - remainder
            else:
                crop_x_min = 0
                crop_x_max = im_w-1
        elif right_space >= width_to_add_right and left_space < width_to_add_left:
            remainder = width_to_add_left - left_space
            if x_max+width_to_add_right+remainder < im_w:
                crop_x_min = x_min - width_to_add_left + remainder
                crop_x_max = x_max + width_to_add_right + remainder
            else:
                crop_x_min = 0
                crop_x_max = im_w-1
        else:
            crop_x_min = 0
            crop_x_max = im_w-1

    # vertical cropping
    if height_to_add <= 0:
        crop_y_min = y_min
        crop_y_max = y_max
    else:
        if up_space >= height_to_add_up and down_space >= height_to_add_down:
            crop_y_min = y_min - height_to_add_down
            crop_y_max = y_max + height_to_add_up
        elif up_space >= height_to_add_up and down_space < height_to_add_down:
            remainder = height_to_add_down-down_space
            if up_space >= height_to_add_up + remainder:
                crop_y_min = y_min - height_to_add_up - remainder
                crop_y_max = y_max + height_to_add_up - remainder
            else:
                crop_y_min = 0
                crop_y_max = im_h-1
        elif down_space >= height_to_add_down and up_space < height_to_add_up:
            remainder = height_to_add_up - up_space
            if down_space >= height_to_add_down + remainder:
                crop_y_min = y_min - height_to_add_up + remainder
                crop_y_max = y_max + height_to_add_down + remainder
            else:
                crop_y_min = 0
                crop_y_max = im_h - 1
        else:
            crop_y_min = 0
            crop_y_max = im_h-1

    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


def resize_bilinear(img, new_width, new_height):
    old_height, old_width = img.shape
    resized_img = np.zeros((new_height, new_width), dtype=img.dtype)
    scale_x = old_width / new_width
    scale_y = old_height / new_height
    for i in range(new_height):
        for j in range(new_width):
            x = j * scale_x
            y = i * scale_y
            resized_img[i, j] = bilinear_interpolation(img, x, y)
    return resized_img


def bilinear_interpolation(img, x, y):
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
    Q11 = img[y0, x0]
    Q21 = img[y0, x1]
    Q12 = img[y1, x0]
    Q22 = img[y1, x1]
    wx = x - x0
    wy = y - y0
    interpolated_value = (Q11 * (1 - wx) * (1 - wy) +
                          Q21 * wx * (1 - wy) +
                          Q12 * (1 - wx) * wy +
                          Q22 * wx * wy)
    return interpolated_value


def is_all_black(image) -> bool:
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(image.shape) == 3:  # RGB
        return np.all(image == 0)
    elif len(image.shape) == 2:  # Grayscale
        return np.all(image == 0)
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")


def is_all_white(image: np.ndarray) -> bool:
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(image.shape) == 3:  # RGB
        return np.all(image == 255)
    elif len(image.shape) == 2:  # Grayscale
        return np.all(image == 255)
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")


def kps_to_seg_mask(kps, h, w):
    seg_mask = np.zeros((h, w), dtype=int)
    for i, (x, y) in enumerate(kps):
        x_ = max(0, min(round(x), w-1))
        y_ = max(0, min(round(y), h-1))
        seg_mask[y_, x_] = i+1
    return seg_mask

if __name__ == "__main__":
    ...
