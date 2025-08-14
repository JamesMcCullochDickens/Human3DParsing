from typing import List, Tuple, Optional
import os

import glob
import random

import numpy as np
import multiprocessing as mp

import tqdm
import json
from PIL import Image
from pycocotools import mask as maskUtils
from skimage import measure
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.path_utils as path_utils
from utils.pc_utils import farthest_point_sampling

os.environ["cwd"] = path_utils.get_parent(os.getcwd())
import utils.file_utils as file_utils
import data_fps
import src.PyOpenGL.pyopgenl_mesh_human_seg as pyopengl_mesh_human_seg
import src.PyOpenGL.obj_parser as obj_parser
import src.PyOpenGL.third_party_obj_loading_utils as obj_loader
import src.PyOpenGL.pyopengl_pc as pyopengl_pc_vis
import utils.hdf5_utils as hdf5_utils
import src.PyOpenGL.person_kp_pose_reorientv2 as p_orient
import utils.frame_utils as frame_utils
import src.PyOpenGL.seg_labels as seg_labels
import utils.pc_vis as pc_vis
import utils.pc_utils as pc_utils
import src.PyOpenGL.render_part_pc_4view as render_4view
import utils.ifps_cuda.farthest_point_sampling as fps
import src.PyOpenGL.pyopengl_utils as pyopengl_utils

outer_pc_save_fp = os.path.join(data_fps.data_fps["Renders_Outer"], "PCs")
hdf5_save_fp = os.path.join(outer_pc_save_fp, "original_models.hdf5")
os.makedirs(outer_pc_save_fp, exist_ok=True)
data_outer_save_fp = data_fps.data_fps["Human_Seg_PC_Data"]

cihp_labels = {
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

cihp_labels_to_ignore = [1, 3, 4, 11]


sapiens_labels_v1 = {
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


sapiens_labels_v2 = {
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


def get_outer_obj_folder_fps(with_save: bool = False):
    root_path = data_fps.data_fps["Human_3D_Models"]

    obj_folders_abs = []
    obj_folder_names = []

    for dirpath, _, _ in os.walk(root_path):
        if glob.glob(os.path.join(dirpath, "*.obj")):
            obj_folders_abs.append(dirpath)
            obj_folder_names.append(dirpath.replace(root_path, ""))

    obj_folders_abs.sort()
    obj_folder_names.sort()

    if with_save:
        abs_save_fp = "./obj_model_abs_paths.txt"
        folder_names_save_fp = "./obj_folder_names.txt"
        file_utils.write_lines_to_file(abs_save_fp, "w", obj_folders_abs)
        file_utils.write_lines_to_file(folder_names_save_fp, "w", obj_folder_names)

    d = {"obj_absolute_paths": obj_folders_abs,
         "obj_folder_names": obj_folder_names}

    return d


def get_random_sample(num_samples: int = 50):
    abs_fps = get_outer_obj_folder_fps()["obj_folder_names"]
    spec_fps = []
    for ab_fp in abs_fps:
        if "multihuman_single_raw" in ab_fp:
            spec_fps.append(ab_fp)
        if "multihuman_three_raw" in ab_fp:
            spec_fps.append(ab_fp)
    random.shuffle(spec_fps)
    sample_fps = random.sample(abs_fps, k=num_samples-10)
    sample_fps.extend(spec_fps[:10])
    random.shuffle(sample_fps)
    file_utils.write_lines_to_file("sample_obj_folders.txt", "w", sample_fps)


def get_rand_samp_fps():
    lines = file_utils.read_lines_from_file("./sample_obj_folders.txt")
    root_path = data_fps.data_fps["Human_3D_Models"]
    fps = [os.path.join(root_path, f[1:]) for f in lines]
    return fps


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    all_labels = measure.label(mask)
    props = measure.regionprops_table(all_labels, properties=('label', 'area'))
    bad_labels = props['label'][props['area'] < min_size]
    mask[np.isin(all_labels, bad_labels)] = 0
    return mask


def save_render_output(d, outer_save_fp):
    obj_name = d["obj_name"]
    fold_save_fp = os.path.join(outer_save_fp, obj_name)
    os.makedirs(fold_save_fp, exist_ok=True)
    color_ims = d["color_ims"]
    depth_ims = d["depth_ims"]
    for i, (color_im, depth_im) in enumerate(zip(color_ims, depth_ims)):
        color_im_save_fp = os.path.join(fold_save_fp, "color_im_"+str(i)+".png")
        color_im_pil = Image.fromarray(color_im)

        # Sanity check
        #color_im_pil.show()
        #frame_utils.show_depth_im_normalized(depth_im)

        color_im_pil.save(color_im_save_fp)

        depth_im_save_fp = os.path.join(fold_save_fp, "depth_im_"+str(i)+".npz")
        np.savez(depth_im_save_fp, depth_im=depth_im)

    vertices_save_fp = os.path.join(fold_save_fp, "vertices.npz")
    np.savez(vertices_save_fp, vertices=d["vertices"].astype(np.float16), faces=d["faces"].astype(int))
    info_save_fp = os.path.join(fold_save_fp, "info.npz")
    np.savez(info_save_fp, model_matr=d["model_matr"], projection_matr=d["projection_matr"],
             height=d["im_height"], width=d["im_width"], view_matr=d["view_matr"])


def render_and_get_output(obj_folder_fps: List[str], outer_save_fp: str,
                          use_pyassimp: bool = True, with_overwrite: bool = False):
    for obj_folder_fp in tqdm.tqdm(obj_folder_fps):
        obj_fp = obj_parser.get_first_obj(obj_folder_fp)
        parent_fold_name = os.path.basename(path_utils.get_parent(obj_fp))
        grandparent_fold_name = os.path.basename(path_utils.get_parent(path_utils.get_parent(obj_fp)))
        obj_name = os.path.basename(grandparent_fold_name + "_" + parent_fold_name)
        fold_save_fp = os.path.join(outer_save_fp, obj_name)
        if os.path.exists(fold_save_fp) and not with_overwrite:
            print(f"Already written at {fold_save_fp}")
            continue

        if use_pyassimp:
            mesh_d = obj_loader.load_obj_mesh_pyassimp(obj_fp)
        else:
            obj = obj_parser.Obj(obj_fp)
            mesh_d = obj_parser.get_first_mesh_d(obj)
        output_d = pyopengl_mesh_human_seg.get_human_mesh_render(mesh_d, show_window=False)
        output_d["obj_name"] = obj_name
        save_render_output(output_d, outer_save_fp)
        print(f"Rendered {obj_name}\nSaving to {outer_save_fp}...")
        #except Exception as err:
        #    print(f"Exception {err} occurred for obj {obj_fp}")
        #    file_utils.append_lines_to_file(error_file_fp, [obj_fp+"\n"])


def get_all_models_hdf5(obj_folder_fps_batch: List[str], hdf5_save_fp: str,
                          use_pyassimp: bool = True):
    hdf5_outer_folder = path_utils.get_parent(hdf5_save_fp)
    texture_ims_fp = os.path.join(hdf5_outer_folder, "textures")
    os.makedirs(texture_ims_fp, exist_ok=True)
    for obj_num, obj_folder_fp in enumerate(obj_folder_fps_batch):
        print(f"Processing {obj_folder_fp} for {obj_num+1} out of {len(obj_folder_fps_batch)} files... in process {os.getpid()}")
        obj_fp = obj_parser.get_first_obj(obj_folder_fp)
        parent_fold_name = os.path.basename(path_utils.get_parent(obj_fp))
        grandparent_fold_name = os.path.basename(path_utils.get_parent(path_utils.get_parent(obj_fp)))
        obj_name = os.path.basename(grandparent_fold_name + "_" + parent_fold_name)
        if hdf5_utils.is_key_in_hdf5(hdf5_save_fp, obj_name):
            continue
        if use_pyassimp:
            mesh_d = obj_loader.load_obj_mesh_pyassimp(obj_fp)
        else:
            obj = obj_parser.Obj(obj_fp)
            mesh_d = obj_parser.get_first_mesh_d(obj)
        vertices = mesh_d["vertices"]
        faces = mesh_d["faces"]
        uv_coords = mesh_d["uv_coords"]
        num_t = len(mesh_d["im_file_fps"])
        if num_t < 1:
            continue
        d = {obj_name: {"vertices": vertices.astype(np.float16), "faces": faces, "uv_coords": uv_coords.astype(np.float16)}}
        texture_im = mesh_d["texture"][..., :3]
        texture_save_fp = os.path.join(texture_ims_fp, obj_name+".jpg")
        texture_im_pil = Image.fromarray(texture_im)
        texture_im_pil.save(texture_save_fp)
        hdf5_utils.append_hdf5_dataset_from_d_nested(d, hdf5_save_fp, with_compression=True, with_overwrite=True)


def read_saved_results(save_fp: str) -> dict:
    d = {}

    # read the part masks
    part_masks_save_fp = os.path.join(save_fp, "part_masks.png")
    if os.path.exists(part_masks_save_fp):
        part_masks_im = np.asarray(Image.open(part_masks_save_fp)).astype(int)
        d["part_mask"] = part_masks_im
    else:
        d["part_mask"] = []

    # read the instance masks saved in run length encoding as a json
    person_mask_info_json_fp = os.path.join(save_fp, "person_info.json")
    if os.path.exists(person_mask_info_json_fp):
        with open(person_mask_info_json_fp, "r") as f:
            rles_d = json.load(f)
        if rles_d:
            person_masks = []
            for rle_info in rles_d["person_masks"]:
                person_mask = maskUtils.decode(rle_info)
                person_masks.append(person_mask)
            if len(person_masks) != 0:
                person_masks = np.array(person_masks).astype(int)
            d["person_mask_scores"] = rles_d.get("person_mask_scores", [])
            d["person_masks"] = person_masks
        else:
            d["person_mask_scores"] = []
            d["person_masks"] = []
    else:
        d["person_mask_scores"] = []
        d["person_masks"] = []

    prob_masks_fp = os.path.join(save_fp, "part_masks_prob.npz")
    if os.path.exists(prob_masks_fp):
        d["part_masks_probs"] = np.load(prob_masks_fp)["probs"].astype(float)/255.0

    return d


def remove_duplicate_vertices(points: np.ndarray) -> np.ndarray:
    return np.unique(points.view([('', points.dtype)] * points.shape[1])).view(points.dtype).reshape(-1, points.shape[1])


def filter_multi_person(fp):
    filter_keywords = ["two", "three", "DATA"]
    for k in filter_keywords:
        if k in fp:
            return False
    return True


def get_all_models_hdf5_mp(obj_folder_fps: List[str], hdf5_save_fp: str,
                        num_workers: int = 1):
    num_folders = len(obj_folder_fps)
    chunk_size = (num_folders // num_workers) + 1  # Ensure all folders are assigned
    obj_folder_batches = [obj_folder_fps[i:i + chunk_size] for i in range(0, num_folders, chunk_size)]

    procs = []
    for i in range(len(obj_folder_batches)):
        proc = mp.Process(target=get_all_models_hdf5, args=(obj_folder_batches[i], hdf5_save_fp, True))
        proc.start()
        procs.append(proc)

    for p in procs:
        p.join()


def reorient_meshes_to_hdf5(hdf5_fp: str, save_hdf5_fp: str) -> None:
    keys = hdf5_utils.get_hdf5_keys(hdf5_fp)
    keys = list(filter(lambda x: filter_multi_person(x), keys))
    keys.sort()
    texture_outer_fp = os.path.join(path_utils.get_parent(hdf5_fp), "textures")
    before_after_reorient_fp = os.path.join(path_utils.get_parent(hdf5_fp), "Before_After_Reorient")
    for model_num, key in tqdm.tqdm(enumerate(keys), total=len(keys)):
        print(f"Processing model {key}, number {model_num}")
        #if os.path.exists(save_hdf5_fp):
            #if hdf5_utils.is_key_in_hdf5(save_hdf5_fp, key):
            #    continue
        model_d = hdf5_utils.load_entry_by_key_nested(hdf5_fp, key)

        # Visualization before
        # vertices_before = model_d["vertices"]
        # pc_vis.display_pc_from_arr(vertices_before, flip_z=True)

        # reorient
        texture_fp = os.path.join(texture_outer_fp, key+".jpg")
        texture = np.asarray(Image.open(texture_fp))[..., :3].astype(np.uint8)
        model_d["texture"] = texture
        mesh_d, before_color_im, after_color_im = p_orient.iterative_auto_adjust_mesh(model_d)

        # pc_vis.display_pc_from_arr(reoriented_vertices)
        reoriented_vertices = mesh_d["vertices"]

        # Visualization After
        #pc_vis.display_pc_from_arr(reoriented_vertices, flip_z=True)


        concat_im = frame_utils.concatenate_images_horizontally([before_color_im, after_color_im])
        before_after_viz_save_fp = os.path.join(before_after_reorient_fp, key+".png")
        frame_utils.save_from_arr(concat_im, before_after_viz_save_fp)

        hdf5_utils.append_hdf5_dataset_from_d({key: reoriented_vertices.astype(np.float16)}, save_hdf5_fp,
                                              with_compression=True, with_overwrite=True)


def get_vertex_labels(save_fp: str, with_sapiens: bool = False,
                      is_sapiens_v2: bool = True):

    og_hdf5_save_path = hdf5_save_fp
    reoriented_save_path = os.path.join(path_utils.get_parent(og_hdf5_save_path), "reoriented_vertices.hdf5")
    keys = hdf5_utils.get_hdf5_keys(reoriented_save_path)
    keys.sort()
    # random.shuffle(keys)
    texture_outer_fp = os.path.join(path_utils.get_parent(og_hdf5_save_path), "textures")

    if with_sapiens:
        version_string = "" if not is_sapiens_v2 else "v2"
        part_vis_outer_fp = os.path.join(path_utils.get_parent(og_hdf5_save_path), "views_sapiens"+version_string)
        if is_sapiens_v2:
            save_fp = save_fp.replace(".hdf5", version_string+".hdf5")
        os.makedirs(part_vis_outer_fp, exist_ok=True)
    else:
        part_vis_outer_fp = os.path.join(path_utils.get_parent(og_hdf5_save_path), "views_chip")
        os.makedirs(part_vis_outer_fp, exist_ok=True)

    for key_num, key in tqdm.tqdm(enumerate(keys), total=len(keys)):
        print(f"At key {key_num} out of {len(keys)} keys named {key}")
        if os.path.exists(save_fp):
            if hdf5_utils.is_key_in_hdf5(save_fp, key): continue
        texture_fp = os.path.join(texture_outer_fp, key+".jpg")
        model_d = hdf5_utils.load_entry_by_key_nested(og_hdf5_save_path, key)
        vertices = hdf5_utils.load_entry_by_key(reoriented_save_path, key)
        #pc_vis.display_pc_from_arr(vertices, flip_z=True)

        texture = np.asarray(Image.open(texture_fp))[..., :3].astype(np.uint8)
        model_d["texture"] = texture
        model_d["vertices"] = vertices

        vertex_labels = seg_labels.mesh_triangle_visibility_and_seg(
            model_d, 2000, 2000, with_sapiens=with_sapiens,
            is_sapiens_v2=is_sapiens_v2).astype(int)

        hdf5_utils.append_hdf5_dataset_from_d({key: vertex_labels}, save_fp)

        # Visualization stuff
        if not with_sapiens:
            num_vals = 20
        else:
            if is_sapiens_v2:
                num_vals = 18
            else:
                num_vals = 15
        vertex_label_colors = frame_utils.get_color_map(num_vals=num_vals)
        part_colors = pc_vis.parts_to_colors(vertex_labels, vertex_label_colors)
        part_colors_norm = part_colors.astype(float)/255.0
        if not with_sapiens:
            label_names = seg_labels.cihp_label_names
        else:
            if is_sapiens_v2:
                label_names = seg_labels.new_sapiens_label_namesv2
            else:
                label_names = seg_labels.new_sapiens_label_namesv1
        label_d = {}
        for i, label_name in enumerate(label_names.items()):
            label_d[label_name] = vertex_label_colors[i].astype(float)/255.0

        #pc_vis.display_pc_from_arr(model_d["vertices"], colors=part_colors_norm)

        vis_d = {"points": model_d["vertices"], "colors": part_colors_norm}
        view = render_4view.render_pcd_4view(vis_d, width=2000, height=2000, show_window=False)
        view4_im = frame_utils.concatenate_images_horizontally([v for v in view])

        views_save_fp = os.path.join(part_vis_outer_fp, key+".jpg")
        im_with_legend = frame_utils.get_legend_with_im(view4_im, color_labels=label_d,
                                                        save_fp=views_save_fp,
                                                        legend_scale=3.0)

        # sanity check
        #frame_utils.show_im(im_with_legend)


def vis_reoriented_vertices():
    og_hdf5_save_path = hdf5_save_fp
    reoriented_save_path = os.path.join(path_utils.get_parent(og_hdf5_save_path), "reoriented_vertices.hdf5")
    keys = hdf5_utils.get_hdf5_keys(reoriented_save_path)
    keys.sort()
    for k in keys:
        vertices = hdf5_utils.load_entry_by_key(reoriented_save_path, k)
        pc_vis.display_pc_from_arr(vertices, flip_z=True)


def get_ifps_sampled_vertices(outer_vertices_fp: str, outer_labels_fp: str,
                              sampled_save_fp: str, K: int, max_points_per_window: int = 5000,
                              ignore_keys: List[str]= None, ignore_labels:Optional[List] = None,
                              with_overrite: bool = False) -> None:
    keys = hdf5_utils.get_hdf5_keys(outer_vertices_fp)
    if ignore_keys is not None:
        ignore_keys = set(ignore_keys)
        keys = [k for k in keys if k not in ignore_keys]
    for key_num, k in tqdm.tqdm(enumerate(keys), total=len(keys)):
        #print(f"Processing key {key_num}: {k} out of {len(keys)}...")
        vertices = hdf5_utils.load_entry_by_key(outer_vertices_fp, k)
        #vertices_c = vertices.copy()
        labels = hdf5_utils.load_entry_by_key(outer_labels_fp, k)
        vertices = torch.tensor(vertices).float().to(0)
        labels = torch.tensor(labels).to(0)

        if ignore_labels is not None:
            for il in ignore_labels:
                label_mask = labels != il
                vertices = vertices[label_mask]
                labels = labels[label_mask]

        vertices, code = pc_utils.sort_pcs_by_z_order(vertices, 0.001, to_gpu=True, return_code=True)
        vertices = vertices[0]
        labels = labels[code]

        N = vertices.shape[0]
        num_windows = (N + max_points_per_window - 1) // max_points_per_window

        padded_vertices_list = []
        padded_labels_list = []
        valid_mask_list = []

        window_K = (K//num_windows) + 1

        for w in range(num_windows):
            start = w * max_points_per_window
            end = min((w + 1) * max_points_per_window, N)

            v_win = vertices[start:end]
            l_win = labels[start:end]

            pad_len = max_points_per_window - v_win.shape[0]
            valid_mask = torch.ones(v_win.shape[0], dtype=torch.bool, device=v_win.device)

            if pad_len > 0:
                v_win = torch.cat([v_win, torch.zeros(pad_len, 3, device=v_win.device)], dim=0)
                l_win = torch.cat([l_win, torch.full((pad_len,), -1, dtype=l_win.dtype, device=l_win.device)], dim=0)
                valid_mask = torch.cat([valid_mask, torch.zeros(pad_len, dtype=torch.bool, device=v_win.device)])

            padded_vertices_list.append(v_win)
            padded_labels_list.append(l_win)
            valid_mask_list.append(valid_mask)

        # Combine all windows into a big padded batch
        all_vertices = torch.cat([torch.unsqueeze(t, dim=0) for t in padded_vertices_list], dim=0)
        all_labels = torch.cat([torch.unsqueeze(t, dim=0) for t in padded_labels_list], dim=0)
        all_valid_mask = torch.cat([torch.unsqueeze(t, dim=0) for t in valid_mask_list], dim=0)

        # Run FPS on entire (padded) point set
        idx = fps.farthest_point_sampling(all_vertices, window_K)
        idx = idx.long()

        # Only keep valid (non-padding) samples
        sampled_vertices = torch.gather(all_vertices, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 3))
        sampled_vertices = sampled_vertices.reshape(-1, 3)

        sampled_labels = torch.gather(all_labels, dim=1, index=idx)
        sampled_labels = sampled_labels.reshape(-1)

        sampled_valid_mask = torch.gather(all_valid_mask, dim=1, index=idx)
        sampled_valid_mask = sampled_valid_mask.reshape(-1)

        sampled_vertices = sampled_vertices[sampled_valid_mask].cpu().numpy().astype(np.float16)
        sampled_labels = sampled_labels[sampled_valid_mask].cpu().numpy().astype(np.uint8)

        sampled_vertices = sampled_vertices[:K]
        sampled_labels = sampled_labels[:K]

        # Visualization sanity check
        #pc_vis.display_pc_from_arr(sampled_vertices)
        #pc_vis.display_part_pc(sampled_vertices, sampled_labels)

        d = {k: {"sampled_vertices": sampled_vertices, "sampled_labels": sampled_labels}}
        hdf5_utils.append_hdf5_dataset_from_d_nested(d, sampled_save_fp)


def get_cihp_ifps_sampled_vertices(outer_vertices_fp: str, outer_labels_fp: str,
                              sampled_save_fp: str, K: int, max_points_per_window: int = 5000,
                                K_ratios: Tuple[int, int, int] = (0.9, 0.05, 0.05),
                                   ignore_labels:Optional[List]=None) -> None:
    left_arm_idx, right_arm_idx = 14, 15
    keys = hdf5_utils.get_hdf5_keys(outer_vertices_fp)
    for key_num, k in tqdm.tqdm(enumerate(keys), total=len(keys)):
        #print(f"Processing key {key_num}: {k} out of {len(keys)}...")
        vertices = hdf5_utils.load_entry_by_key(outer_vertices_fp, k)

        labels = hdf5_utils.load_entry_by_key(outer_labels_fp, k)
        vertices = torch.tensor(vertices).float().to(0)
        labels = torch.tensor(labels).to(0)

        if ignore_labels is not None:
            for il in ignore_labels:
                labels_mask = labels != il
                vertices = vertices[labels_mask]
                labels = labels[labels_mask]


        left_arm_mask = labels==left_arm_idx
        right_arm_mask = labels==right_arm_idx
        body_mask = torch.logical_and(~left_arm_mask, ~right_arm_mask)

        labels_left_arm, labels_right_arm, labels_body = labels[left_arm_mask], labels[right_arm_mask], labels[body_mask]
        vertices_left_arm, vertices_right_arm, vertices_body = vertices[left_arm_mask], vertices[right_arm_mask], vertices[body_mask]

        labels_all = []
        vertices_all = []

        num_left_arm, num_right_arm, num_body = (torch.sum(left_arm_mask).item(),
                                                 torch.sum(right_arm_mask).item(),
                                                 torch.sum(body_mask).item())
        num_left_arm_samples = int(min(K*K_ratios[1], num_left_arm))
        num_right_arm_samples = int(min(K*K_ratios[2], num_right_arm))
        num_body_samples_diff = K - (num_left_arm_samples+num_right_arm_samples)
        if num_body_samples_diff > num_body:
            print(f"Not enough body vertices for {num_body_samples_diff} samples... Skipping... ")
            continue

        k_vals = (num_body_samples_diff, num_left_arm_samples, num_right_arm_samples)
        for i, (vertices, labels) in enumerate(zip([vertices_body, vertices_left_arm, vertices_right_arm],
                                                 [labels_body, labels_left_arm, labels_right_arm])):

            if vertices.shape[0] == 0:
                continue

            vertices, code = pc_utils.sort_pcs_by_z_order(vertices, 0.001, to_gpu=True, return_code=True)
            vertices = vertices[0]
            labels = labels[code]

            N = vertices.shape[0]
            num_windows = (N + max_points_per_window - 1) // max_points_per_window

            padded_vertices_list = []
            padded_labels_list = []
            valid_mask_list = []

            k_val = k_vals[i]
            if k_val == 0:
                continue

            window_K = (k_val//num_windows) + 1

            for w in range(num_windows):
                start = w * max_points_per_window
                end = min((w + 1) * max_points_per_window, N)

                v_win = vertices[start:end]
                l_win = labels[start:end]

                pad_len = max_points_per_window - v_win.shape[0]
                valid_mask = torch.ones(v_win.shape[0], dtype=torch.bool, device=v_win.device)

                if pad_len > 0:
                    v_win = torch.cat([v_win, torch.zeros(pad_len, 3, device=v_win.device)], dim=0)
                    l_win = torch.cat([l_win, torch.full((pad_len,), -1, dtype=l_win.dtype, device=l_win.device)], dim=0)
                    valid_mask = torch.cat([valid_mask, torch.zeros(pad_len, dtype=torch.bool, device=v_win.device)])

                padded_vertices_list.append(v_win)
                padded_labels_list.append(l_win)
                valid_mask_list.append(valid_mask)

            # Combine all windows into a big padded batch
            all_vertices = torch.cat([torch.unsqueeze(t, dim=0) for t in padded_vertices_list], dim=0)
            all_labels = torch.cat([torch.unsqueeze(t, dim=0) for t in padded_labels_list], dim=0)
            all_valid_mask = torch.cat([torch.unsqueeze(t, dim=0) for t in valid_mask_list], dim=0)

            # Run FPS on entire (padded) point set
            idx = fps.farthest_point_sampling(all_vertices, window_K)
            idx = idx.long()

            # Only keep valid (non-padding) samples
            sampled_vertices = torch.gather(all_vertices, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 3))
            sampled_vertices = sampled_vertices.reshape(-1, 3)

            sampled_labels = torch.gather(all_labels, dim=1, index=idx)
            sampled_labels = sampled_labels.reshape(-1)

            sampled_valid_mask = torch.gather(all_valid_mask, dim=1, index=idx)
            sampled_valid_mask = sampled_valid_mask.reshape(-1)

            sampled_vertices = sampled_vertices[sampled_valid_mask].cpu().numpy().astype(np.float16)
            sampled_labels = sampled_labels[sampled_valid_mask].cpu().numpy().astype(np.uint8)

            sampled_vertices = sampled_vertices[:K]
            sampled_labels = sampled_labels[:K]

            labels_all.append(sampled_labels)
            vertices_all.append(sampled_vertices)

        sampled_vertices = [x for x in vertices_all if isinstance(x, np.ndarray)]
        sampled_labels = [x for x in labels_all if isinstance(x, np.ndarray)]

        sampled_vertices = np.concatenate(sampled_vertices, axis=0)
        sampled_labels = np.concatenate(sampled_labels, axis=0)

        sampled_vertices = sampled_vertices[:K]
        sampled_labels = sampled_labels[:K]

        # Visualization sanity check
        assert sampled_vertices.shape[0] == K, f"vertex sampling failed for key {k}"
        assert sampled_labels.shape[0] == K, f"label sampling failed for key {k}"
        d = {k: {"sampled_vertices": sampled_vertices, "sampled_labels": sampled_labels}}
        hdf5_utils.append_hdf5_dataset_from_d_nested(d, sampled_save_fp)


def get_sapiens_ifps_sampled_vertices(outer_vertices_fp: str, outer_labels_fp: str,
                              sampled_save_fp: str, K1: int, K2:int, max_points_per_window: int = 5000,
                                K_ratios: Tuple[float] = (0.05, 0.05, 0.05, 0.05, 0.8),
                                ignore_keys: List[str] = None, is_sapiens_v2:bool = False) -> None:
    keys = hdf5_utils.get_hdf5_keys(outer_vertices_fp)
    if ignore_keys is not None:
        ignore_keys = set(ignore_keys)
        keys = [k for k in keys if k not in ignore_keys]

    if not is_sapiens_v2:
        left_hand_idx, left_arm_idx, right_hand_idx, right_arm_idx = 5, 6, 10, 11
    else:
        left_hand_idx, left_arm_idx, right_hand_idx = 5, 6, 10
        right_arm_idx, lip_idx, teeth_idx, tongue_idx = 11, 15, 16, 17

    for key_num, k in tqdm.tqdm(enumerate(keys), total=len(keys)):
        vertices = hdf5_utils.load_entry_by_key(outer_vertices_fp, k)
        labels = hdf5_utils.load_entry_by_key(outer_labels_fp, k)
        vertices = torch.tensor(vertices).float().to(0)
        labels = torch.tensor(labels).to(0)

        masks = {
            "left_hand": labels == left_hand_idx,
            "right_hand": labels == right_hand_idx,
            "left_arm": labels == left_arm_idx,
            "right_arm": labels == right_arm_idx,
        }

        if is_sapiens_v2:
            masks.update({
                "lip": labels == lip_idx,
                "teeth": labels == teeth_idx,
                "tongue": labels == tongue_idx
            })

        combined_mask = torch.zeros_like(labels, dtype=torch.bool)
        for m in masks.values():
            combined_mask |= m
        body_mask = ~combined_mask

        part_names = list(masks.keys())
        vertices_parts = [vertices[masks[n]] for n in part_names] + [vertices[body_mask]]
        labels_parts = [labels[masks[n]] for n in part_names] + [labels[body_mask]]
        counts_parts = [torch.sum(masks[n]).item() for n in part_names] + [torch.sum(body_mask).item()]

        num_samples_parts = [int(min(counts_parts[i], K1 * K_ratios[i])) for i in range(len(K_ratios))]

        labels_all = []
        vertices_all = []

        for i, (v_part, l_part, k_val) in enumerate(zip(vertices_parts, labels_parts, num_samples_parts)):
            if v_part.shape[0] == 0 or k_val == 0:
                continue

            v_part, code = pc_utils.sort_pcs_by_z_order(v_part, 0.001, to_gpu=True, return_code=True)
            v_part = v_part[0]
            l_part = l_part[code]

            N = v_part.shape[0]
            num_windows = (N + max_points_per_window - 1) // max_points_per_window
            window_K = (k_val // num_windows) + 1

            padded_vertices_list = []
            padded_labels_list = []
            valid_mask_list = []

            for w in range(num_windows):
                start = w * max_points_per_window
                end = min((w + 1) * max_points_per_window, N)

                v_win = v_part[start:end]
                l_win = l_part[start:end]

                pad_len = max_points_per_window - v_win.shape[0]
                valid_mask = torch.ones(v_win.shape[0], dtype=torch.bool, device=v_win.device)

                if pad_len > 0:
                    v_win = torch.cat([v_win, torch.zeros(pad_len, 3, device=v_win.device)], dim=0)
                    l_win = torch.cat([l_win, torch.full((pad_len,), -1, dtype=l_win.dtype, device=l_win.device)], dim=0)
                    valid_mask = torch.cat([valid_mask, torch.zeros(pad_len, dtype=torch.bool, device=v_win.device)])

                padded_vertices_list.append(v_win)
                padded_labels_list.append(l_win)
                valid_mask_list.append(valid_mask)

            all_vertices = torch.stack(padded_vertices_list)
            all_labels = torch.stack(padded_labels_list)
            all_valid_mask = torch.stack(valid_mask_list)

            idx = fps.farthest_point_sampling(all_vertices, window_K).long()
            sampled_vertices = torch.gather(all_vertices, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).reshape(-1, 3)
            sampled_labels = torch.gather(all_labels, 1, idx).reshape(-1)
            sampled_valid_mask = torch.gather(all_valid_mask, 1, idx).reshape(-1)

            sampled_vertices = sampled_vertices[sampled_valid_mask].cpu().numpy().astype(np.float16)
            sampled_labels = sampled_labels[sampled_valid_mask].cpu().numpy().astype(np.uint8)

            labels_all.append(sampled_labels)
            vertices_all.append(sampled_vertices)

        sampled_vertices = np.concatenate([x for x in vertices_all if isinstance(x, np.ndarray)], axis=0)
        sampled_labels = np.concatenate([x for x in labels_all if isinstance(x, np.ndarray)], axis=0)

        # Visualization sanity check
        #pc_vis.display_pc_from_arr(sampled_vertices)
        #pc_vis.display_part_pc(sampled_vertices, sampled_labels)

        sampled_vertices = sampled_vertices[:K2]
        sampled_labels = sampled_labels[:K2]
        assert sampled_vertices.shape[0] == K2, f"insufficient samples: {sampled_vertices.shape[0]}"
        assert sampled_labels.shape[0] == K2, f"Insufficient samples: {sampled_labels.shape[0]}"

        d = {k: {"sampled_vertices": sampled_vertices, "sampled_labels": sampled_labels}}
        hdf5_utils.append_hdf5_dataset_from_d_nested(d, sampled_save_fp)


def get_bad_sapiens_keys():
    fp = "../Data/PC_Data/bad_samples_sapiens.json"
    with open(fp, "r") as f:
        d = json.load(f)
    keys = list(d.keys())
    keys = [k.replace(".jpg", "") for k in keys]
    return keys


if __name__ == "__main__":

    # import h5py
    # if not os.path.exists(hdf5_save_fp):
    #     with h5py.File(hdf5_save_fp, 'w') as f:
    #         pass  

    obj_fps = get_outer_obj_folder_fps()
    print("Found obj folders:", obj_fps["obj_absolute_paths"])
    print("Number of folders found:", len(obj_fps["obj_absolute_paths"]))

    # Reorientation
    get_all_models_hdf5_mp(get_outer_obj_folder_fps()["obj_absolute_paths"], hdf5_save_fp)
    og_hdf5_save_path = hdf5_save_fp
    reoriented_save_path = os.path.join(path_utils.get_parent(og_hdf5_save_path), "reoriented_vertices.hdf5")
    reorient_meshes_to_hdf5(og_hdf5_save_path, reoriented_save_path)

    #vis_reoriented_vertices()

    # cihp labelling
    #cihp_labels_save_fp = "../Data/PC_Data/cihp_vertex_labels.hdf5"
    # get_vertex_labels(cihp_labels_save_fp)
    # sapiens labeling
    #sapiens_labels_save_fp = os.path.join(data_outer_save_fp, "sapiens_vertex_labels.hdf5")
    #get_vertex_labels(sapiens_labels_save_fp, with_sapiens=True)
    #get_vertex_labels(sapiens_labels_save_fp, with_sapiens=True, is_sapiens_v2=False)

    # cihp vertex sampling
    # og_hdf5_save_path = hdf5_save_fp
    # reoriented_save_path = os.path.join(path_utils.get_parent(og_hdf5_save_path), "reoriented_vertices.hdf5")
    #keys = hdf5_utils.get_hdf5_keys(reoriented_save_path)
    #get_ifps_sampled_vertices(reoriented_save_path, cihp_labels_save_fp,
    #                          "../Data/PC_Data/cihp_sampled_pcs_data.hdf5",
    #                          K=10000, ignore_labels=cihp_labels_to_ignore)
    #get_cihp_ifps_sampled_vertices(reoriented_save_path, cihp_labels_save_fp,
    #                          "../Data/PC_Data/cihp_part_sampled_pcs_data.hdf5",
    #                          K=10000, ignore_labels=cihp_labels_to_ignore)

    # sapiens vertex sampling v1
    """
    bad_sapiens_keys = get_bad_sapiens_keys()
    get_ifps_sampled_vertices(reoriented_save_path, sapiens_labels_save_fp,
                              "../Data/PC_Data/sapiens_sampled_pcs_data.hdf5",
                             K=10000, ignore_keys=bad_sapiens_keys)
    get_sapiens_ifps_sampled_vertices(reoriented_save_path, sapiens_labels_save_fp,
                              "../Data/PC_Data/sapiens_part_sampled_pcs_data.hdf5",
                              K=10000, ignore_keys=bad_sapiens_keys)
    """

    # bad_sapiens_keys = get_bad_sapiens_keys()
    sapiens_labels_v2_save_fp = os.path.join(data_outer_save_fp, "sapiens_vertex_labelsv2.hdf5")
    #get_ifps_sampled_vertices(reoriented_save_path, sapiens_labels_v2_save_fp,
    #                          os.path.join("../Data/PC_Data", "sapiensv2_sampled_pcs_data.hdf5"),
    #                          K=10000, ignore_keys=bad_sapiens_keys)
    # get_sapiens_ifps_sampled_vertices(reoriented_save_path, sapiens_labels_v2_save_fp,
    #                                   os.path.join("../Data/PC_Data", "sapiensv2_part_sampled_pcs_data.hdf5"),
    #                                   K1=20000, K2=10000,
    #                                   K_ratios=(0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.70), is_sapiens_v2=True)

    ...
