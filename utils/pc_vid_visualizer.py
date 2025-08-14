from typing import Optional, List
import math

import os
os.environ['PYFREETYPE_QUIET'] = '1'

import faulthandler
faulthandler.disable()
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import numpy as np
import trimesh
from trimesh import Scene
from PIL import Image
import io
import cv2
import matplotlib as plt

import Vid_Utils.vid_utils as vid_utils


def get_part_color_map(num_vals) -> dict:
    global part_d_num_to_str
    keys = list(range(-1, num_vals))
    color_d = {}
    biases, gain, power = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, power)) % 255.0
        g = int(math.pow((key+biases[1])*gain, power)) % 255.0
        b = int(math.pow((key+biases[2])*gain, power)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d


def is_empty_list_of_list(l):
    if len(l) == 0:
        return True
    if len(l[0]) == 0:
        return True
    return False


def get_perspective_camera_distance(min_z, max_z, fov_degrees=30, padding=0.25):
    range = max_z-min_z
    camera_distance = range / (2*math.tan(math.radians(fov_degrees / 2)))
    return -(camera_distance+padding)


def visualize_pc_vid(pcs: List[List[np.ndarray]], colors: Optional[List[List[np.ndarray]]],
                     save_fp: str, with_voxelization: bool, camera_distance: Optional[float] = None):
    num_frames = len(pcs)
    if num_frames == 0:
        print(f"No frames, returning.")
        return

    num_people = len(pcs[0])
    if num_people == 0:
        print(f"No people, returning.")
        return

    render_obj = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    render_obj.scene.set_background([0.3, 0.3, 0.3, 1])
    render_obj.scene.set_lighting(render_obj.scene.LightingProfile.SOFT_SHADOWS, (0.577, -0.577, -0.577))

    min_x_val, min_y_val, min_z_val = float("inf"), float("inf"), float("inf")
    max_x_val, max_y_val, max_z_val = -1 * float("inf"), -1 * float("inf"), -1 * float("inf")

    for frame_num in range(num_frames):
        frame_pc = pcs[frame_num]
        if not is_empty_list_of_list(frame_pc):
            for person_num in range(num_people):
                if isinstance(frame_pc[person_num], np.ndarray):
                    min_x_val = min(min_x_val, np.min(frame_pc[person_num][:, 0]))
                    max_x_val = max(max_x_val, np.max(frame_pc[person_num][:, 0]))

                    min_y_val = min(min_y_val, np.min(frame_pc[person_num][:, 1]))
                    max_y_val = max(max_y_val, np.max(frame_pc[person_num][:, 1]))

                    min_z_val = min(min_z_val, np.min(frame_pc[person_num][:, 2]))
                    max_z_val = max(max_z_val, np.max(frame_pc[person_num][:, 2]))

    ave_x, ave_y, ave_z = (min_x_val + max_x_val) / 2.0, (min_y_val + max_y_val) / 2.0, (min_z_val + max_z_val) / 2.0

    # Camera look direction, center at the median values
    center = np.array([ave_x, ave_y, ave_z])

    # Camera position
    y_padding = 0.1
    # camera_distance = get_perspective_camera_distance(min_z_val, max_z_val)
    if camera_distance is None:
        camera_distance = -1.2
    eye = [center[0], center[1]-y_padding, min_z_val + camera_distance]

    # The up direction
    up = [0.0, 1.0, 0.0]

    render_obj.scene.camera.look_at(center=center, eye=eye, up=up)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = [0.8, 0.2, 0.2, 1.0]
    mat.shader = "default"
    mat.point_size = 2.0

    rendered_ims = []
    for frame_num in range(num_frames):
        img = None
        frame_pc = pcs[frame_num]
        if not is_empty_list_of_list(frame_pc):
            for person_num in range(num_people):
                person_points = frame_pc[person_num]
                if isinstance(person_points, np.ndarray):
                    if colors is not None:
                        part_colors_f = colors[frame_num][person_num]
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(person_points)
                    point_cloud.estimate_normals()
                    if colors is not None:
                        point_cloud.colors = o3d.utility.Vector3dVector(part_colors_f)
                    if with_voxelization:
                        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.02)
                        render_obj.scene.add_geometry("voxels_" + str(person_num), voxels, mat)
                    else:
                        render_obj.scene.add_geometry("points_" + str(person_num), point_cloud, mat)
            img_o3d = render_obj.render_to_image()
            img = np.array(img_o3d)

        render_obj.scene.clear_geometry()
        if img is not None:
            rendered_ims.append(img)

    rendered_ims = np.array(rendered_ims, dtype=np.uint8)
    vid_utils.write_vid_from_frames(rendered_ims, save_fp)


def z_val_to_blue(z_vals: np.ndarray) -> np.ndarray:
    color_vals = np.zeros((z_vals.shape[0], 3), dtype=np.uint8)
    min_, max_ = np.min(z_vals), np.max(z_vals)
    z_vals_norm = (z_vals-min_)/(max_-min_)*255.0
    z_vals_norm = z_vals_norm.astype(np.uint8)
    color_vals[:, -1] += z_vals_norm
    return color_vals


def z_val_to_jetmap(z_vals: np.ndarray) -> np.ndarray:
    num_vals = z_vals.shape[0]
    z_vals = z_vals.reshape(num_vals, 1)
    min_ = np.min(z_vals)
    z_vals = (z_vals-min_)/(np.max(z_vals)-min_)
    jet_image = plt.cm.jet(z_vals)
    jet_image = (jet_image[:, :, :3] * 255).astype(np.uint8)[:, 0, :]
    return jet_image


def visualize_pc_vid_trimesh(pcs: List[List[np.ndarray]], colors: Optional[List[List[np.ndarray]]]=None,
                             save_fp: str = None) -> np.ndarray:
    num_frames = len(pcs)
    if num_frames == 0:
        print(f"No frames, returning.")
        return

    num_people = len(pcs[0])
    if num_people == 0:
        print(f"No people, returning.")
        return

    rendered_ims = []
    scene = Scene()

    for frame_num in range(num_frames):
        for person_num in range(num_people):
            pc = pcs[frame_num][person_num].copy()
            if isinstance(pc, np.ndarray):
                pc[:, 0] *= -1
                pc_tm = trimesh.PointCloud(pc)
                if colors is not None:
                    pc_tm.colors = colors[frame_num][person_num]
                else:
                    pc_tm.colors = z_val_to_jetmap(pc[:, -1])
                scene.add_geometry(pc_tm)
            else:
                ...
        png_bytes = scene.save_image(resolution=(600, 800))
        im_pil = Image.open(io.BytesIO(png_bytes))
        im_arr = np.asarray(im_pil)[:, :, :3]
        rendered_ims.append(im_arr)

        # clear scene
        all_names = list(scene.geometry.keys())
        scene.delete_geometry(all_names)

    rendered_ims = np.array(rendered_ims, dtype=np.uint8)
    if save_fp:
        vid_utils.write_vid_from_frames(rendered_ims, save_fp)
    return rendered_ims