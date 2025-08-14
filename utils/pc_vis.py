from typing import Optional, List, Union, Tuple
import os
import math

import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import trimesh
from trimesh import Scene
import io
import time

import utils.pc_utils as pc_utils
from utils.pc_utils import get_seq_info
import utils.frame_utils as frame_vis_utils


def get_jetmap_colors_from_pc(pc: np.ndarray, with_normalization: bool = False):
    z_vals = pc[:, -1]
    if with_normalization:
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        if z_min != z_max:
            z_vals_normalized = (z_vals-z_min)/(z_max-z_min)
        else:
            z_vals_normalized = z_vals.copy()
        colors = plt.cm.jet(z_vals_normalized)[:, :3]*255.0
    else:
        colors = plt.cm.jet(z_vals)[:, :3]*255.0
    colors = colors.astype(np.uint8)
    return colors

def get_part_color_map(num_vals: int) -> dict:
    keys = list(range(-1, num_vals+1))
    color_d = {}
    biases, gain, power = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, power)) % 255.0
        g = int(math.pow((key+biases[1])*gain, power)) % 255.0
        b = int(math.pow((key+biases[2])*gain, power)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d


def render_pc_4_views(pc: np.ndarray, colors: Optional[np.ndarray],
                      outer_save_fp: str, view_as_spheres: bool = False) -> None:
    pc_ = pc.copy()
    rot_matr = pc_utils.get_y_rotation(np.pi/2)
    for i in range(4):
        if i != 0:
            pc_ = (rot_matr @ pc_.T).T
        view_save_fp = os.path.join(outer_save_fp, "view_"+str(i)+".png")
        render_pc(pc_, colors, view_save_fp, view_as_spheres)


def parts_to_colors(parts: np.ndarray, color_map: dict = None) -> np.ndarray:
    part_colors = np.zeros((parts.shape[0], 3))
    unique_parts = np.unique(parts)
    if color_map is None:
        color_map = get_part_color_map(np.max(unique_parts)+1)
    for unique_part in unique_parts:
        part_colors[parts == unique_part] = color_map[unique_part]
    return part_colors

def parts_to_part_colors_pc_vid(parts: List[List[np.ndarray]], color_map: Optional[dict] = None) -> List[List[np.ndarray]] :
    part_colors = []
    num_frames = len(parts)
    if num_frames == 0:
        print("All empty frames, returning the empty list.")
        return []

    num_people = len(parts[0])
    if num_people == 0:
        print("No people, returning the empty list.")
        return []

    if color_map is None:
        color_map = get_part_color_map(num_vals=100) # TODO should loop over and get max part, but for now this is fine

    for frame_num in range(num_frames):
        inner_part_colors = []
        for person_num in range(num_people):
            part_frame_person = parts[frame_num][person_num]
            if isinstance(part_frame_person, np.ndarray):
                inner_part_colors.append(parts_to_colors(part_frame_person, color_map))
            else:
                inner_part_colors.append([])
        part_colors.append(inner_part_colors)

    return part_colors

def render_pc(pc: np.ndarray, colors: Optional[np.ndarray],
              save_fp: Optional[str], view_as_spheres: bool = False) -> np.ndarray:
    xmin, xmax = np.min(pc[:, 0]), np.max(pc[:, 0])
    ymin, ymax = np.min(pc[:, 1]), np.max(pc[:, 1])
    zmin, zmax = np.min(pc[:, 2]), np.max(pc[:, 2])

    ave_x, ave_y, ave_z = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0

    # Camera look direction, center at the average values
    center = np.array([ave_x, ave_y, ave_z])

    # Camera position
    y_padding = 0.2
    camera_distance = -1.0
    eye = [center[0], center[1] - y_padding, zmin + camera_distance]

    # The up direction
    up = [0.0, 1.0, 0.0]
    render_obj = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    render_obj.scene.set_background([0.3, 0.3, 0.3, 1])
    render_obj.scene.set_lighting(render_obj.scene.LightingProfile.SOFT_SHADOWS, (0.577, -0.577, -0.577))

    render_obj.scene.camera.look_at(center=center, eye=eye, up=up)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5.0

    if not view_as_spheres:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)
        point_cloud.estimate_normals()
        if colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(colors/255.0)  # assumes colors are not already normalized
        render_obj.scene.add_geometry("points", point_cloud, mat)
    else:
        spheres = o3d.geometry.TriangleMesh()
        radius = 0.01
        for i, point in enumerate(pc):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(point)
            if colors is not None:
                sphere.paint_uniform_color(colors[i]/255.0)
            spheres += sphere
        render_obj.scene.add_geometry("points_", spheres, mat)

    img_o3d = render_obj.render_to_image()
    img = np.array(img_o3d)
    if save_fp is not None:
        img_pil = Image.fromarray(img)
        img_pil.save(save_fp)
    return img


def display_pc_from_arr(pc: np.ndarray, with_normals: bool = False,
                        colors: Optional[np.ndarray] = None, with_default_view: bool = True,
                        flip_z: bool = False):

    if flip_z:
        pc = pc.copy()
        pc[..., -1] *= -1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if with_normals:
        pcd.estimate_normals()
    if colors is not None:
        colors_ = colors.copy().astype(float)
        if colors_.ndim == 1: # grayscale to color
            colors_ = np.expand_dims(colors_, axis=-1)
        if colors_.shape[-1] == 1:
            colors_ = np.tile(colors_, (1, 3))
        if np.max(colors_) > 1.0:
            colors_ /= 255.0
        pcd.colors =  o3d.utility.Vector3dVector(colors_)
    if with_default_view:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_front([0, 0, -1])
        view_control.set_zoom(0.8)
        vis.run()
        vis.destroy_window()
    else:
     o3d.visualization.draw_geometries([pcd])


def display_part_pc(pc: np.ndarray, parts: np.ndarray,
                    with_normals: bool = False, with_default_view: bool = True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    colors = parts_to_colors(parts, None)
    colors /= 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if with_normals:
        pcd.estimate_normals()
    if with_default_view:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_front([0, 0, -1])
        view_control.set_zoom(0.8)
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd])


def display_mesh(vertices: np.ndarray, faces: np.ndarray, with_wireframe: bool = False) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([mesh], window_name="Mesh Viewer", mesh_show_wireframe=with_wireframe)


def pcd_to_spheres(pcd: Union[np.ndarray, o3d.geometry.PointCloud], sphere_radius:float=0.1):
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points).copy()
        colors = np.asarray(pcd.colors).copy()
        if colors.size == 0:
            colors = get_jetmap_colors_from_pc(points, with_normalization=True)/255.0
    else:
        points = pcd.copy()
        colors = get_jetmap_colors_from_pc(points, with_normalization=True)/255.0
    spheres = []
    for i, point in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        if colors is not None:
            sphere.paint_uniform_color(colors[i])
        spheres.append(sphere)
    return spheres


def print_camera_parameters(params: o3d.camera.PinholeCameraParameters):
    print("\n--- Intrinsic Parameters ---")
    print("Width:", params.intrinsic.width)
    print("Height:", params.intrinsic.height)
    print("fx:", params.intrinsic.get_focal_length()[0])
    print("fy:", params.intrinsic.get_focal_length()[1])
    print("cx:", params.intrinsic.get_principal_point()[0])
    print("cy:", params.intrinsic.get_principal_point()[1])
    print("Intrinsic Matrix:\n", params.intrinsic.intrinsic_matrix)

    print("\n--- Extrinsic Parameters ---")
    print("Extrinsic Matrix:\n", params.extrinsic)


def print_camera_parameters_from_visualizer(vis: o3d.visualization.Visualizer):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    print_camera_parameters(params)



def change_ctr_z_val(ctr, new_z_val:float):
    params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic_copy = params.extrinsic.copy()  # Make a copy
    extrinsic_copy[3, 2] = new_z_val
    params.extrinsic = extrinsic_copy  # Assign the modified copy back
    ctr.convert_from_pinhole_camera_parameters(params)


def vis_pc(pc_: np.ndarray, save_fp: Optional[str]=None, with_normals=False,
           colors: Optional[np.ndarray] = None, with_z_flip: bool = False,
           is_headless: bool = False, to_spheres: bool = True,
           with_postprocess: bool = True, sphere_radius: Optional[float] = None,
           remove_zero: bool = False, with_display: bool = False) -> np.ndarray:
    to_delete = False
    if save_fp is None:
        to_delete = True
        save_fp = "./temp.png"

    render_resolution = (720, 1280) # height by width
    resize_resolution = (600, 900)  # height by width

    if remove_zero:
        pc_ = pc_utils.remove_zero_points(pc_)
        if pc_.size == 0:
            if with_postprocess:
                empty_im = np.zeros((resize_resolution[0], resize_resolution[1], 3), dtype=np.uint8)
            else:
                empty_im = np.zeros((render_resolution[0], render_resolution[1], 3), dtype=np.uint8)
            #empty_im.fill(255)
            return empty_im

    if np.sum(pc_.reshape(-1)) == 0:
        if with_postprocess:
            empty_im = np.zeros((resize_resolution[0], resize_resolution[1], 3), dtype=np.uint8)
        else:
            empty_im = np.zeros((render_resolution[0], render_resolution[1], 3), dtype=np.uint8)
        #empty_im.fill(255)
        return empty_im

    pc = pc_.copy()
    pcd = o3d.geometry.PointCloud()
    if with_z_flip:
        pc[:, -1] *= -1
    centroid = np.mean(pc, axis=0)
    pcd.points = o3d.utility.Vector3dVector(pc)
    if with_normals:
        pcd.estimate_normals()
    if colors is not None:
        colors_ = colors.copy().astype(float)
        if np.max(colors_) > 1.0:
            colors_ /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_)
    else:
        pcd.colors = o3d.utility.Vector3dVector(get_jetmap_colors_from_pc(pc, with_normalization=True))

    if to_spheres:
        if sphere_radius is None:
            min_distance = pc_utils.smallest_distance(pc)
            sphere_radius = max(min_distance/2.5, 0.00001)
        geom = pcd_to_spheres(pcd, sphere_radius=sphere_radius)
    else:
        geom = [pcd]
    if not is_headless:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, left=0, top=0, height=render_resolution[0], width=render_resolution[1])
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        for g in geom:
            vis.add_geometry(g)
        ctr = vis.get_view_control()
        ctr.set_lookat(centroid.tolist())
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        vis.poll_events()
        vis.update_renderer()
        if with_display:
            vis.run()
        vis.capture_screen_image(save_fp)
        vis.destroy_window()
    else:
        render_obj = o3d.visualization.rendering.OffscreenRenderer(render_resolution[1], resize_resolution[0])
        render_obj.scene.set_background([0.0, 0.0, 0.0, 1])
        render_obj.scene.camera.look_at(center=centroid.tolist(), eye=[0.5, 0.5, -1.5], up=[0, 1, 0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        for i, g in enumerate(geom):
            render_obj.scene.add_geometry("geom_"+str(i), g, mat)
        img_o3d = render_obj.render_to_image()
        img = np.array(img_o3d)
        frame_vis_utils.save_from_arr(img, save_fp)
        """
        if frame_vis_utils.is_value_image(img, 255) or frame_vis_utils.is_value_image(img, 0):
            bounds = pc_utils.get_pc_bounds(pc)
            print(f"Empty image. Bounds are {bounds}")
            exit(-1)
        """
        if with_display:
            frame_vis_utils.show_im(save_fp)
    im_arr = frame_vis_utils.im_fp_to_arr(save_fp)
    if with_postprocess:
        #im_arr = frame_vis_utils.min_max_nonvalue_crop(im_arr, save_fp=None, value=255, padding_px=50, padding_py=50)
        im_arr = frame_vis_utils.resize_im_arr(im_arr, (resize_resolution[0], resize_resolution[1]))
    if to_delete:
        os.remove(save_fp)
    return im_arr

def create_grid(x_range, y_range, z_range, step: Union[float, Tuple[float, float, float]]):
    lines = []
    points = []

    if isinstance(step, int):
        step = (step, step, step)

    # Generate grid lines along x-axis
    for y in np.arange(y_range[0], y_range[1], step[1]):
        for z in np.arange(z_range[0], z_range[1], step[2]):
            points.append([x_range[0], y, z])
            points.append([x_range[1], y, z])
            lines.append([len(points) - 2, len(points) - 1])

    # Generate grid lines along y-axis
    for x in np.arange(x_range[0], x_range[1], step[0]):
        for z in np.arange(z_range[0], z_range[1], step[2]):
            points.append([x, y_range[0], z])
            points.append([x, y_range[1], z])
            lines.append([len(points) - 2, len(points) - 1])

    # Generate grid lines along z-axis
    for x in np.arange(x_range[0], x_range[1], step[0]):
        for y in np.arange(y_range[0], y_range[1], step[1]):
            points.append([x, y, z_range[0]])
            points.append([x, y, z_range[1]])
            lines.append([len(points) - 2, len(points) - 1])

    # Create LineSet from the grid lines
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)

    return grid


def get_voxel_grid(pc: np.ndarray, colors: Optional[np.ndarray] = None, grid_size: Optional[float] = 0.1,
                       voxel_grid_size: float = 0.02, with_display: bool = False, save_fp: Optional[str] = None,
                   grid_buckets: Optional[Union[int, Tuple[int, int, int]]] = (10, 10, 10)):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    if colors is not None:
        colors_ = colors.copy()
    else:  # jetmap default
        colors_ = get_jetmap_colors_from_pc(pc)

    if np.max(colors_) > 1:
        colors_ = colors_ / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors_)

    # Create the voxel grid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_grid_size)

    x_min, x_max = np.min(pc[:, 0]), np.max(pc[:, 0])
    y_min, y_max = np.min(pc[:, 1]), np.max(pc[:, 1])
    z_min, z_max = np.min(pc[:, 2]), np.max(pc[:, 2])

    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)
    elif grid_size is None:
        if isinstance(grid_buckets, int):
            grid_buckets = (grid_buckets, grid_buckets, grid_buckets)
        x_grid_size = (x_max - x_min) / grid_buckets[0]
        y_grid_size = (y_max - y_min) / grid_buckets[1]
        z_grid_size = (z_max - z_min) / grid_buckets[2]
        grid_size = (x_grid_size, y_grid_size, z_grid_size)


    x_min, x_max = x_min-grid_size[0], x_max+grid_size[0]
    y_min, y_max = y_min-grid_size[1], y_max+grid_size[1]
    z_min, z_max = z_min-grid_size[2], z_max+grid_size[2]
    grid = create_grid((x_min, x_max), (y_min, y_max), (z_min, z_max), step=grid_size)

    geometries = [voxel_grid, grid]
    if with_display:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for g in geometries:
            vis.add_geometry(g)
        vis.run()
        if save_fp is not None:
            vis.capture_screen_image(save_fp)

    return geometries


def get_grid_centroid(grid):
    grid_points = np.asarray(grid.points)
    min_x, max_x, = np.min(grid_points[:, 0]), np.max(grid_points[:, 0])
    min_y, max_y, = np.min(grid_points[:, 1]), np.max(grid_points[:, 1])
    min_z, max_z, = np.min(grid_points[:, 2]), np.max(grid_points[:, 2])
    mid_point = np.array([(max_x+min_x)/2, (max_y+min_y)/2, (max_z+min_z)/2])
    return mid_point


def get_spheres_between_points(point1: np.ndarray, point2: np.ndarray, num_spheres=3, radius=0.1):
    direction = point2 - point1
    sphere_positions = [point1 + (i / (num_spheres - 1)) * direction for i in range(num_spheres)]
    spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(pos) for pos in sphere_positions]
    return spheres


def vis_pc_trimesh(pc: np.ndarray, save_fp: Optional[str] = None):
    pc_tm = trimesh.PointCloud(pc)
    pc_tm.colors = pc
    scene = Scene()
    scene.add_geometry(pc_tm)
    png_bytes = scene.save_image(resolution=(600, 800))
    im_pil = Image.open(io.BytesIO(png_bytes))
    im_arr = np.asarray(im_pil)
    if save_fp is not None:
        frame_vis_utils.save_from_arr(im_arr, save_fp)
