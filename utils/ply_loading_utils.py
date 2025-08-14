import open3d as o3d
import numpy as np

def load_ply_o3d(ply_fp: str, with_colors: bool, with_normals: bool):
    pcd = o3d.io.read_point_cloud(ply_fp)
    points = np.asarray(pcd.points)

    has_colors = np.asarray(pcd.colors).size > 0
    has_normals = np.asarray(pcd.normals).size > 0

    colors = np.asarray(pcd.colors) if with_colors and has_colors else None
    normals = np.asarray(pcd.normals) if with_normals and has_normals else None

    if with_normals and with_colors:
        return points, colors, normals

    elif with_normals and not with_colors:
        return points, normals

    elif not with_normals and with_colors:
        return points, colors

    else:
        return points
