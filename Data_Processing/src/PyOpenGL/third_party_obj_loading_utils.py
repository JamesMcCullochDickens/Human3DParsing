from typing import *

import pyassimp
import numpy as np
import os
from PIL import Image
import re

import utils.path_utils as path_utils


def load_obj_mesh_pyassimp(obj_fp: str) -> Union[dict, List[dict]]:
    outer_fp = path_utils.get_parent(obj_fp)
    with pyassimp.load(obj_fp) as scene:
        if not scene.meshes:
            print("Error: No meshes found in file.")
            return None
        num_meshes = len(scene.meshes)
        mesh_d_l = []
        im_fps = []
        for mesh_num in range(num_meshes):
            mesh = scene.meshes[0]
            vertices = np.array(mesh.vertices)
            num_vertices = vertices.shape[0]
            faces = np.array(mesh.faces)
            try:
                uvs = mesh.texturecoords[0, :, :2]
            except:
                print(f"UV Coordinates missing for .obj {obj_fp} for mesh num {mesh_num}. Setting all to (0.0, 0.0)")
                uvs = np.zeros((num_vertices, 2), dtype=float)

            texture_array = None
            if scene.materials:
                material = scene.materials[mesh.materialindex]
                im_file_inner_fp = material.properties.get(('file', 1))
                if im_file_inner_fp is not None:
                    if re.search(r"\\\\+", im_file_inner_fp):
                        im_file_inner_fp = im_file_inner_fp.replace("\\\\", "\\")
                        im_file_inner_fp = im_file_inner_fp.replace("\\", "/")
                    im_file_fp = os.path.join(outer_fp, im_file_inner_fp)
                    if os.path.exists(im_file_fp):
                        im_fps.append(im_file_fp)
                        texture_image = Image.open(im_file_fp)
                        texture_array = np.array(texture_image)

            if texture_array is None:
                print(f"No texture found/properly loaded for obj {obj_fp} at mesh num {mesh_num}, using grey square of resolution 100X100...")
                texture_array = np.zeros((100, 100, 3), dtype=np.uint8)
                texture_array.fill(122)
            mesh_d = {"vertices": np.array(vertices, dtype=float), "faces": np.array(faces, dtype=int),
                      "uv_coords": np.array(uvs, dtype=float), "texture": np.flipud(texture_array),
                      "rgb_vals": None, "im_file_fps": im_fps}
            mesh_d_l.append(mesh_d)
    if len(mesh_d_l) >= 1:
        return mesh_d_l[0]
    else:
        return mesh_d_l


if __name__ == "__main__":
    sample_fp = "/mnt/Samsung_SSD_870_2/Human 3D Models/multihuman_single_raw/multihuman_single/DATA459/NORMAL.obj"
    mesh_d = load_obj_mesh_pyassimp(sample_fp)
    import PyOpenGL.pyopengl_mesh as pyopengl_mesh
    pyopengl_mesh.render_textured_mesh(mesh_d)