import os

import numpy as np
from PIL import Image
from jupyter_core.version import parts

from utils import path_utils

def filt_empty_string(l_str):
    return list(filter(lambda x: x not in [""], l_str))

class Material:
    def __init__(self, name):
        self.name = name
        self.Ka = None  # Ambient color
        self.Kd = None  # Diffuse color
        self.Ks = None  # Specular color
        self.Ns = None  # Specular exponent
        self.d = None   # Transparency (dissolve)
        self.map_Kd = None  # Diffuse texture map


class Mtl:
    def __init__(self, mtl_fp: str):
        self.mtl_fp = mtl_fp
        self.outer_fp = path_utils.get_parent(mtl_fp)
        self.materials = {}
        self.textures = {}
        self.parse_mtl(self.mtl_fp)

    def parse_mtl(self, file_path):
        current_material = None

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()

                if not parts or parts[0].startswith('#'):
                    continue  # Skip empty lines or comments

                if parts[0] == 'newmtl':  # New material definition
                    material_name = parts[1]
                    current_material = Material(material_name)
                    self.materials[material_name] = current_material

                elif parts[0] == 'Ka' and current_material:  # Ambient color
                    current_material.Ka = [float(val) for val in parts[1:]]

                elif parts[0] == 'Kd' and current_material:  # Diffuse color
                    current_material.Kd = [float(val) for val in parts[1:]]

                elif parts[0] == 'Ks' and current_material:  # Specular color
                    current_material.Ks = [float(val) for val in parts[1:]]

                elif parts[0] == 'Ns' and current_material:  # Specular exponent
                    current_material.Ns = float(parts[1])

                elif parts[0] == 'd' and current_material:  # Transparency (dissolve)
                    current_material.d = float(parts[1])

                elif parts[0] == 'map_Kd' and current_material:  # Diffuse texture map
                    current_material.map_Kd = parts[1]
                    im_fp = os.path.join(self.outer_fp, parts[1])
                    im = np.asarray(Image.open(im_fp))
                    self.textures[current_material.name] = im


class Mesh:
    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.uv_coords = []
        self.faces = []
        self.normals = []
        self.texture = None

class Obj:
    def __init__(self, file_path):
        self.meshes = []
        self.current_mesh = None
        self.outer_dir = path_utils.get_parent(file_path)
        self.mtl_fps = []
        self.mtl_objs = []
        self.textures = {}
        self.parse_obj(file_path)


    def parse_obj(self, file_path):
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file):
                parts = line.strip().split()

                if not parts:
                    continue

                if parts[0] == "usemtl":
                    if self.current_mesh is None:
                        mesh_name = 'mesh_' + str(len(self.meshes))
                        self.start_new_mesh(mesh_name)
                    self.current_mesh.texture = self.textures.get(parts[1], None)

                if parts[0] == "mtllib":
                    mtl_fp = os.path.join(self.outer_dir, parts[1])
                    self.mtl_fps.append(mtl_fp)
                    mtl_obj = Mtl(mtl_fp)
                    self.mtl_objs.append(mtl_obj)
                    self.textures.update(mtl_obj.textures)

                if parts[0] == 'o' or parts[0] == 'g':  # New object or group (mesh)
                    mesh_name = parts[1] if len(parts) > 1 else 'mesh_' + str(len(self.meshes))
                    if self.current_mesh is None:
                        self.start_new_mesh(mesh_name)
                    else:
                        if self.current_mesh.faces:
                            self.start_new_mesh(mesh_name)
                        else:
                            ...

                elif parts[0] == 'v':  # Vertex
                    vertex = [float(val) for val in parts[1:]]
                    if self.current_mesh is None:
                        mesh_name = 'mesh_' + str(len(self.meshes))
                        self.current_mesh = Mesh(mesh_name)
                    vertex = np.array(vertex, dtype=float)
                    self.current_mesh.vertices.append(vertex)

                elif parts[0] == "vn": # Vertex normal
                    normal = [float(val) for val in parts[1:]]
                    normal = np.array(normal, dtype=float)
                    self.current_mesh.normals.append(normal)

                elif parts[0] == 'vt':  # Texture coordinates (UVs)
                    uv = [float(val) for val in parts[1:]]
                    uv = np.array(uv, dtype=float)
                    self.current_mesh.uv_coords.append(uv)

                elif parts[0] == 'f':  # Face
                    face = []
                    if len(parts[1:]) == 3:
                        for part in parts[1:]:
                            face_indices = part.split("/")
                            face.append([int(idx) if idx else -1 for idx in face_indices])
                        face = np.array(face, dtype=int)
                        self.current_mesh.faces.append(face)
                    elif len(parts[1:]) == 4: # The case of a quad
                        for part in parts[1:]:
                            face_indices = part.split("/")
                            face.append([int(idx) if idx else None for idx in face_indices])
                        face1 = np.array(face[:3], dtype=int)
                        face2 = np.array(face[1:], dtype=int)

                        self.current_mesh.faces.append(face1)
                        self.current_mesh.faces.append(face2)
                    else: # a polygon
                        for part in parts[1:]:
                            face_indices = part.split('/')
                            face.append([int(idx) if idx else None for idx in face_indices])
                        face = np.array(face, dtype=int)
                        self.current_mesh.faces.append(face)

        if self.current_mesh:
            self.meshes.append(self.current_mesh)


    def start_new_mesh(self, name):
        if self.current_mesh:
            self.meshes.append(self.current_mesh)
        self.current_mesh = Mesh(name)

def vertex_counts_triangle_faces(faces):
    d = {}
    for f in faces:
        v1, v2, v3 = f[0][0], f[1][0], f[2][0]
        if v1 not in d.keys():
            d[v1] = 1
        else:
            d[v1] += 1
        if v2 not in d.keys():
            d[v2] = 1
        else:
            d[v2] += 1
        if v3 not in d.keys():
            d[v3] = 1
        else:
            d[v3] += 1
    return d


def remap_vertices_uvs_faces(vertices, uvs, faces):
    new_vertices = []
    new_uvs = []
    vertex_map = {}  # To map (vertex_index, uv_index) -> new index
    new_faces = []

    for face in faces:
        new_face = []
        for f in face:
            v_idx = f[0]-1
            uv_idx = f[1]-1
            key = (v_idx, uv_idx)
            if key not in vertex_map:
                new_vertices.append(vertices[v_idx])
                new_uvs.append(uvs[uv_idx])
                vertex_map[key] = len(new_vertices) - 1
            new_face.append(vertex_map[key])
        new_faces.append(new_face)
    return new_vertices, new_uvs, new_faces


def get_max_uv_idx(faces):
    max_uv_idx = 0
    for face in faces:
        for f in face:
            uv_idx = f[1] - 1
            max_uv_idx = max(uv_idx, max_uv_idx)
    return max_uv_idx

def get_first_mesh_d(obj):
    faces = obj.meshes[0].faces
    vertices = obj.meshes[0].vertices
    uvs = obj.meshes[0].uv_coords
    num_vertices, num_faces, num_uvs = len(vertices), len(faces), len(uvs)

    if num_vertices == 0:
        raise ValueError("No vertices in the first mesh.")

    if num_faces == 0:
        raise ValueError("No faces in the first mesh.")

    if num_uvs == 0:
        print(f"No uv coordinates in the first mesh.")

    # Determine if remapping is necessary:
    if num_vertices < get_max_uv_idx(faces):
        vertices, uvs, faces = remap_vertices_uvs_faces(vertices, uvs, faces)

    vertices = np.array(vertices, dtype=float)
    if vertices.shape[-1] == 6:
        rgb_vals = vertices[:, 3:]
        vertices = vertices[:, :3]
    else:
        rgb_vals = None

    if obj.meshes[0].texture is not None:
        texture = np.flipud(obj.meshes[0].texture.copy())
    else:
        texture_keys = list(obj.textures.keys())
        if texture_keys:
            texture = np.flipud(obj.textures[texture_keys[0]].copy())
        else:
            texture = np.zeros((100, 100, 3), dtype=np.uint8).fill(125)
    mesh_d = {"vertices": np.array(vertices, dtype=float), "faces": np.array(faces, dtype=int),
              "uv_coords": np.array(uvs, dtype=float), "texture": texture, "rgb_vals": rgb_vals}
    return mesh_d


def get_first_obj(outer_fp: str):
    inner_folder_fps = path_utils.join_inner_paths(outer_fp)
    obj_fps = list(filter(lambda x: ".obj" in x, inner_folder_fps))
    obj_fps.sort()
    if len(obj_fps) == 0:
        print(f"No .obj files in folder {outer_fp}, returning None")
        return None
    else:
        return obj_fps[0]

if __name__ == "__main__":
    """
    obj = Obj("/mnt/Samsung_SSD_870_2/Human 3D Models/multihuman_single_raw/multihuman_single/DATA459/NORMAL.obj")
    mesh_d = get_first_mesh_d(obj)
    import PyOpenGL.pyopengl_mesh as pyopgenl_mesh
    pyopgenl_mesh.render_textured_mesh(mesh_d)
    """
    ...