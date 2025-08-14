import open3d as o3d
import numpy as np


def load_obj_mesh_o3d(obj_mesh_fp: str, with_normals: bool = False) -> dict:
    mesh = o3d.io.read_triangle_mesh(obj_mesh_fp)
    if with_normals:
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    uv_coords_per_face = np.asarray(mesh.triangle_uvs)
    uv_coords_per_vertex = uv_coords_per_face_to_uv_coords_per_vertex(uv_coords_per_face, faces, vertices.shape[0])
    texture = np.asarray(mesh.textures[-1])
    return {"vertices": vertices, "faces": faces, "uv_coords_per_face": uv_coords_per_face,
            "texture": texture, "uv_coords": uv_coords_per_vertex}


def uv_coords_per_face_to_uv_coords_per_vertex(uv_coords_per_face, faces, num_vertices: int):
    uv_coords_per_vertex = np.zeros((num_vertices, 2), dtype=float)
    processed = set()
    for face_num, f in enumerate(faces):
        for i in range(3):
            f_i = f[i]
            if f_i not in processed:
                uv_vals = uv_coords_per_face[face_num*3 + i]
                uv_coords_per_vertex[f_i] = uv_vals
    return uv_coords_per_vertex

