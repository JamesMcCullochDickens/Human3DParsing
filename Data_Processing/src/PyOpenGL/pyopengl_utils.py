import os
from typing import *

import glfw
import glm
from OpenGL.GL import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import utils.pc_utils as pc_utils


def np_to_fp32(np_arr: np.ndarray):
    if np_arr.dtype != np.float32:
        np_arr = np_arr.astype(np.float32)
    np_arr = np.ascontiguousarray(np_arr)
    return np_arr


def np_to_uint32(np_arr: np.ndarray):
    if np_arr.dtype != np.uint32:
        np_arr = np_arr.astype(np.uint32)
    np_arr = np.ascontiguousarray(np_arr)
    return np_arr


def error_callback(error, description):
    print(f"[GLFW ERROR] ({error}): {description.decode()}")


def create_window(width: int=800, height: int=600, is_hidden: bool=False):
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    if is_hidden:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    else:
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 16)
    window = glfw.create_window(width, height, "Window", None, None)
    if is_hidden:
        glfw.set_window_pos(window, -10000, -10000)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    if is_hidden:
        glfw.hide_window(window)
    return window



def create_fbo(width: int, height: int):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    if not fbo:
        raise RuntimeError("Failed to create framebuffer.")

    # Create a texture to store the color buffer
    color_texture = glGenTextures(1)
    if not color_texture:
        raise RuntimeError("Failed to create color texture for frame buffer.")
    glBindTexture(GL_TEXTURE_2D, color_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0)

    # Create a renderbuffer for depth
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer is not complete")

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, color_texture, rbo


def render_to_texture(window, fbo, width, height):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, width, height)
    glClearColor(0.1, 0.2, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def cleanup_fbo(fbo: int, color_texture: int, rbo: int):
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteRenderbuffers(1, [rbo])
    glDeleteTextures(1, [color_texture])
    glDeleteFramebuffers(1, [fbo])


def get_mesh_vbo_vao(vertices: np.ndarray, faces: np.ndarray,
                     uv_coords: Optional[np.ndarray] = None, with_uv_clipping: bool = False):
    # Validate input dimensions
    assert vertices.shape[1] == 3, "Vertices should be an Nx3 array"
    assert faces.shape[1] in {3, 4}, "Faces should be an Mx3 or Mx4 array"
    if uv_coords is not None:
        assert uv_coords.shape[1] == 2, "UV coordinates should be an Nx2 array"

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vertices = np_to_fp32(vertices)
    faces = np_to_uint32(faces)
    if uv_coords is not None:
        uv_coords = np_to_fp32(uv_coords)
        if with_uv_clipping:
            uv_coords = np.clip(uv_coords, a_min=0.0, a_max=1.0)

    # Create and bind vertex buffer
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    # Create and bind UV buffer if present
    uv_buffer = None
    if uv_coords is not None:
        uv_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, uv_buffer)
        glBufferData(GL_ARRAY_BUFFER, uv_coords.nbytes, uv_coords, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

    # Create and bind face buffer
    face_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    # Unbind VAO and buffers
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return vao, vertex_buffer, face_buffer, uv_buffer


def get_vertex_ids(num_vertices):
    vertex_ids = np.arange(0, num_vertices).astype(int)
    vertex_ids = np_to_uint32(vertex_ids)
    return vertex_ids


def get_pcd_vao(points: np.ndarray, colors: Optional[np.ndarray]):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    points_t = np_to_fp32(points)

    points_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, points_buffer)
    glBufferData(GL_ARRAY_BUFFER, points_t.nbytes, points_t, GL_STATIC_DRAW)

    if colors is not None:
        colors = np_to_fp32(colors)
        colors_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, colors_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    else:
        colors_buffer = None

    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, points_buffer)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * points_t.itemsize, None)

    if colors is not None:
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, colors_buffer)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * colors.itemsize, None)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, points_buffer, colors_buffer


def shader_file_to_str(shader_fp: str):
    assert os.path.exists(shader_fp), f"Shader path {shader_fp} does not exist."
    with open(shader_fp, 'r') as file:
        src = file.read()
    return src


def compile_shader(shader_type, source_fp: str):
    # The source may be a string or a file path
    if source_fp.endswith(".glsl"):
        source = shader_file_to_str(source_fp)
    else:
        source = source_fp[:]
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(f"Shader compilation failed for {source_fp}: {glGetShaderInfoLog(shader)}")
    return shader


def validate_program(program):
    glValidateProgram(program)
    if glGetProgramiv(program, GL_VALIDATE_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(program)
        raise RuntimeError(f"Program validation failed: {log}")



def compile_and_link_vertex_and_frag_shader(v_shad_fp: str, f_shad_fp: str):
    if glfw.get_current_context() is None:
        raise RuntimeError("No active OpenGL context. Create a window and set a context before compiling shaders.")
    shader_program = glCreateProgram()
    vertex_shader = compile_shader(GL_VERTEX_SHADER, v_shad_fp)
    frag_shader = compile_shader(GL_FRAGMENT_SHADER, f_shad_fp)
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, frag_shader)
    glLinkProgram(shader_program)
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(shader_program)
        raise RuntimeError(f"Shader linking failed: {log}")
    validate_program(shader_program)
    return shader_program


def configure_opengl_state(width, height, bkg_color:Tuple[float, float, float]=(1.0, 1.0, 1.0), window=None):
    glViewport(0, 0, width, height)
    glClearColor(bkg_color[0], bkg_color[1], bkg_color[2], 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def cleanup_and_exit(vaos, buffers, shader_program):
    if not isinstance(vaos, list):
        vaos = [vaos]
    glDeleteVertexArrays(len(vaos), vaos)
    glDeleteBuffers(len(buffers), buffers)
    if shader_program is not None:
        glDeleteProgram(shader_program)
        glUseProgram(0)


def numpy_to_glm_array(numpy_array):
    numpy_array = numpy_array.astype(np.float32)
    glm_array = glm.array(numpy_array.T)
    return glm_array


def glm_to_numpy_array(glm_array):
    numpy_array = np.array(glm_array)
    return numpy_array


def get_model_matrix_from_trs(translation: np.array, scale: float = 1.0, rotation: Optional[np.array] = None):
    translation = np_to_fp32(translation)
    if rotation is not None:
        rotation = np_to_fp32(rotation)
    translation = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2]))
    if rotation is None:
        rotation = glm.mat4(1.0)
    scale = glm.mat4(scale)
    model = translation * rotation * scale
    return model


def get_view_matrix(camera_pos: np.ndarray,
                    camera_target: np.ndarray = np.array([0.0, 0.0, 0.0]),
                    camera_up: np.ndarray = np.array([0.0, 1.0, 0.0])):
    camera_pos_v = glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2])
    camera_target_v = glm.vec3(camera_target[0], camera_target[1], camera_target[2])
    camera_up_v = glm.vec3(camera_up[0], camera_up[1], camera_up[2])
    view = glm.lookAt(camera_pos_v, camera_target_v, camera_up_v)
    return view


def get_perspective_matr(fov: float, aspect_ratio: float, near_clip: float = 0.1, far_clip: float = 100):
    fov = glm.radians(fov)
    projection = glm.perspective(fov, aspect_ratio, near_clip, far_clip)
    return projection


def get_rgb_frame_as_numpy(width, height, is_double_buffer: bool = False):
    if is_double_buffer:
        glReadBuffer(GL_BACK)
    else:
        glReadBuffer(GL_FRONT)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frame)
    # Flip the depth buffer vertically (OpenGL's origin is bottom-left)
    frame = np.flipud(frame)
    return frame


def get_depth_frame_as_numpy(width, height):
    glEnable(GL_DEPTH_TEST)
    depth = np.zeros((height, width), dtype=np.float32)
    glReadBuffer(GL_NONE)
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth)
    # Flip the depth buffer vertically (OpenGL's origin is bottom-left)
    depth = np.flipud(depth)
    return depth


def pixel_to_ndc(coords, viewport):
    x, y, width, height = viewport
    pixel_x = coords[..., 0]
    pixel_y = coords[..., 1]
    ndc_x = (2 * (pixel_x - x) / width) - 1
    ndc_y = 1 - (2 * (pixel_y - y) / height)
    ndc_coords = np.stack([ndc_x, ndc_y], axis=-1)
    return ndc_coords


def get_xy_grid(width: int, height: int):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
    coords = np.concatenate([x, y], axis=-1)
    return coords


def to_hom_from_3d(vec: np.ndarray):
    hom_vec = np.array([vec[0], vec[1], vec[2], 1.0], dtype=vec.dtype)
    return hom_vec


def ndc_to_camera(ndc_coords_xy: np.ndarray, depth_im: np.ndarray,
                 inv_matr_, remove_bkg: bool = True, mask: Optional[np.ndarray] = None):
    if not isinstance(inv_matr_, np.ndarray):
        inv_matr = glm_to_numpy_array(inv_matr_)
    else:
        inv_matr = inv_matr_.copy()
    h, w = ndc_coords_xy.shape[:2]

    ndc_coords_z = 2.0 * depth_im - 1.0
    max_depth = 1.0
    ndc_coords_z = np.expand_dims(ndc_coords_z, axis=-1)
    ndc_coords_x = np.expand_dims(ndc_coords_xy[:, :, 0], axis=-1)
    ndc_coords_y = np.expand_dims(ndc_coords_xy[:, :, 1], axis=-1)
    hom_coords_matr = np.concatenate([ndc_coords_x, ndc_coords_y,
                                      ndc_coords_z, np.ones((h, w, 1), dtype=ndc_coords_xy.dtype)], axis=-1)
    hom_coords_matr = hom_coords_matr.reshape(-1, 4)
    if remove_bkg:
        non_bkg_mask = hom_coords_matr[:, -2] != max_depth
        if mask is not None:
            non_bkg_mask = np.logical_and(mask, non_bkg_mask)
        hom_coords_matr_non_bkg = hom_coords_matr[non_bkg_mask]
        hom_camera_coords = (inv_matr @ hom_coords_matr_non_bkg.T).T
        threed_coords = hom_camera_coords[:, :3] / hom_camera_coords[:, 3][:, np.newaxis]
    else:
        non_bkg_mask = None
        hom_camera_coords = (inv_matr @ hom_coords_matr.T).T
        threed_coords = hom_camera_coords[:, :3] / hom_camera_coords[:, 3][:, np.newaxis]
        inf_rows = np.any(np.isinf(threed_coords), axis=1)
        threed_coords[inf_rows] = -1e6
        nan_rows = np.any(np.isnan(threed_coords), axis=1)
        threed_coords[nan_rows] = -1e6

    return threed_coords, non_bkg_mask


def numpy_array_to_texture_id(im_arr: np.ndarray):
    height, width, channels = im_arr.shape
    if channels == 3:
        internal_format = GL_RGB
        format = GL_RGB
    elif channels == 4:
        internal_format = GL_RGBA
        format = GL_RGBA
    else:
        raise ValueError("Unsupported array format. Expecting 3 (RGB) or 4 (RGBA) channels.")

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -1)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, GL_UNSIGNED_BYTE, im_arr)
    glGenerateMipmap(GL_TEXTURE_2D)
    """
    try:
        max_anisotropy = glGetFloatv(GL_TEXTURE_MAX_ANISOTROPY)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, int(max_anisotropy))
    except:
        print(f"Anisotropic texture max not supported...")
    """
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


def get_transform_matrix_inverse(projection_matr , view_matr, model_matr):
    inverse_matrix = np.linalg.inv(projection_matr @ view_matr @ model_matr)
    return inverse_matrix


def o3d_mesh_uvs_to_vertex_color(mesh):
    import open3d as o3d
    if not mesh.has_triangle_uvs():
        raise ValueError("Mesh does not have UV coordinates.")

    texture = np.asarray(mesh.textures[0])
    triangle_uvs = np.asarray(mesh.triangle_uvs)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    vertex_count = np.zeros(len(mesh.vertices),)

    for triangle_index, triangle in enumerate(triangles):
        for i, vertex_index in enumerate(triangle):
            uvs = triangle_uvs[triangle_index * 3 + i]
            u = uvs[0]
            v = uvs[1]
            x = int(u * (texture.shape[1] - 1))
            y = int(v * (texture.shape[0] - 1))
            color = texture[y, x]
            vertex_colors[vertex_index] += color[:3]
            vertex_count[vertex_index] += 1

    # Avoid division by zero
    valid_mask = vertex_count > 0
    vertex_colors[valid_mask] =  vertex_colors[valid_mask]/vertex_count[valid_mask][:, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def rotate_vertices_z_axis(points, theta: float):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    points_ = (rotation_matrix@points.T).T
    return points_


def get_camera_position_from_view_matrix(view_matrix):
    inv_view = np.linalg.inv(view_matrix)
    camera_pos = inv_view @ np.array([0, 0, 0, 1])
    return camera_pos[:3]


def rotate_vertices_y_axis(points, theta: float):
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    points_ = (rotation_matrix @ points.T).T
    return points_


def o3d_uv_vertices_to_ogl_vertices(mesh):
    triangle_uvs = np.asarray(mesh.triangle_uvs)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    is_set = set()
    vertex_uvs = np.zeros((vertices.shape[0], 2), dtype=vertices.dtype)
    for triangle_index, triangle in enumerate(triangles):
        for i, vertex_index in enumerate(triangle):
            if vertex_index not in is_set:
                uvs = triangle_uvs[triangle_index * 3 + i]
                u = uvs[0]
                v = uvs[1]
                vertex_uvs[vertex_index][0] = u
                vertex_uvs[vertex_index][1] = v
    return vertex_uvs


def get_inv_matr(glm_matr):
    inv_matr = glm.inverse(glm_matr)
    return inv_matr


def error_check():
    glFinish()
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {error}")
        exit()


def get_color_buffer(x: int, y: int, width: int, height: int) -> np.ndarray:
  buffer = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
  color_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
  color_array = np.flipud(color_array)
  return color_array

def get_depth_buffer(x: int, y: int, width: int, height: int):
    buffer = glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_array = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)
    depth_array = np.flipud(depth_array)
    return depth_array

def rotate_model_matr_y_axis(model_matr, angle):
    rot_y = glm.rotate(glm.mat4(1.0), glm.radians(angle), glm.vec3(0, 1, 0))
    model_matr_rotated = rot_y * model_matr
    return model_matr_rotated


def align_mesh_with_max_variance(vertices: np.ndarray):
    centered_vertices = vertices - np.mean(vertices, axis=0)
    covariance_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    max_variance_axis = eigenvectors[:, np.argmax(eigenvalues)]
    y_axis = np.array([0, 1, 0])
    axis_of_rotation = np.cross(max_variance_axis, y_axis)
    angle_of_rotation = np.arccos(np.dot(max_variance_axis, y_axis))

    if np.linalg.norm(axis_of_rotation) < 1e-8:
        rotation_matrix = np.eye(3)  # No rotation needed if max_variance_axis is already aligned with y
    else:
        rotation = R.from_rotvec(axis_of_rotation / np.linalg.norm(axis_of_rotation) * angle_of_rotation)
        rotation_matrix = rotation.as_matrix()
    aligned_vertices = centered_vertices @ rotation_matrix.T
    return aligned_vertices, rotation_matrix


def get_mesh_vbo_vao_point_color(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vertices = np_to_fp32(vertices)
    colors = np_to_fp32(colors)
    faces = np_to_uint32(faces)

    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    color_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

    face_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vertex_buffer, color_buffer, face_buffer


def get_closest_camera(vertices, model_matr, view_matr, num_rotations):
    rotation_angle = 360 // num_rotations
    dists_from_camera = []
    for i in range(num_rotations):
        vertices_in_camera_space = get_vertices_camera_space(vertices, model_matr, view_matr)
        dist_from_camera = np.linalg.norm(vertices_in_camera_space, axis=-1)
        dists_from_camera.append(dist_from_camera)
        model_matr = rotate_model_matr_y_axis(model_matr, rotation_angle)
    dists_from_camera = np.array(dists_from_camera)
    closest_camera = np.argmin(dists_from_camera, axis=0)
    return dists_from_camera, closest_camera



def get_vertices_camera_space(vertices, model_matr, view_matr):
    num_vertices = vertices.shape[0]
    vertices_hom = np.concatenate([vertices, np.expand_dims(np.ones(num_vertices), axis=-1)], axis=-1)
    vertices_in_camera_space = (np.array(view_matr) @ np.array(model_matr) @ vertices_hom.T).T
    vertices_in_camera_space = vertices_in_camera_space[:, :3]
    return vertices_in_camera_space



def backproject_to_closest(vertices, width, height, depth_buffer_im,
                           inv_matr, to_gpu: bool = False, mask: Optional[np.ndarray]=None):
    xy_grid = get_xy_grid(width, height)
    ndc_coords_xy = pixel_to_ndc(xy_grid, viewport=[0, 0, width, height])
    camera_coords, non_bkg_mask = ndc_to_camera(ndc_coords_xy, depth_buffer_im, inv_matr,
                                                remove_bkg=True, mask=mask)

    assert np.sum(non_bkg_mask) == camera_coords.shape[0], f"Error somewhere"

    grid_map = np.flatnonzero(non_bkg_mask)

    if to_gpu: # Faster but very memory intensive, OOM errors on both the cpu and gpu
        closest_idx = pc_utils.closest_point_idx_pt_batched(vertices, camera_coords, batch_size=5000, to_gpu=True)
    else:
        closest_idx = pc_utils.closest_point_idx(vertices, camera_coords)

    closest_idx_remapped = grid_map[closest_idx]

    return camera_coords, closest_idx_remapped


def bind_vertex_ids_to_vao(vao, vertex_ids, attribute_number: int):
    glBindVertexArray(vao)
    index_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, index_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertex_ids.nbytes, vertex_ids, GL_STATIC_DRAW)
    glEnableVertexAttribArray(attribute_number)
    glVertexAttribIPointer(attribute_number, 1, GL_INT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return index_buffer

def process_camera_movement(window, camera):
    key_direction_map = {
        glfw.KEY_W: "FORWARD",
        glfw.KEY_S: "BACKWARD",
        glfw.KEY_A: "LEFT",
        glfw.KEY_D: "RIGHT"
    }

    for key, direction in key_direction_map.items():
        if glfw.get_key(window, key) == glfw.PRESS:
            camera.process_keyboard(direction)

def cursor_pos_callback(window, xpos, ypos, camera):
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        x_offset = -1 * (xpos - camera.last_x)
        y_offset = -1 * (camera.last_y - ypos)
        camera.last_x = xpos
        camera.last_y = ypos
        if camera.is_left_clicked:
            camera.process_mouse_movement(x_offset, y_offset)
        else:
            camera.is_left_clicked = True


def mouse_button_callback(window, button, action, mods, camera):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        camera.is_left_clicked = False


def inspect_vao(vao_id):
    glBindVertexArray(vao_id)

    info = {
        "attributes": [],
        "enabled_attributes": [],
        "buffer_bindings": [],
    }

    # Query the maximum number of vertex attributes
    max_attributes = glGetIntegerv(GL_MAX_VERTEX_ATTRIBS)

    for i in range(max_attributes):
        # Check if the attribute is enabled
        enabled = glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED)
        if enabled:
            info["enabled_attributes"].append(i)
            attribute_info = {
                "index": i,
                "size": glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE),
                "type": glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE),
                "stride": glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE),
                "buffer_binding": glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING),
                "normalized": bool(glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED)),
                "pointer": glGetVertexAttribPointerv(i, GL_VERTEX_ATTRIB_ARRAY_POINTER),
            }
            info["attributes"].append(attribute_info)

    # Query buffer bindings
    for binding_index in range(max_attributes):
        buffer_id = glGetIntegeri_v(GL_VERTEX_ARRAY_BINDING, binding_index)
        if buffer_id:
            binding_info = {
                "binding_index": binding_index,
                "buffer_id": buffer_id,
            }
            info["buffer_bindings"].append(binding_info)

    # Unbind the VAO
    glBindVertexArray(0)
    return info


def unbind_all_textures():
    # Get the number of texture units
    max_texture_units = glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)

    for unit in range(max_texture_units):
        # Activate each texture unit
        glActiveTexture(GL_TEXTURE0 + unit)
        # Unbind textures from all possible targets
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindTexture(GL_TEXTURE_3D, 0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glBindTexture(GL_TEXTURE_1D, 0)
        glBindTexture(GL_TEXTURE_1D_ARRAY, 0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        glBindTexture(GL_TEXTURE_RECTANGLE, 0)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0)

    # Reset to default texture unit (optional)
    glActiveTexture(GL_TEXTURE0)


def get_per_point_colors(mesh_d: dict) -> np.ndarray:
    texture = mesh_d["texture"]
    uv_coords = mesh_d["uv_coords"]

    texture = texture.astype(np.float32)
    if np.max(texture) > 1.0:
        texture /= 255.0

    h, w = texture.shape[:2]
    u_px = np.clip((uv_coords[:, 0] * w).astype(int), 0, w - 1)
    v_px = np.clip(((1.0 - uv_coords[:, 1]) * h).astype(int), 0, h - 1)
    colors = texture[v_px, u_px]

    return colors



