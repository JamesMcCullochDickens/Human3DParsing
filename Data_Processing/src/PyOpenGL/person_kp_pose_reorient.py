import numpy as np
from OpenGL.GL import *
import glm
import glfw

import PyOpenGL.pyopengl_utils as pyogl_utils
import PC_Utils.pc_utils as pc_utils
import Visualization.pc_vis as pc_vis
import Visualization.frame_utils as frame_utils
from PIL import Image
import Yolov8.yolo_prediction as yolo_pred

left_shoulder_idx = 6
right_shoulder_idx = 7
left_hip_idx = 12

v_shad = """
    #version 330 core

    layout(location = 0) in vec3 vertex_position;
    layout(location = 1) in vec2 vertex_uv;

    out vec2 fragment_uv;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(vertex_position, 1.0);
        fragment_uv = vertex_uv;
    }
    """

f_shad = """
    #version 330 core

    in vec2 fragment_uv;
    out vec4 color;

    uniform sampler2D texture_sampler;

    void main() {
        vec4 sampledColor = texture(texture_sampler, fragment_uv);
        color = vec4(sampledColor.rgb, sampledColor.a);
    }

"""

def get_first_label_instance(l, label):
    indices = np.where(l == label)[0]
    if len(indices) > 0:
        first_index = indices[0]
        return first_index
    else:
        return -1


def closest_axis(vector):
    """
    Finds the closest axis (x or y) to the given 2D vector,
    considering flipped directions.

    Args:
        vector (tuple or list): A 2D vector (x, y).

    Returns:
        str: 'x' or 'y', indicating the closest axis.
    """
    # Normalize the input vector
    vector = np.array(vector)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("The input vector must not be zero.")
    normalized_vector = vector / magnitude

    # Define the axes
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Compute the absolute dot products
    x_similarity = abs(np.dot(normalized_vector, x_axis))
    y_similarity = abs(np.dot(normalized_vector, y_axis))

    # Determine the closest axis
    return 'x' if x_similarity >= y_similarity else 'y'


def normalize_vec(v):
    vec_norm = np.linalg.norm(v)
    if vec_norm == 0:
        return vec_norm
    else:
        return v/vec_norm


def person_kp_pose_orient(mesh_d, height: int, width: int,
                          with_mvalign_bool = True, show_window:bool = False):
    # vertices, faces, and per-vertex face ids
    window = pyogl_utils.create_window(width=width, height=height, is_hidden=not show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    vertices, faces = mesh_d["vertices"], mesh_d["faces"]
    if with_mvalign_bool:
        vertices = pyogl_utils.align_mesh_with_max_variance(vertices)[0]
    uv_coords = mesh_d["uv_coords"]
    im_fp = str(mesh_d["im_file_fp"])
    aspect_ratio = width / height
    texture =  np.flipud(np.asarray(Image.open(im_fp)))[..., :3].astype(np.uint8)
    vertices = pyogl_utils.np_to_fp32(vertices)
    faces = pyogl_utils.np_to_uint32(faces)
    uv_coords = pyogl_utils.np_to_fp32(uv_coords)

    num_triangles = faces.shape[0]
    num_indices = num_triangles * 3
    if texture.shape[-1] == 4:  # Remove alpha channel
        texture = texture[:, :, :3]
    if texture.dtype != np.uint8:
        texture = texture.astype(np.uint8)

    texture_id = pyogl_utils.numpy_array_to_texture_id(texture)

    # model matrix
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1 * center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 2.7 * model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos)

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.01, far_clip=1000)
    # inv_matr = glm.inverse(projection_matr * view_matr * model_matr)

    vao, vertex_buffer, face_buffer, uv_buffer = pyogl_utils.get_mesh_vbo_vao(vertices, faces, uv_coords)
    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad, f_shad)
    buffers = [vao, vertex_buffer, face_buffer, uv_buffer]

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    texture_loc = glGetUniformLocation(shader_program, "texture_sampler")

    glUseProgram(shader_program)
    glBindVertexArray(vao)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(texture_loc, 0)

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matr))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matr))

    pyogl_utils.error_check()
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glEnable(GL_DEPTH_TEST)
    while not glfw.window_should_close(window):
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        color_im = pyogl_utils.get_color_buffer(0, 0, width, height)
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        else:
            glfw.swap_buffers(window)
        glfw.poll_events()
        break

    glBindVertexArray(0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if not show_window:
        pyogl_utils.cleanup_fbo(fbo, color_texture, rbo)
    glDeleteTextures(1, [texture_id])
    pyogl_utils.unbind_all_textures()
    if window:
        glfw.destroy_window(window)
    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)
    glfw.terminate()

    kps, kp_conf_vals = yolo_pred.get_kps(color_im)
    kps = kps[0]
    kp_im = frame_utils.get_kp_im(color_im, kps, height, width)
    frame_utils.show_im(kp_im)
    kp_conf_vals = kp_conf_vals[0]

    hip_to_shoulder_vec = normalize_vec(np.array(kps[left_shoulder_idx])-np.array(kps[left_hip_idx]))
    if closest_axis(hip_to_shoulder_vec) == "x":  # if the body is flipped horizontally
        vertices = pyogl_utils.rotate_vertices_z_axis(vertices, np.pi/2)
    if kps[left_shoulder_idx][1] > kps[left_hip_idx][1]: # if the body is flipped vertically
        vertices[:, 1] *= -1
    if kp_conf_vals[0] <= 0.5:  # if the nose is not visible, we are viewing from behind
        vertices[:, -1] *= -1

    # sanity check
    #mesh_d["vertices"] = vertices.copy()
    #person_kp_pose_orient(mesh_d, height, width, with_mvalign_bool=False)

    return vertices



if __name__ == "__main__":
    sampled_model_npz_fp = "/mnt/SabrentRocket_4TB/Human_3D_Models/Renders/THuman2.1_Release_0-300_0040/model.npz"
    outer_texture_fp = "/mnt/Samsung_SSD_870_2/Human 3D Models/THuman2.1_Release_0-300/0040"
    mesh_d = dict(np.load(sampled_model_npz_fp, allow_pickle=True))
    mesh_d["im_file_fp"] = "/mnt/Samsung_SSD_870_2/Human 3D Models/THuman2.1_Release_0-300/0040/material0.jpeg"
    person_kp_pose_orient(mesh_d, 640, 640)