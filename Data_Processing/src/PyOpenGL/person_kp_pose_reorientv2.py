import glfw
import glm
import numpy as np
from OpenGL.GL import *
from PIL import Image

import src.PyOpenGL.pyopengl_utils as pyogl_utils
from utils import pc_utils
from utils import frame_utils
import src.Yolov8.yolo_prediction as yolo_pred
import utils.frame_utils as frame_utils


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


# Constants
left_shoulder_idx = 6
right_shoulder_idx = 7
left_hip_idx = 12
right_hip_idx = 13


def closest_axis(vector):
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
        return v / vec_norm


def should_apply_adjustment(kps, kp_conf_vals):
    # Check conditions that indicate adjustment is needed
    needs_adjustment = False

    # 1. Check horizontal alignment using shoulder-hip vector
    hip_to_shoulder_vec = normalize_vec(np.array(kps[left_shoulder_idx]) - np.array(kps[left_hip_idx]))
    if closest_axis(hip_to_shoulder_vec) == "x":  # if body is horizontally flipped
        needs_adjustment = True

    # 2. Check vertical orientation
    if kps[left_shoulder_idx][1] > kps[left_hip_idx][1]:  # if body is vertically flipped
        needs_adjustment = True

    # 3. Check depth/front-back orientation
    if kp_conf_vals[0] <= 0.3:  # if nose not visible (likely facing away)
        needs_adjustment = True

    # 4. Check side lean based on shoulder confidence differences
    if 0.15 < abs(kp_conf_vals[3] - kp_conf_vals[4]):  # shoulder asymmetry
        needs_adjustment = True

    # 5. Check forward/backward lean
    if (abs(kp_conf_vals[1] - kp_conf_vals[2]) > 0.3) and kp_conf_vals[0] >= 0.3:
        needs_adjustment = True

    return needs_adjustment


def im_based_adjust(kps, kp_conf_vals, mesh_d):

    vertices = mesh_d["vertices"]

    # Horizontal flip detection
    hip_to_shoulder_vec = normalize_vec(np.array(kps[left_shoulder_idx]) - np.array(kps[left_hip_idx]))
    if closest_axis(hip_to_shoulder_vec) == "x":  # if the body is flipped horizontally
        vertices = pyogl_utils.rotate_vertices_z_axis(vertices, np.pi / 2)

    # Vertical flip detection
    if kps[left_shoulder_idx][1] > kps[left_hip_idx][1]:  # if the body is flipped vertically
        vertices[:, 1] *= -1

    # Depth flip detection (back/front)
    if kp_conf_vals[0] <= 0.3:  # if the nose is not visible, we are viewing from behind
        vertices[:, -1] *= -1
    if 0.15 < abs(kp_conf_vals[3] - kp_conf_vals[4]):
        # vertices[:, -1] *= -1

        if kp_conf_vals[3] < kp_conf_vals[4]:
            vertices = pyogl_utils.rotate_vertices_y_axis(vertices, -np.pi / 8)
        else:
            vertices = pyogl_utils.rotate_vertices_y_axis(vertices, np.pi / 8)

    if (abs(kp_conf_vals[1] - kp_conf_vals[2]) > 0.3) and kp_conf_vals[0] >= 0.3:
        if kp_conf_vals[3] < kp_conf_vals[4]:
            vertices = pyogl_utils.rotate_vertices_y_axis(vertices, -np.pi / 4)
        else:
            vertices = pyogl_utils.rotate_vertices_y_axis(vertices, np.pi / 4)

    mesh_d["vertices"] = vertices
    return mesh_d


def get_pose_metrics(kps, kp_conf_vals):
    metrics = {
        # Existing metrics
        'shoulder_hip_alignment': abs(np.array(kps[left_shoulder_idx])[1] - np.array(kps[left_hip_idx])[1]),
        'shoulder_confidence': (kp_conf_vals[left_shoulder_idx] + kp_conf_vals[right_shoulder_idx]) / 2,
        'face_visibility': kp_conf_vals[0],  # nose confidence
        'overall_confidence': np.mean(kp_conf_vals),

        # New eye and ear metrics
        'left_eye_conf': kp_conf_vals[1],
        'right_eye_conf': kp_conf_vals[2],
        'left_ear_conf': kp_conf_vals[3],
        'right_ear_conf': kp_conf_vals[4],

        # Average eye and ear confidence
        'eyes_confidence': (kp_conf_vals[1] + kp_conf_vals[2]) / 2,
        'ears_confidence': (kp_conf_vals[3] + kp_conf_vals[4]) / 2
    }
    return metrics


def is_pose_improved(prev_metrics, curr_metrics, ear_threshold=0.4):
    """Compare if current pose metrics are better than previous with a threshold for ear improvement"""

    # Ear improvements with a threshold
    ear_improvements = [
        (curr_metrics['left_ear_conf'] - prev_metrics['left_ear_conf']) > ear_threshold,
        (curr_metrics['right_ear_conf'] - prev_metrics['right_ear_conf']) > ear_threshold
    ]

    # core_improved = sum(core_improvements) > len(core_improvements)/2
    ears_improved = all(ear_improvements)

    # return core_improved and ears_improved
    return ears_improved


def iterative_auto_adjust_mesh(mesh_d, max_iterations=2):
    prev_metrics = None
    iteration = 0

    mesh_d, color_im = person_kp_pose_orient(mesh_d, with_mvalign_bool=True)

    init_color_im = color_im.copy()
    while iteration < max_iterations:
        # Get current pose estimation
        kps, kp_conf_vals = yolo_pred.get_kps(color_im)
        if kps.size == 0:
            break
        kps = kps[0]
        kp_conf_vals = kp_conf_vals[0]
        curr_metrics = get_pose_metrics(kps, kp_conf_vals)


        # Check if adjustment needed
        if not should_apply_adjustment(kps, kp_conf_vals):
            #print("No adjustment needed")
            break

        # Check if pose improved from previous iteration
        if prev_metrics and is_pose_improved(prev_metrics, curr_metrics):
            #print("No further improvement")
            break

        # Apply adjustment
        mesh_d = im_based_adjust(kps, kp_conf_vals, mesh_d)
        mesh_d, color_im = person_kp_pose_orient(mesh_d, with_mvalign_bool=False)

        prev_metrics = curr_metrics
        iteration += 1

    # Sanity Check
    #frame_utils.show_im(init_color_im)
    #frame_utils.show_im(color_im)
    return mesh_d, init_color_im, color_im


def person_kp_pose_orient(mesh_d, height: int = 640, width: int = 640,
                          with_mvalign_bool: bool = True, show_window: bool = False):

    # vertices, faces, and per-vertex face ids
    window = pyogl_utils.create_window(width=width, height=height, is_hidden=not show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    vertices, faces = mesh_d["vertices"], mesh_d["faces"]
    if with_mvalign_bool:
        vertices = pyogl_utils.align_mesh_with_max_variance(vertices)[0]
    uv_coords = mesh_d["uv_coords"]
    aspect_ratio = width / height
    texture =  mesh_d["texture"]
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

    # frame_utils.show_im(color_im)

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


    mesh_d["vertices"] = vertices
    return mesh_d, color_im

if __name__ == "__main__":
    ...