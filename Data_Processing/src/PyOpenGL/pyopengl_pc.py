import glfw
import glm
from OpenGL.GL import *
import numpy as np

import src.PyOpenGL.pyopengl_utils as pyogl_utils
import utils.pc_utils as pc_utils


def get_sample_pcd_d():
    points = np.random.rand(100, 3).astype(float)
    colors = np.random.rand(100, 3)
    mesh_d = {"points": points, "colors": colors}
    return mesh_d


def window_size_callback(window, width, height):
    # Set the viewport to the new window size
    glViewport(0, 0, width, height)
    print(f"Window resized to: {width} x {height}")


v_shad = """
#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;

out vec3 fragment_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(vertex_position, 1.0);
    fragment_color = vertex_color;
}
"""

f_shad = """
#version 330 core

in vec3 fragment_color;
out vec4 color;

void main() {
    color = vec4(fragment_color, 1.0);
}
"""


def render_pcd(pcd_d: dict, width: int = 800, height: int = 600,
               show_window: bool = False) -> np.ndarray:
    global v_shad, f_shad
    aspect_ratio = width/height

    glEnable(GL_DEPTH_TEST)
    window = pyogl_utils.create_window(width=width, height=height, is_hidden=show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    points, colors = pcd_d["points"], pcd_d["colors"]
    num_points = points.shape[0]

    # model matrix
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1*center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(points)
    camera_distance = 2.5*model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos)

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.1, far_clip=100)
    #inv_matr = glm.inverse(projection_matr * view_matr * model_matr)

    vao, points_buffer, colors_buffer = pyogl_utils.get_pcd_vao(points, colors)
    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad, f_shad)
    buffers = [vao, points_buffer]
    if colors_buffer is not None:
        buffers.append(colors_buffer)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")

    glUseProgram(shader_program)
    glBindVertexArray(vao)

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matr))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matr))

    pyogl_utils.error_check()
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glfw.set_window_size_callback(window, window_size_callback)
    glPointSize(10.0)
    if not show_window:
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_POINTS, 0, num_points)
    glfw.swap_buffers(window)
    color_im = pyogl_utils.get_color_buffer(0, 0, width, height)
    """
    if not show_window:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    else:
        glfw.swap_buffers(window)
    """


    if window:
        glfw.destroy_window(window)

    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)
    glfw.terminate()
    return color_im


if __name__ == "__main__":
    # Sample use

    sample_pcd_d = get_sample_pcd_d()
    color_im = render_pcd(sample_pcd_d, show_window=False)
    import Visualization.frame_utils as frame_utils
    frame_utils.show_im(color_im)

    ...
