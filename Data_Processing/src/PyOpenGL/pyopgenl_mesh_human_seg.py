import os
import glfw
import glm
from OpenGL.GL import *
import numpy as np
import utils.path_utils as path_utils

if "cwd" not in os.environ.keys():
    os.environ["cwd"] = path_utils.get_parent(os.getcwd())


import src.PyOpenGL.pyopengl_utils as pyogl_utils
import utils.pc_utils as pc_utils


def window_size_callback(window, width, height):
    # Set the viewport to the new window size
    glViewport(0, 0, width, height)
    print(f"Window resized to: {width} x {height}")


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



def get_human_mesh_render(mesh_d: dict, rotation_angle: int = 45,
                          show_window: bool = False, width: int = 1920,
                          height: int = 1080) -> dict:
    aspect_ratio = width/height
    global v_shad, f_shad

    window = pyogl_utils.create_window(width=width, height=height, is_hidden=not show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    glEnable(GL_DEPTH_TEST)
    vertices, texture, faces, uv_coords = (mesh_d["vertices"], mesh_d["texture"],
                                           mesh_d["faces"], mesh_d["uv_coords"])

    vertices, _ = pyogl_utils.align_mesh_with_max_variance(vertices)

    num_triangles = faces.shape[0]
    num_indices = num_triangles*3

    texture_id = pyogl_utils.numpy_array_to_texture_id(texture)
    if not texture_id:
        raise RuntimeError("Failed to create texture from numpy array")

    # model matrix
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1*center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 2.7*model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos)

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.1, far_clip=1000.0)

    vao, vertex_buffer, face_buffer, uv_buffer = pyogl_utils.get_mesh_vbo_vao(vertices, faces, uv_coords,
                                                                                  with_uv_clipping=True)

    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad, f_shad)
    buffers = [vertex_buffer, face_buffer, uv_buffer]

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    texture_loc = glGetUniformLocation(shader_program, "texture_sampler")
    for loc in [model_loc, view_loc, projection_loc, texture_loc]:
        if loc == -1:
            raise RuntimeError("Failed to find uniform location")

    glUseProgram(shader_program)
    glBindVertexArray(vao)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(texture_loc, 0)

    model_matr_orig = np.array(model_matr).copy()
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matr))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matr))

    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)

    if show_window:
        glfw.set_window_size_callback(window, window_size_callback)

    counter = 0
    color_buffer_ims = []
    depth_buffer_ims = []
    angle = rotation_angle
    num_rotations = 360 // angle
    pyogl_utils.error_check()
    while not glfw.window_should_close(window):
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        glFinish()

        # Read from back buffer or frame buffer
        color_buffer_im = pyogl_utils.get_color_buffer(0, 0, width, height)
        depth_buffer_im = pyogl_utils.get_depth_buffer(0, 0, width, height)

        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        else:
            glfw.swap_buffers(window)  # Swap only for visible window

        glfw.poll_events()

        color_buffer_ims.append(color_buffer_im)
        depth_buffer_ims.append(depth_buffer_im)

        counter += 1
        if counter > num_rotations - 1:
            break

        # Update model matrix and re-upload it
        model_matr = pyogl_utils.rotate_model_matr_y_axis(model_matr, angle)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))


    if not show_window:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    pyogl_utils.unbind_all_textures()
    glDeleteTextures(1, [texture_id])
    if not show_window:
        pyogl_utils.cleanup_fbo(fbo, color_texture, rbo)
    if window:
        glfw.destroy_window(window)
    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)


    d = {"color_ims": color_buffer_ims, "depth_ims": depth_buffer_ims,
         "im_height": height, "im_width": width, "vertices": vertices,
          "faces": faces, "model_matr": model_matr_orig,
         "view_matr": np.array(view_matr), "projection_matr": projection_matr}

    return d


if __name__ == "__main__":
    """
    sample_human_obj = "/home/jamesdickens/Desktop/Code/ThesisResearch/PyOpenGL/Sample_OBJ_Meshes/Ex4/mesh.obj"
    obj = obj_parser.Obj("/home/jamesdickens/Desktop/Code/ThesisResearch/PyOpenGL/Sample_OBJ_Meshes/Ex4/mesh.obj")
    mesh_d = obj_parser.get_first_mesh_d(obj)
    get_human_mesh_render(mesh_d, show_window=True, with_back_project=False)
    """
    ...