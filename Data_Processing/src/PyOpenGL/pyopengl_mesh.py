import glfw
import glm
from OpenGL.GL import *
import numpy as np
from functools import partial

import PyOpenGL.pyopengl_utils as pyogl_utils
import PyOpenGL.interactive_camera as interactive_camera
import PC_Utils.pc_utils as pc_utils


def window_size_callback(window, width, height):
    # Set the viewport to the new window size
    glViewport(0, 0, width, height)
    print(f"Window resized to: {width} x {height}")


def render_textured_mesh(mesh_d: dict,  width: int = 800,
                         height: int = 600, show_window: bool = True):
    aspect_ratio = width/height
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

    window = pyogl_utils.create_window(width=width, height=height, is_hidden= not show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)

    glEnable(GL_DEPTH_TEST)
    vertices, texture, faces, uv_coords = mesh_d["vertices"], mesh_d["texture"], mesh_d["faces"], mesh_d["uv_coords"]
    num_triangles = faces.shape[0]
    num_indices = num_triangles*3
    if texture.shape[-1] == 4: # Remove alpha channel
        texture = texture[:, :, :3]
    if texture.dtype != np.uint8:
        texture = texture.astype(np.uint8)

    texture_id = pyogl_utils.numpy_array_to_texture_id(texture)

    # model matrix
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1*center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 2.5*model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos)

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.01, far_clip=1000)
    #inv_matr = glm.inverse(projection_matr * view_matr * model_matr)

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
    glfw.set_window_size_callback(window, window_size_callback)
    #color_frame_buffer = []
    while not glfw.window_should_close(window):
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        color_buffer_im = pyogl_utils.get_color_buffer(0, 0, width, height)
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

    return color_buffer_im

def render_mesh_point_color(mesh_d, width: int = 800, height: int = 600,
                            show_window: bool = True):
    aspect_ratio = width / height
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

    out vec4 FragColor; 

    void main() {
        FragColor = vec4(fragment_color, 1.0); 
    }
    """
    window = pyogl_utils.create_window(width=width, height=height, is_hidden= not show_window)

    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)

    vertices, colors, faces = mesh_d["vertices"], mesh_d["colors"], mesh_d["faces"]
    num_triangles = faces.shape[0]
    num_indices = num_triangles * 3

    # model matrix
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1 * center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 5.0 * model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    camera = interactive_camera.InteractiveCamera(position=camera_pos, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))
    glfw.set_cursor_pos_callback(window, partial(pyogl_utils.cursor_pos_callback, camera=camera))
    glfw.set_mouse_button_callback(window, partial(pyogl_utils.mouse_button_callback, camera=camera))

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.1, far_clip=100)
    # inv_matr = glm.inverse(projection_matr * view_matr * model_matr)

    vao, vertex_buffer, color_buffer, face_buffer = pyogl_utils.get_mesh_vbo_vao_point_color(vertices, faces, colors)
    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad, f_shad)
    buffers = [vao, vertex_buffer, face_buffer, color_buffer]

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")

    glUseProgram(shader_program)
    glBindVertexArray(vao)

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(camera.view_matrix))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matr))

    pyogl_utils.error_check()
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glfw.set_window_size_callback(window, window_size_callback)

    while not glfw.window_should_close(window):
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        color_buffer_im = pyogl_utils.get_color_buffer(0, 0, width, height)
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
    if window:
        glfw.destroy_window(window)
    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)

    return color_buffer_im


def render_untextured_mesh(mesh_d: dict, width: int = 800,
                           height: int = 600, show_window: bool = True):
    aspect_ratio = width / height
    v_shad = """
    #version 330 core 
    layout(location = 0) in vec3 vertex_position;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() {
        gl_Position = projection * view * model * vec4(vertex_position, 1.0);
    }
    """

    f_shad = """
    #version 330 core

    out vec4 FragColor;
    uniform vec3 color;
    
    void main()
    {
        FragColor = vec4(color, 1.0);
    }
        
    """

    window = pyogl_utils.create_window(width=width, height=height, is_hidden=not show_window)
    if not show_window:
        fbo, color_texture, rbo = pyogl_utils.create_fbo(width, height)
    glEnable(GL_DEPTH_TEST)
    vertices, faces = mesh_d["vertices"],  mesh_d["faces"]
    num_triangles = faces.shape[0]
    num_indices = num_triangles*3

    # model matrix
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1 * center)

    # view matrix
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 2.5 * model_radius
    camera_pos = np.array([0.0, 0.0, camera_distance])
    view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos)

    # projection matrix
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.1,
                                                       far_clip=10000000)
    # inv_matr = glm.inverse(projection_matr * view_matr * model_matr)

    vao, vertex_buffer, face_buffer, colors_buffer = pyogl_utils.get_mesh_vbo_vao(vertices, faces, uv_coords=None)
    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad, f_shad)
    buffers = [vao, vertex_buffer, face_buffer]

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    color_loc = glGetUniformLocation(shader_program, "color")

    glUseProgram(shader_program)
    glBindVertexArray(vao)

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matr))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matr))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matr))
    glUniform3f(color_loc, 1.0, 0.5, 0.2)  # Example orange color

    pyogl_utils.error_check()
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glfw.set_window_size_callback(window, window_size_callback)

    while not glfw.window_should_close(window):
        if not show_window:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        color_buffer_im = pyogl_utils.get_color_buffer(0, 0, width, height)
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
    if window:
        glfw.destroy_window(window)
    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)

    return color_buffer_im



if __name__ == "__main__":
   import PyOpenGL.geom_gen as geom_gen
   import Visualization.frame_utils as frame_utils
   cube_d = geom_gen.gen_cube(10, 10, 10)
   num_vertices = cube_d["vertices"].shape[0]
   cube_d["colors"] = np.random.randint(0, 256, (num_vertices, 3)).astype(float)/255.0
   im1 = render_mesh_point_color(cube_d, 1200, 1000, show_window=False)
   frame_utils.show_im(im1)
   im2 = render_untextured_mesh(cube_d, show_window=False)
   frame_utils.show_im(im2)
