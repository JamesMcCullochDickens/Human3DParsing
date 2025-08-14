from dataclasses import dataclass

import glfw
import glm
from OpenGL.GL import *
import numpy as np

import PyOpenGL.pyopengl_utils as pyogl_utils
import PC_Utils.pc_utils as pc_utils
import PyOpenGL.geom_gen as geom_gen

mouse_click_pos = None
selected_point_idx = None
is_dragging = False
last_cursor_pos = None
camera = None
sphere_right_clicked = False


v_shad = """
#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;

out vec3 fragColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(vertex_position, 1.0);
    fragColor = vertex_color; 
}
"""

# Fragment shader now supports override_color
frag_shad = """
#version 330 core

in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""

@dataclass
class PointMovingCamera:
    points: np.ndarray
    colors: np.ndarray
    aspect_ratio: float
    fov: float = 45.0
    near_clip: float = 0.1
    far_clip: float = 100.0
    camera_speed: float = 0.05

    def __post_init__(self):
        self.center = self._compute_center()
        self.model_matrix = self._compute_model_matrix()
        self.radius = self._compute_radius()
        self.camera_distance = 2.5 * self.radius
        self.position = glm.vec3(0.0, 0.0, self.camera_distance)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.camera_target =  glm.vec3(0.0, 0.0, 0.0)
        self.pitch = 0.0  # rotation around X-axis
        self.yaw = 0.0  # rotation around Y-axis
        self.roll = 0.0  # rotation around Z-axis
        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
        self._compute_inv_matr()

    def _compute_center(self):
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        center = (min_coords + max_coords) / 2.0
        return center


    def _compute_model_matrix(self):
        translation = -1 * self.center
        return pyogl_utils.get_model_matrix_from_trs(translation=translation)


    def _compute_radius(self):
        return pc_utils.get_vertex_radius(self.points)


    def rotate_x(self, angle_deg):
        angle_rad = glm.radians(angle_deg)

        forward = glm.normalize(self.camera_target - self.position)
        right = glm.normalize(glm.cross(forward, self.up))

        # Rotate the forward and up vectors around the right (local x) axis
        q_pitch = glm.angleAxis(angle_rad, right)

        rotated_forward = glm.normalize(q_pitch * forward)
        self.up = glm.normalize(q_pitch * self.up)

        self.camera_target = self.position + rotated_forward
        self.update_view_matrix()


    def rotate_y(self, angle_deg):
        angle_rad = glm.radians(angle_deg)

        forward = glm.normalize(self.camera_target - self.position)
        # Rotate around the local up axis
        q_yaw = glm.angleAxis(angle_rad, self.up)

        rotated_forward = glm.normalize(q_yaw * forward)
        right = glm.normalize(glm.cross(rotated_forward, self.up))
        self.up = glm.normalize(glm.cross(right, rotated_forward))

        self.camera_target = self.position + rotated_forward
        self.update_view_matrix()


    def rotate_z(self, angle_deg):
        angle_rad = glm.radians(angle_deg)

        forward = glm.normalize(self.camera_target - self.position)
        q_roll = glm.angleAxis(angle_rad, forward)

        self.up = glm.normalize(q_roll * self.up)
        self.update_view_matrix()


    def _compute_view_matrix(self):
        return glm.lookAt(self.position, self.camera_target, self.up)


    def _compute_projection_matrix(self):
        return pyogl_utils.get_perspective_matr(
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            near_clip=self.near_clip,
            far_clip=self.far_clip
        )


    def update_view_matrix(self):
        self.view_matrix = self._compute_view_matrix()
        self._compute_inv_matr()

    def process_keyboard(self, direction):
        forward = glm.normalize(self.camera_target - self.position)
        right = glm.normalize(glm.cross(forward, self.up))

        # Move forward or backward
        if direction == 'UPWARD':
            self.position += self.up * self.camera_speed
            self.camera_target += self.up * self.camera_speed
        elif direction == 'DOWNWARD':
            self.position -= self.up * self.camera_speed
            self.camera_target -= self.up * self.camera_speed

        # Move left or right, keeping direction fixed
        elif direction == 'LEFT':
            self.position -= right * self.camera_speed
            self.camera_target -= right * self.camera_speed
        elif direction == 'RIGHT':
            self.position += right * self.camera_speed
            self.camera_target += right * self.camera_speed

        self.update_view_matrix()

    def _compute_inv_matr(self):
        self.inv_matr = glm.inverse(self.projection_matrix * self.view_matrix * self.model_matrix)

    def zoom_in(self, amount=1.0):
        self.fov = max(5.0, self.fov - amount)  # Prevent excessively narrow FOV
        self._update_projection_matrix()

    def zoom_out(self, amount=1.0):
        self.fov = min(120.0, self.fov + amount)  # Prevent excessively wide FOV
        self._update_projection_matrix()

    def _update_projection_matrix(self):
        self.projection_matrix = self._compute_projection_matrix()
        self._compute_inv_matr()


def get_sample_pcd_d():
    points = np.random.rand(100, 3).astype(float)
    colors = np.random.randint(low=0, high=255, size=(100, 3)).astype(float)
    colors /= 255.0
    mesh_d = {"points": points, "colors": colors}
    return mesh_d


def window_size_callback(window, width, height):
    glViewport(0, 0, width, height)
    print(f"Window resized to: {width} x {height}")


def mouse_button_callback(window, button, action, mods):
    global mouse_click_pos, is_dragging, selected_point_idx, last_cursor_pos, sphere_right_clicked
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            mouse_click_pos = (x, y)
            last_cursor_pos = None
            is_dragging = True
        elif action == glfw.RELEASE:
            is_dragging = False
            selected_point_idx = None
            last_cursor_pos = None
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        x, y = glfw.get_cursor_pos(window)
        if action == glfw.PRESS:
            win_w, win_h = glfw.get_framebuffer_size(window)
            idx = get_closest_point_from_mouse_click(
                camera.points, x, y, win_w, win_h, camera.inv_matr
            )
            if idx is not None:
                add_sphere_at_point(camera, idx)
            sphere_right_clicked = True


def add_sphere_at_point(camera, point_idx, radius=0.01, lat=5, lon=5):
    base_point = camera.points[point_idx]

    sphere_vertices = geom_gen.gen_sphere_vertices(radius, lat, lon)
    sphere_vertices = np.array(sphere_vertices, dtype=np.float32)

    # Translate to selected point
    sphere_vertices += base_point

    # Append to existing buffers
    camera.points = np.vstack([camera.points, sphere_vertices])
    camera.points = np.ascontiguousarray(camera.points).astype(np.float32)

    # Color is handled separately, but to keep VAO sizes consistent:
    num_new = sphere_vertices.shape[0]
    sphere_colors = np.tile([1.0, 0.0, 0.0], (num_new, 1))  # temp color (red)

    camera.colors = np.vstack([camera.colors, sphere_colors])
    camera.colors = np.ascontiguousarray(camera.colors).astype(np.float32)


def cursor_position_callback(window, xpos, ypos):
    global is_dragging, selected_point_idx, last_cursor_pos, mouse_click_pos

    if is_dragging and selected_point_idx is not None:
        last_cursor_pos = mouse_click_pos
        mouse_click_pos = (xpos, ypos)


def key_callback(window, key, scancode, action, mods):
    global camera

    if action in (glfw.PRESS, glfw.REPEAT):
        if key == glfw.KEY_W:
            camera.process_keyboard("UPWARD")
        elif key == glfw.KEY_S:
            camera.process_keyboard("DOWNWARD")
        elif key == glfw.KEY_A:
            camera.process_keyboard("LEFT")
        elif key == glfw.KEY_D:
            camera.process_keyboard("RIGHT")
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):  # "+" key
            camera.zoom_in(amount=2.0)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):  # "-" key
            camera.zoom_out(amount=2.0)

            # Rotation around X-axis
        elif key == glfw.KEY_Z:
            camera.rotate_x(2.0)
        elif key == glfw.KEY_X:
            camera.rotate_x(-2.0)

            # Rotation around Y-axis
        elif key == glfw.KEY_C:
            camera.rotate_y(2.0)
        elif key == glfw.KEY_V:
            camera.rotate_y(-2.0)

            # Rotation around Z-axis
        elif key == glfw.KEY_B:
            camera.rotate_z(2.0)
        elif key == glfw.KEY_N:
            camera.rotate_z(-2.0)


def get_ray_from_mouse(x: int, y: int, win_width: int, win_height: int, inv_mvp):
    ndc_x = (2.0 * x) / win_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / win_height
    ndc_near = glm.vec4(ndc_x, ndc_y, -1.0, 1.0)
    ndc_far = glm.vec4(ndc_x, ndc_y, 1.0, 1.0)

    world_near = inv_mvp * ndc_near
    world_near /= world_near.w
    world_far = inv_mvp * ndc_far
    world_far /= world_far.w

    ray_origin = np.array([world_near.x, world_near.y, world_near.z])
    ray_dir = np.array([world_far.x, world_far.y, world_far.z]) - ray_origin
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_origin, ray_dir


def get_closest_point_from_mouse_click(points: np.ndarray, x: int, y: int, win_width: int, win_height: int, inv_mvp):
    ray_origin, ray_dir = get_ray_from_mouse(x, y, win_width, win_height, inv_mvp)
    v = points - ray_origin
    t = np.dot(v, ray_dir)
    closest_points = ray_origin + np.outer(t, ray_dir)
    dists = np.linalg.norm(points - closest_points, axis=1)
    best_idx = np.argmin(dists)
    return best_idx


def get_closest_point_from_mouse_click_kd_tree(points: np.ndarray, x: int, y: int, win_width: int, win_height: int, inv_mvp, kdtree):
    ray_origin, ray_dir = get_ray_from_mouse(x, y, win_width, win_height, inv_mvp)

    # Search nearby points using a bounding sphere along the ray
    # Choose a small radius and a long enough segment along the ray
    max_ray_t = 100.0  # how far along the ray we look
    radius = 0.05      # tolerance around the ray
    num_samples = 100  # number of points along the ray

    # Sample points along the ray
    t_vals = np.linspace(0, max_ray_t, num_samples)
    ray_samples = ray_origin + np.outer(t_vals, ray_dir)

    # Query points within radius of any sample along the ray
    indices = set()
    for pt in ray_samples:
        nearby = kdtree.query_ball_point(pt, r=radius)
        indices.update(nearby)

    # If no nearby points found, return None
    if not indices:
        print("No points found near ray.")
        return None

    # Compute actual distances for the small subset of candidates
    candidates = np.array(list(indices))
    candidate_points = points[candidates]
    v = candidate_points - ray_origin
    t = np.dot(v, ray_dir)
    closest_pts = ray_origin + np.outer(t, ray_dir)
    dists = np.linalg.norm(candidate_points - closest_pts, axis=1)
    best_local_idx = np.argmin(dists)
    return candidates[best_local_idx]


def ray_pick_pcd(pcd_d: dict, height: int = 600, width: int = 800) -> None:
    global frag_shad, selected_point_idx, mouse_click_pos, \
        v_shad, camera, last_cursor_pos, sphere_right_clicked

    aspect_ratio = width / height
    v_shad_fp_ = v_shad
    f_shad_fp_ = frag_shad

    window = pyogl_utils.create_window(width=width, height=height)
    points, colors = pcd_d["points"], pcd_d["colors"]
    num_points = points.shape[0]

    camera = PointMovingCamera(points=points, aspect_ratio=aspect_ratio, colors=colors)

    vao, points_buffer, colors_buffer = pyogl_utils.get_pcd_vao(points, colors)
    shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(v_shad_fp_, f_shad_fp_)
    buffers = [vao, points_buffer, colors_buffer]

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")

    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(camera.model_matrix))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(camera.view_matrix))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(camera.projection_matrix))

    pyogl_utils.error_check()
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)
    glPointSize(20.0)

    while not glfw.window_should_close(window):
        num_points = camera.points.shape[0]
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw all points (default color)
        glDrawArrays(GL_POINTS, 0, num_points)
        glfw.swap_buffers(window)
        glfw.poll_events()

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(camera.view_matrix))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(camera.projection_matrix))
        camera._compute_inv_matr()

        win_w, win_h = glfw.get_framebuffer_size(window)

        if is_dragging is True and selected_point_idx is None:
            selected_point_idx = get_closest_point_from_mouse_click(points, mouse_click_pos[0],
                                                                    mouse_click_pos[1], win_w, win_h,
                                                                    camera.inv_matr)

        if is_dragging and selected_point_idx is not None:
            x, y = glfw.get_cursor_pos(window)
            if last_cursor_pos is not None and last_cursor_pos != (x, y):
                win_w, win_h = glfw.get_framebuffer_size(window)
                ray_origin, ray_dir = get_ray_from_mouse(x, y, win_w, win_h, camera.inv_matr)
                t = np.dot(points[selected_point_idx] - ray_origin, ray_dir)
                new_point_pos = ray_origin + t * ray_dir
                points[selected_point_idx] = new_point_pos

                vertices = np.ascontiguousarray(points, dtype=np.float32)
                glBindBuffer(GL_ARRAY_BUFFER, points_buffer)
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        if sphere_right_clicked:
            sphere_right_clicked = False
            glBindBuffer(GL_ARRAY_BUFFER, points_buffer)
            glBufferData(GL_ARRAY_BUFFER, camera.points.nbytes, camera.points, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, colors_buffer)
            glBufferData(GL_ARRAY_BUFFER, camera.colors.nbytes, camera.colors, GL_STATIC_DRAW)

        #glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(camera.projection_matrix))

    if window:
        glfw.destroy_window(window)
    pyogl_utils.cleanup_and_exit(vao, buffers, shader_program)

if __name__ == "__main__":
    sample_pcd_d = get_sample_pcd_d()
    ray_pick_pcd(sample_pcd_d, 1600, 1600)
