import glm


class InteractiveCamera:
    def __init__(self, position, target, up):
        self.position = glm.vec3(position)
        self.target = glm.vec3(target)
        self.up = glm.vec3(up)
        self.distance_to_target = glm.length(self.target - self.position)
        self.last_x = 0.0
        self.last_y = 0.0
        self.yaw = -90.0
        self.pitch = 0.0
        self.sensitivity = 0.05
        self.sensitivity_multiplier = 1.0
        self.zoom = 45.0
        self.camera_speed = 0.1
        self.is_left_clicked = False
        self.view_matrix = self.update_view_matrix()

    def update_view_matrix(self):
        direction = glm.normalize(self.target - self.position)
        right = glm.normalize(glm.cross(self.up, direction))
        self.up = glm.cross(direction, right)
        return glm.lookAt(self.position, self.target, self.up)

    def process_mouse_movement(self, x_offset, y_offset):
        x_offset *= self.sensitivity * self.sensitivity_multiplier
        y_offset *= self.sensitivity * self.sensitivity_multiplier
        self.yaw += x_offset
        self.pitch += y_offset
        # Corrected direction calculation
        direction = glm.vec3(
            glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch)),
            glm.sin(glm.radians(self.pitch)),
            glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        )
        self.target = self.position + glm.normalize(direction)
        self.view_matrix = self.update_view_matrix()

    def process_mouse_scroll(self, y_offset):
        self.zoom -= y_offset
        self.zoom = max(1.0, min(45.0, self.zoom))

    def process_keyboard(self, direction):
        forward = glm.normalize(self.target - self.position)
        right = glm.normalize(glm.cross(forward, self.up))

        # Move forward or backward
        if direction == 'FORWARD':
            self.position += self.up * self.camera_speed
            self.target +=  self.up * self.camera_speed
        elif direction == 'BACKWARD':
            self.position -=  self.up * self.camera_speed
            self.target -=  self.up * self.camera_speed

        # Move left or right, keeping direction fixed
        elif direction == 'LEFT':
            self.position -= right * self.camera_speed
            self.target -= right * self.camera_speed
        elif direction == 'RIGHT':
            self.position += right * self.camera_speed
            self.target += right * self.camera_speed

        self.view_matrix = self.update_view_matrix()
