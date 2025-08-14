import numpy as np

def gen_sphere_vertices(radius: float, num_latitudes: int, num_longitudes) -> np.ndarray:
    theta = np.linspace(0, np.pi, num_latitudes)
    phi = np.linspace(0, 2 * np.pi, num_longitudes)
    theta, phi = np.meshgrid(theta, phi)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    vertices = vertices.astype(np.float32)
    return vertices


def gen_sphere_faces(num_latitudes: int, num_longitudes: int) -> np.ndarray:
    faces = []
    for i in range(num_latitudes - 1):
        for j in range(num_longitudes - 1):
            v1 = i * num_longitudes + j
            v2 = (i + 1) * num_longitudes + j
            v3 = (i + 1) * num_longitudes + (j + 1)
            v4 = i * num_longitudes + (j + 1)
            faces.extend([[v1, v2, v3], [v1, v3, v4]])
    return np.array(faces).astype(np.uint32)


def generate_fibonacci_sphere_vertices(samples: int, radius: float = 1.0) -> np.ndarray:
    indices = np.arange(samples)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (indices / float(samples - 1)) * 2.0
    radius_at_y = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y
    return np.column_stack((x, y, z)) * radius


def get_sphere_mesh(radius: float, num_latitudes:int, num_longitudes:int) -> dict:
    vertices = gen_sphere_vertices(radius, num_latitudes, num_longitudes)
    faces = gen_sphere_faces(num_latitudes, num_longitudes)
    return {"vertices": vertices, "faces": faces}


def rodrigues_matrix(axis, angle):
    # Normalize the rotation axis
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    # Extract components of the axis
    x, y, z = axis

    # Compute the skew-symmetric matrix K
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Compute the rotation matrix using the Rodrigues' formula
    I = np.eye(3)  # Identity matrix
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def get_cylinder_mesh_from_vectors(vec1_, vec2_, radius: float, num_polar_angles: int, num_height: int):
    vec1, vec2 = vec1_.copy(), vec2_.copy()
    direction = np.array(vec2) - np.array(vec1)
    length = np.linalg.norm(direction)
    if length == 0:
        eps = 0.0000001
        vec2 = vec1 + np.array([eps, eps, eps])
        print(f"Vectors 1 and 2 are the same, adding a small epsilon value to vec2.")
        direction = np.array(vec2) - np.array(vec1)
        length = np.linalg.norm(direction)
    cylinder_d = gen_cylinder_mesh(radius, length, num_polar_angles, num_height)
    source = np.array([0.0, 1.0, 0.0])
    target = direction / length
    rotation_axis = np.cross(source, target)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(source, target))
    R = rodrigues_matrix(rotation_axis, angle)
    cylinder_vertices = cylinder_d["vertices"]
    cylinder_vertices = (R@cylinder_vertices.T).T
    mid_point = vec1 + (vec2-vec1)/2
    cylinder_vertices += np.expand_dims(mid_point, axis=0)
    cylinder_d["vertices"] = cylinder_vertices
    return cylinder_d


def gen_n_sphere(num_spheres:int, radius: float, num_lattitudes:int, num_longitudes: int) -> dict:
    all_vertices, all_faces = [], []
    offset = 0
    for i in range(num_spheres):
        vertices = gen_sphere_vertices(radius, num_lattitudes, num_longitudes)
        vertices_offset = vertices.shape[0]
        faces = gen_sphere_faces(num_lattitudes, num_longitudes) + offset
        all_vertices.append(vertices)
        all_faces.append(faces)
        offset += vertices_offset
    return {"vertices": np.array(all_vertices).astype(np.float32),
            "faces": np.array(all_faces).astype(np.uint32)}


def gen_n_cylinder(num_cylinders:int, radius: float, height: float, num_polar_angles: int, num_height: int):
    all_vertices, all_faces = [], []
    for i in range(num_cylinders):
        vertices = gen_cylinder_vertices(radius, height, num_polar_angles, num_height)
        vertices_offset = vertices.shape[0]
        faces = gen_cylinder_faces(num_polar_angles, num_height) + i * vertices_offset
        all_vertices.append(vertices)
        all_faces.append(faces)
    return {"vertices": np.array(all_vertices).astype(np.float32),
            "faces": np.array(all_faces).astype(np.uint32)}



def gen_cylinder_vertices(radius: float, height: float, num_polar_angles: int, num_height: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_polar_angles, endpoint=False)
    height_vals = np.linspace(-height / 2, height / 2, num_height)
    angle, height_grid = np.meshgrid(angles, height_vals)
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    y = height_grid
    cylinder_vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    top_center = np.array([0, height / 2, 0])
    bottom_center = np.array([0, -height / 2, 0])
    cylinder_vertices = np.vstack([cylinder_vertices, top_center, bottom_center])

    return cylinder_vertices.astype(np.float32)


def gen_cylinder_faces(num_polar_angles: int, num_height: int) -> np.ndarray:
    faces = []
    for i in range(num_height - 1):
        for j in range(num_polar_angles):
            v1 = i * num_polar_angles + j
            v2 = (i + 1) * num_polar_angles + j
            v3 = (i + 1) * num_polar_angles + (j + 1) % num_polar_angles
            v4 = i * num_polar_angles + (j + 1) % num_polar_angles
            faces.extend([[v1, v2, v3], [v1, v3, v4]])

    # Add top and bottom caps
    top_center_index = num_polar_angles * num_height
    bottom_center_index = top_center_index + 1

    for j in range(num_polar_angles):
        # Top cap
        v1 = j
        v2 = (j + 1) % num_polar_angles
        faces.append([top_center_index, v1, v2])

        # Bottom cap
        v1 = num_polar_angles * (num_height - 1) + j
        v2 = num_polar_angles * (num_height - 1) + (j + 1) % num_polar_angles
        faces.append([bottom_center_index, v2, v1])

    return np.array(faces).astype(np.uint32)


def gen_cylinder_mesh(radius: float, height: float, num_polar_angles: int, num_height: int) -> dict:
    cyl_vertices =  gen_cylinder_vertices(radius, height, num_polar_angles, num_height)
    faces = gen_cylinder_faces(num_polar_angles, num_height)
    return {"vertices": cyl_vertices, "faces": faces}


def gen_cube_vertices(height: float, width: float, length: float) -> np.ndarray:
    vertices = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = -width / 2 if i % 2 == 0 else width / 2 # left to right
                y = height / 2 if j % 2 == 0 else -height / 2 # top to bottom
                z = length / 2 if k % 2 == 0 else -length / 2 # front to back
                vertices.append([x, y, z])
    return np.array(vertices, dtype=float)


def gen_cube_face():
    # TODO make sure this is correct in terms of the winding order
    return np.array([
        [0, 1, 3],
        [0, 3, 2],
        [4, 5, 7],
        [4, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 3, 7],
        [1, 7, 5],
        [2, 3, 7],
        [2, 7, 6],
        [0, 5, 5],
        [0, 5, 4]
    ])


def gen_cube(height: float, width: float, length: float):
    cube_vertices = gen_cube_vertices(height, width, length)
    cube_face = gen_cube_face()
    return {"vertices": cube_vertices, "faces": cube_face}


def fibonacci_sphere(num_points):
    points = []
    phi = np.pi * (3. - np.sqrt(5))  # golden angle in radians

    for i in range(num_points):
        z = 1 - (2 * i) / (num_points - 1)  # z-coordinate
        r = np.sqrt(1 - z * z)  # radius at z
        theta = phi * i  # longitude angle

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        points.append((x, y, z))

    return np.array(points)

if __name__ == "__main__":
    ...
