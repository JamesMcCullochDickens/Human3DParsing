from typing import Optional, List, Tuple, Union
from collections import defaultdict, deque

import os
import numpy as np
import torch
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
import random
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform

import utils.path_utils as path_utils
import utils.numpy_utils as numpy_utils
import Data_Processing.src.PointTransformerV3.serialization.default as serialize
import scipy as scp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def random_pc_jitter(pc: np.ndarray, lower_bound: Optional[int] = -1, upper_bound: Optional[int] = 1):
    rand_normal_noise = 0.01*np.random.normal(loc=0, scale=1.0, size=pc.shape).astype(pc.dtype)
    pc = pc + rand_normal_noise
    if lower_bound is not None:
        pc = np.clip(pc, a_min=lower_bound, a_max=upper_bound)
    return pc


def farthest_point_sampling(pc: np.ndarray, num_samples: int) -> np.ndarray:
    N, D = pc.shape
    if num_samples >= N:
        print(f" The number of requested samples:{num_samples} is larger than or equal to the "
              f"original point cloud size. Returning the original point cloud.")
        return pc
    sampled_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)  # Initialize distances to infinity

    # Choose the first point randomly
    sampled_indices[0] = np.random.randint(N)

    for i in range(1, num_samples):
        # Calculate distances from the last sampled point to all other points
        last_sampled_point = pc[sampled_indices[i - 1]]
        new_distances = np.linalg.norm(pc - last_sampled_point, axis=1)

        # Update the distances array to track the minimum distance to any sampled point
        distances = np.minimum(distances, new_distances)

        # Find the index of the point farthest from all sampled points
        farthest_index = np.argmax(distances)
        sampled_indices[i] = farthest_index

    return sampled_indices


@ torch.no_grad()
def farthest_point_sampling_torch(pc: torch.Tensor, num_samples: int) -> torch.Tensor:
    device = pc.device
    N, D = pc.shape
    if num_samples >= N:
        raise ValueError("The number of requested samples is larger than or equal to the original point cloud size. "
              "Returning the original point cloud.")
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    sampled_indices[0] = torch.randint(N, (1,), device=device)
    for i in range(1, num_samples):
        last_sampled_point = pc[sampled_indices[i - 1]].view(1, -1)
        new_distances = torch.cdist(last_sampled_point, pc).squeeze()
        distances = torch.minimum(distances, new_distances)
        farthest_index = torch.argmax(distances)
        sampled_indices[i] = farthest_index
    return sampled_indices


def batched_farthest_point_sampling(pc: torch.Tensor, num_samples: int) -> torch.Tensor:
    B, N, D = pc.shape

    if num_samples >= N:
        print(f" The number of requested samples:{num_samples} is larger than or equal to the "
              f"original point cloud size. Returning the original point cloud indices.")
        return torch.arange(N).expand(B, num_samples)  # Return original indices

    sampled_indices = torch.zeros((B, num_samples), dtype=torch.long, device=pc.device)
    distances = torch.full((B, N), float('inf'), device=pc.device)

    # Randomly choose the first point for each batch element
    sampled_indices[:, 0] = torch.randint(N, (B,), device=pc.device)

    for i in range(1, num_samples):
        last_sampled_points = pc[torch.arange(B), sampled_indices[:, i - 1]].unsqueeze(1)  # (B, 1, D)
        new_distances = torch.cdist(last_sampled_points, pc, p=2).squeeze(1)  # (B, N)
        distances = torch.min(distances, new_distances)
        sampled_indices[:, i] = torch.argmax(distances, dim=1)
    return sampled_indices


def uniform_sampling_var(pc: np.ndarray, num_samples: int):
    vars = np.array([np.var(pc[:, 0]), np.var(pc[:, 1]), np.var(pc[:, 2])])
    var_sort_idx = np.argsort(vars)
    pc_ = torch.tensor(pc)
    _, idx1 = torch.sort(pc_[:, var_sort_idx[0]], stable=True)
    pc_ = pc_[idx1]
    _, idx2 = torch.sort(pc_[:, var_sort_idx[1]], stable=True)
    pc_ = pc_[idx2]
    _, idx3 = torch.sort(pc_[:, var_sort_idx[2]], stable=True)
    final_idx = idx1[idx2[idx2]]
    final_idx = numpy_utils.evenly_spaced_samples(final_idx, num_samples)
    return final_idx


def uniform_sampling(pc: np.ndarray, num_samples: int, axis: int = 0):
    idx = np.argsort(pc[:, axis])
    idx = numpy_utils.evenly_spaced_samples(idx, num_samples)
    return idx


def hungarian_match_closest_k(n_points, m_points, k):
    """
    Finds a Hungarian matching for k closest unique points of m_points to n_points.

    Args:
        n_points: (N, 3) tensor of N points in 3D space.
        m_points: (M, 3) tensor of M points in 3D space.
        k: The number of closest matches to find (k <= min(N, M)).

    Returns:
        matches: (k, 2) tensor of indices, where matches[i] = [n_idx, m_idx]
                 indicates that n_points[n_idx] is matched to m_points[m_idx].
    """
    n_points_ = torch.tensor(n_points, dtype=torch.float)
    m_points_ = torch.tensor(m_points, dtype=torch.float)

    # Calculate pairwise distances (N x M)
    dist = torch.cdist(n_points_, m_points_)

    # Get top-k closest distances for each n_point and their indices
    topk_values, topk_indices = torch.topk(dist, k, dim=1, largest=False)

    # Prepare cost matrix for Hungarian matching
    cost_matrix = topk_values  # Use distances as costs

    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

    # Convert back to PyTorch tensors
    row_ind = torch.from_numpy(row_ind).to(cost_matrix.device)
    col_ind = torch.from_numpy(col_ind).to(cost_matrix.device)

    # Extract matching indices
    matches = torch.stack([row_ind, topk_indices[row_ind, col_ind]], dim=1)
    return matches[:, -1]



def torch_cdist(p1, p2):
    p1 = torch.tensor(p1).to(0)
    p2 = torch.tensor(p2).to(0)
    cdist = torch.cdist(p1, p2, p=2)
    return cdist.cpu().numpy()


def hungarian_matching_pcs(points1: np.ndarray, points2: np.ndarray, to_gpu: bool = False) -> List[Tuple]:
    if to_gpu:
        cost_matrix = torch_cdist(points1, points2)
    else:
        cost_matrix = cdist(points1, points2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))
    return matches


def windowed_hungarian_matching(pcs1: np.ndarray, pcs2: np.ndarray, to_gpu: bool = False):
    num_windows = pcs1.shape[0]
    points_per_window = pcs1.shape[1]
    matchings = []
    for window_num in range(num_windows):
        matching = np.array(hungarian_matching_pcs(pcs1[window_num], pcs2[window_num], to_gpu)) + (window_num*points_per_window)
        matchings.append(matching)
    matchings = np.concatenate([m for m in matchings], axis=0)
    return matchings



def get_motion_pc(pcd1: np.ndarray, pcd2: np.ndarray, matching_type: str = "default",
                  to_gpu: bool = False) -> np.ndarray:
    if matching_type == "default":
        matches = hungarian_matching_pcs(pcd1, pcd2, to_gpu)
        idx1 = np.array([matches[i][0] for i in range(pcd1.shape[0])], dtype=int)
        idx2 = np.array([matches[i][1] for i in range(pcd1.shape[0])], dtype=int)
    elif matching_type == "windowed_hungarian":
        matches = windowed_hungarian_matching(pcd1, pcd2)
        idx1 = matches[:, 0]
        idx2 = matches[:, 1]
    else:
        raise ValueError(f"Matching type {matching_type} not supported.")
    if matching_type == "windowed_hungarian":
        pcd1 = pcd1.reshape(-1, 3)
        pcd2 = pcd2.reshape(-1, 3)
    motion_pc = pcd2[idx2]-pcd1[idx1]
    return motion_pc


def pcs_to_grid_coord(pcs_, grid_size: float):
    if pcs_.ndim == 2:
        pcs = torch.unsqueeze(pcs_, dim=0)
    else:
        pcs = pcs_
    batch_size, num_points, C = pcs.shape
    flattened_feats = pcs.reshape(-1, C)
    pcs_f = flattened_feats[:, :3]
    grid_coord = torch.div(pcs_f - pcs_f.min(0)[0], grid_size, rounding_mode="trunc").int()
    return grid_coord


def get_z_code(grid_coords: torch.Tensor, depth:int = 8) -> torch.Tensor:
   z_code = serialize.z_order_encode(grid_coords, depth=depth)
   return z_code


def get_hilbert_z_code(grid_coords: torch.Tensor, depth:int = 8):
    hilbert_z_code = serialize.hilbert_encode(grid_coords, depth=depth)
    return hilbert_z_code


def sort_pcs_by_z_order(pcs_: Union[np.ndarray, torch.Tensor], grid_size: float,
                        to_gpu: bool = False, return_code: bool = False):
    if isinstance(pcs_, np.ndarray):
        pcs = torch.tensor(pcs_)
    else:
        pcs = pcs_.clone()
    if to_gpu:
        pcs = pcs.to(0)
    if pcs.ndim == 2:
        pcs = torch.unsqueeze(pcs, dim=0)
    batch_size, num_points, C = pcs.shape
    pcs_f = pcs.reshape(-1, C)
    grid_coord = pcs_to_grid_coord(pcs, grid_size)
    code = get_z_code(grid_coord, 8)
    code_sorted = torch.argsort(code)
    if to_gpu: # We end up back on the cpu and as a numpy tensor either way, but one requires an extra .cpu() call
        sorted_pcd = pcs_f[code_sorted].reshape(batch_size, num_points, C)
    else:
        sorted_pcd = pcs_f[code_sorted].reshape(batch_size, num_points, C).cpu().numpy()
    if return_code:
        return sorted_pcd, code_sorted
    else:
        return sorted_pcd


def sort_pcs_by_hilbert_z_order(pcs_: Union[np.ndarray, torch.Tensor], grid_size: float, to_gpu: bool = False):
    if isinstance(pcs_, np.ndarray):
        pcs = torch.tensor(pcs_)
    else:
        pcs = pcs_.clone()
    if to_gpu:
        pcs = pcs.to(0)
    if pcs.ndim == 2:
        pcs = torch.unsqueeze(pcs, dim=0)
    batch_size, num_points, C = pcs.shape
    pcs_f = pcs.reshape(-1, C)
    grid_coord = pcs_to_grid_coord(pcs, grid_size)
    code = get_hilbert_z_code(grid_coord, 8)
    code_sorted = torch.argsort(code)
    if to_gpu: # We end up back on the cpu and as a numpy tensor either way, but one requires an extra .cpu() call
        sorted_pcd = pcs_f[code_sorted].reshape(batch_size, num_points, C).cpu().numpy()
    else:
        sorted_pcd = pcs_f[code_sorted].reshape(batch_size, num_points, C).numpy()
    return sorted_pcd


def normalize_pcd_per_frame_min_max(pcd: np.ndarray) -> np.ndarray:
    ndim = pcd.ndim

    if ndim == 3:
        # pcd of shape Num_Frames, Num_Points, num_channels
        min_vals = np.min(pcd, axis=1, keepdims=True)  # Shape: (Num_Frames, 1, num_channels)
        max_vals = np.max(pcd, axis=1, keepdims=True)  # Shape: (Num_Frames, 1, num_channels)
        diff = np.where(max_vals - min_vals != 0, max_vals - min_vals, 1)
        pcd_norm = (pcd - min_vals) / diff
    else:
        # pcd of shape Num_Points, num_channels
        min_vals = np.min(pcd, axis=0, keepdims=True)  # Shape: (1, num_channels)
        max_vals = np.max(pcd, axis=0, keepdims=True)  # Shape: (1, num_channels)
        diff = np.where(max_vals - min_vals != 0, max_vals - min_vals, 1)
        pcd_norm = (pcd - min_vals) / diff

    return pcd_norm


def normalize_pcd_per_sequence_min_max(pcd: np.ndarray) -> np.ndarray:
    min_vals = np.amin(pcd, axis=(0, 1), keepdims=True)  # Shape (1, 1, C)
    max_vals = np.amax(pcd, axis=(0, 1), keepdims=True)  # Shape (1, 1, C)
    range_vals = max_vals - min_vals
    normalized_tensor = np.where(range_vals != 0, (pcd - min_vals) / range_vals, 1)
    return normalized_tensor



def get_pcd_centroids(pcds: np.ndarray, with_min_max_normalize: bool = False) -> np.ndarray:
    # pcd of shape Num_Frames, Num_Points, num_channels
    num_frames, num_points, num_channels = pcds.shape
    mean_vals = np.mean(pcds, axis=1, keepdims=True)  # Shape: (Num_Frames, 1, num_channels)
    mean_vals = mean_vals.reshape(num_frames, num_channels)
    if with_min_max_normalize:
        max_vals = np.max(mean_vals, axis=0)
        min_vals = np.min(mean_vals, axis=0)
        diff = max_vals-min_vals
        if np.sum(np.abs(diff).reshape(-1)) == 0:
            return mean_vals
        else:
            min_vals = np.tile(np.expand_dims(min_vals, axis=0), (num_frames, 1))
            diff = np.tile(np.expand_dims(diff, axis=0), (num_frames, 1))
            mean_vals = (mean_vals-min_vals)/diff
    return mean_vals


def get_centroid_displacements(centroids: np.ndarray) -> np.ndarray:
    # centroids of shape T, num_channels
    first_centroid = np.expand_dims(centroids[0], 0)
    displaced_centroids = centroids-first_centroid
    return displaced_centroids


def farthest_point_sampling_o3d(pc: np.ndarray, num_samples: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    sampled_pcd = pcd.farthest_point_down_sample(num_samples)
    sampled_points = np.asarray(sampled_pcd.points)
    tree = cKDTree(pc)
    _, sampled_indices = tree.query(sampled_points, k=1)
    return sampled_indices


def get_num_people(fps: List[str]):
    num_people = 1
    for fp in fps:
        person_num = get_person_num(fp) + 1
        num_people = max(num_people, person_num)
    return num_people


def get_frame_num(fp: str):
    base_fp = os.path.basename(fp)
    frame_num = int(base_fp.split("_")[1])
    return frame_num


def get_person_num(fp: str):
    base_fp = os.path.basename(fp).split(".")[0]
    person_num = int(base_fp.split("_")[2])
    return person_num


def seq_comparator(fp1: str, fp2: str):
    frame_num1, frame_num2, = get_frame_num(fp1), get_frame_num(fp2)
    if frame_num1 < frame_num2:
        return -1
    elif frame_num1 > frame_num2:
        return 1
    else:
        person_num1, person_num2 = get_person_num(fp1), get_person_num(fp2)
        if person_num1 < person_num2:
            return -1
        else:
            return 1


def get_sample_func(sample_type: Optional[str]):
    if sample_type == "uniform":
        return uniform_sampling
    elif sample_type == "farthest_point":
        # return farthest_point_sampling
        # return farthest_point_sampling_gpu
        return farthest_point_sampling_o3d # the fastest so far
    elif sample_type == "arm":
        return None
    else:
        raise ValueError(f"Sampling type {sample_type} not supported.")


def get_original_num_frames(fps):
    max_frame_num = -1
    for fp in fps:
        frame_num = int(get_frame_num(fp))
        max_frame_num = max(max_frame_num, frame_num)
    return max_frame_num + 1


def evenly_spaced_samples(data, num_samples):
    """Returns num_samples evenly spaced from data."""
    indices = np.linspace(0, len(data) - 1, num_samples).round().astype(int)
    return [data[i] for i in indices]


def read_ntu_seq_centroids(seq_fp: str):
    centroids = dict(np.load(seq_fp))["centroids"]  # shape T, M 3
    return centroids


def get_seq_info(seq_outer_fp: str) -> dict:
    seq_d_info = {}

    inner_fps = path_utils.join_inner_paths(seq_outer_fp)
    num_frames = get_original_num_frames(inner_fps)
    seq_d_info["num_frames"] = num_frames

    # get bad/empty frames
    keys = ["pc", "parts", "normals", "ir"]
    inner_fps_c = inner_fps.copy()
    bad_frames_p1 = []
    bad_frames_p2 = []
    # get bad frames
    for inner_fp in inner_fps_c:
        frame_num = get_frame_num(inner_fp)
        person_num = get_person_num(inner_fp)
        is_bad_frame = True
        for key in keys:
            if key in inner_fp:
                is_bad_frame = False
                break
        if is_bad_frame:
            if person_num == 0:
                bad_frames_p1.append(frame_num)
            else:
                bad_frames_p2.append(frame_num)
            inner_fps.remove(inner_fp)

    pc_fps = list(filter(lambda x: "pc" in x, inner_fps))
    num_people = get_num_people(inner_fps)
    num_valid_frames = len(pc_fps) // num_people
    seq_d_info["num_valid_frames"] = num_valid_frames
    seq_d_info["bad_frames_p1"] = bad_frames_p1
    if num_people == 2:
        seq_d_info["bad_frames_p2"] = bad_frames_p2
        bad_frames_p1_p2 = list(set(bad_frames_p1).intersection(bad_frames_p2))
        seq_d_info["num_bad_frames"] = len(bad_frames_p1_p2)
    else:
        seq_d_info["num_bad_frames"] = len(bad_frames_p1)
    if num_frames != 0:
        seq_d_info["bad_frames_ratio"] = seq_d_info["num_bad_frames"] / num_frames
    else:
        seq_d_info["bad_frames_ratio"] = 1
    seq_d_info["num_people"] = num_people
    seq_d_info["point_counts"] = []
    bad_frames_d = {0: bad_frames_p1, 1: bad_frames_p2}

    for frame_num in range(num_frames):
        seq_d_info["point_counts"].append([])
        for person_num in range(num_people):
            if frame_num not in bad_frames_d[person_num]:
                pc_fp = os.path.join(seq_outer_fp, "frame_" + str(frame_num) + "_" + str(person_num) + "_pc.npz")
                pc_d = dict(np.load(pc_fp))
                pc_person = pc_d["key"]
                seq_d_info["point_counts"][frame_num].append(pc_person.shape[0])
            else:
                seq_d_info["point_counts"][frame_num].append(0)
    return seq_d_info


def get_x_rotation(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.cos(angle)
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_angle, -sin_angle],
        [sin_angle, 0.0, cos_angle]
    ])
    return rotation_matrix


def get_y_rotation(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])
    return rotation_matrix


def get_z_rotation(theta: float):
    cos_angle, sin_angle = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [cos_angle, sin_angle, 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def get_random_z_rotation(range: Tuple[float, float] = (-np.pi, np.pi)) -> np.ndarray:
    theta = np.random.uniform(range[0], range[1])
    return get_z_rotation(theta)


def get_random_x_rotation(range: Tuple[float, float] = (-np.pi, np.pi)) -> np.ndarray:
    theta = np.random.uniform(range[0], range[1])
    return get_x_rotation(theta)


def get_random_y_rotation(rot_range: Tuple[float, float] = (-np.pi, np.pi)) -> np.ndarray:
    theta = np.random.uniform(rot_range[0], rot_range[1])
    rot_matr = get_y_rotation(theta)
    return rot_matr

def rotate_y_axis_by_theta(pc: np.ndarray, theta: float):
    # pc of shape N, 3
    rot_matr = get_y_rotation(theta)
    pc_ = batch_rotate(pc, rot_matr)
    return pc_


def rotate_x_axis_by_theta(pc: np.ndarray, theta: float):
    # pc of shape N, 3
    rot_matr = get_x_rotation(theta)
    pc_ = batch_rotate(pc, rot_matr)
    return pc_


def random_rotate_z(pc: np.ndarray) -> np.ndarray:
    # pc of shape T, M, num_points, 3
    rot_matr = get_random_z_rotation()
    rotated_pc = np.einsum('ab,mtvb->mtva', rot_matr, pc)
    return rotated_pc


def random_pc_rotate_y(pc: np.ndarray, return_rot_matr: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # N, 3
    rot_matr = get_random_y_rotation().astype(pc.dtype)
    rotated_pc = (rot_matr @ pc.T).T
    if return_rot_matr:
        return rotated_pc, rotated_pc
    else:
        return rotated_pc


def batch_rotate(A: np.ndarray, rot_matr: np.ndarray) -> np.ndarray:
    s = A.shape[:-1]
    A_flattened = A.reshape(-1, 3)
    rotated_A = (rot_matr @ A_flattened.T).T
    rotated_A = rotated_A.reshape(*s, 3)
    return rotated_A


def random_pc_rotate_y_batch(pc: np.ndarray, return_rot_matr: bool = False,
                             rot_range: Tuple[float, float] = (-np.pi, np.pi)) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # PC of shape [...., 3]
    rot_matr = get_random_y_rotation(rot_range=rot_range).astype(pc.dtype)
    rotated_pc = batch_rotate(pc, rot_matr)
    if return_rot_matr:
        return rotated_pc, rot_matr
    else:
        return rotated_pc


def random_euler_angles_rotate(pc: np.ndarray, return_rot_matr, rot_ranges: Tuple[Tuple[float, float],
    Tuple[float, float], Tuple[float, float]]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    x_rot_matr = get_random_x_rotation(rot_ranges[0])
    y_rot_matr = get_random_y_rotation(rot_ranges[1])
    z_rot_matr = get_random_z_rotation(rot_ranges[2])
    matrs = [x_rot_matr, y_rot_matr, z_rot_matr]
    random.shuffle(matrs)
    rot_matr = matrs[0] @ matrs[1] @ matrs[2]
    rotated_pc = batch_rotate(pc, rot_matr)
    if return_rot_matr:
        return rotated_pc, rot_matr
    else:
        return rotated_pc


def random_scale(pcds: np.ndarray, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    scale_fact = np.random.uniform(0.8, 1.2)
    pcds *= scale_fact
    if return_scale:
        return pcds, scale_fact
    else:
        return pcds



def normalize_person_pcds(pcds: np.ndarray, normalization_type: str = "per_frame") -> np.ndarray:
    assert isinstance(pcds, np.ndarray), f"Input must have type np.ndarray"
    assert pcds.size != 0, f"Empty point clouds."
    if pcds.ndim == 2:
        pcds = np.expand_dims(pcds, axis=0)

    # pcds of shape num_frames, num_people, num_points, num_channels
    num_points, num_channels = pcds.shape[-2:]
    shape_t = pcds.shape[:-2]
    if normalization_type == "per_frame":
        pcds_rs = pcds.reshape(-1, num_points, num_channels)
        pcds_rs = normalize_pcd_per_frame_min_max(pcds_rs)
        normalized_pcds = pcds_rs.reshape(*shape_t, num_points, num_channels)
    elif normalization_type == "per_sequence":
        pcds_rs = pcds.reshape(-1, num_points, num_channels)
        pcds_rs = normalize_pcd_per_sequence_min_max(pcds_rs)
        normalized_pcds = pcds_rs.reshape(*shape_t, num_points, num_channels)
    elif normalization_type == "per_sequence_per_person":
        if len(shape_t) > 1: # there exists a separate person axis
            num_people = shape_t[1]
            pcds_ = np.transpose(pcds, (1, 0, 2, 3))
            normalized_pcds = np.zeros_like(pcds_, dtype=pcds.dtype)
            for person_num in range(num_people):
                normalized_pcds[person_num] = normalize_pcd_per_sequence_min_max(pcds_[person_num])
            normalized_pcds = np.transpose(normalized_pcds, (1, 0, 2, 3))
        else:
            pcds_rs = pcds.reshape(-1, num_points, num_channels)
            pcds_rs = normalize_pcd_per_sequence_min_max(pcds_rs)
            normalized_pcds = pcds_rs.reshape(*shape_t, num_points, num_channels)
    else:
        raise ValueError(f"Normalization type {normalization_type} not supported.")
    return normalized_pcds


@torch.no_grad()
def ball_group2(pcs1: torch.Tensor, pcs2: torch.Tensor, dist: float, num_query_points: int,
               return_points: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bs, N, c = pcs1.shape
    _, M, c = pcs2.shape
    assert num_query_points+1 <= N, f"You are requesting more query points: {num_query_points} than input points: {N}."
    dists = torch.cdist(pcs1, pcs2, p=2.0)  # B, N, M
    closest_idx = torch.topk(dists, num_query_points+1, largest=False, sorted=True)[1]  # B, N, max_num_points
    closest_idx = closest_idx[:, :, 1:num_query_points+1]
    first_closest_idx = torch.unsqueeze(closest_idx[:, :, 0], dim=-1).repeat(1, 1, num_query_points)
    dists = torch.gather(input=dists, dim=-1, index=closest_idx)
    dists_mask = dists < dist
    closest_idx = torch.where(dists_mask, closest_idx, first_closest_idx)
    if return_points:
        closest_points_r = torch.unsqueeze(closest_idx, dim=-1).expand(-1, -1, -1, c)
        pc_ = pcs1.unsqueeze(1).expand(-1, N, -1, -1)
        closest_points = torch.gather(pc_, 2, closest_points_r)
        return closest_idx, closest_points
    else:
        return closest_idx, None


@torch.no_grad()
def get_pc_k_nn(pcs: torch.Tensor, k: int, return_points: bool = True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    is_2d = False
    if pcs.ndim == 2:
        pcs = torch.unsqueeze(pcs, dim=0)
        is_2d = True
    bs, N, c = pcs.shape
    assert k+1 <= N, f"You are requesting more query points: {k} than input points: {N}."
    dists = torch.cdist(pcs, pcs, p=2.0)  # B, N, N
    k_nn_idx = torch.topk(dists, k + 1, largest=False, sorted=True)[1]  # B, N, max_num_points
    k_nn_idx = k_nn_idx[:, :, 1:k + 1] # (B, N, k)
    if return_points:
        k_nn_idx_r = torch.unsqueeze(k_nn_idx, dim=-1).expand(-1, -1, -1, c)
        pc_ = pcs.unsqueeze(1).expand(-1, N, -1, -1)
        k_nn_points = torch.gather(pc_, 2, k_nn_idx_r)  # Shape (B, N, k, c)
        if is_2d:
            k_nn_idx = k_nn_idx[0]
            k_nn_points = k_nn_points[0]
        return k_nn_idx, k_nn_points
    else:
        if is_2d:
            k_nn_idx = k_nn_idx[0]
        return k_nn_idx, None


@torch.no_grad()
def get_pc_k_nn2(pcs1: torch.Tensor, pcs2: torch.Tensor, k: int, return_points: bool = True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bs, N, c = pcs1.shape
    _, M, c = pcs2.shape
    assert k+1 <= N, f"You are requesting more query points: {k} than input points: {N}."
    dists = torch.cdist(pcs1, pcs2, p=2.0)  # B, N, N
    k_nn_idx = torch.topk(dists, k + 1, largest=False, sorted=True)[1]  # B, N, max_num_points
    k_nn_idx = k_nn_idx[:, :, 1:k + 1]
    if return_points:
        k_nn_idx_r = torch.unsqueeze(k_nn_idx, dim=-1).expand(-1, -1, -1, c)
        pc_ = pcs1.unsqueeze(1).expand(-1, N, -1, -1)
        k_nn_points = torch.gather(pc_, 2, k_nn_idx_r)
        return k_nn_idx, k_nn_points
    else:
        return k_nn_idx, None


def compute_normals_open3d(pc: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5),)
    normals = np.asarray(pcd.normals).astype(np.float16)
    return normals


"""
def fps_cluster(pcs: torch.Tensor, num_samples: int, random_start: bool = True, return_points: bool = False):
    from torch_geometric.nn import fps
    device = pcs.device
    N, num_points, num_channels = pcs.size()
    batch = torch.arange(N, dtype=torch.long, device=device).repeat_interleave(num_points)
    sampled_indices = fps(pcs.view(N * num_points, num_channels), batch, ratio=num_samples / num_points, random_start=random_start)
    if return_points:
        sampled_points = pcs.view(N * num_points, num_channels)[sampled_indices].view(N, num_samples, num_channels)
        return sampled_indices, sampled_points
    else:
        sampled_indices = sampled_indices.view(N, num_samples)
        return sampled_indices, None
"""

def estimate_gaussian_curvature(points: np.ndarray, k=5) -> np.ndarray:
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k+1)  # Include the point itself

    curvatures = []
    for i in range(len(points)):
        neighbors = points[indices[i, 1:]]  # Exclude the point itself

        # Center the neighbors around the point of interest
        centered_neighbors = neighbors - points[i]

        # Fit a plane to the neighbors using linear regression
        reg = LinearRegression().fit(centered_neighbors[:, :2], centered_neighbors[:, 2])
        normal = np.array([reg.coef_[0], reg.coef_[1], -1])
        normal /= np.linalg.norm(normal)

        # Project the neighbors onto the plane
        projected_neighbors = neighbors - np.outer(np.dot(neighbors - points[i], normal), normal)

        # Center the projected neighbors
        centered_projected_neighbors = projected_neighbors - points[i]

        # Fit a quadratic surface to the projected neighbors
        X = centered_projected_neighbors[:, :2]
        Y = np.sum(centered_projected_neighbors ** 2, axis=1)
        reg = LinearRegression().fit(X, Y)

        # Extract the coefficients
        a, b, c, d, e = reg.coef_[0], reg.coef_[1], 0.5, 0.5, reg.intercept_ - np.sum(points[i]**2)

        # Compute the principal curvatures
        discriminant = (a - d)**2 + 4*b*c
        k1 = (a + d + np.sqrt(discriminant)) / 2
        k2 = (a + d - np.sqrt(discriminant)) / 2

        # Compute the Gaussian curvature
        curvature = k1 * k2
        curvatures.append(curvature)

    return np.array(curvatures)


def add_noisy_points(pc: np.ndarray, num_points: int):
    # pc of shape (Num_Points, Num_channels)

    x_min, y_min, z_min = np.min(pc, axis=0)
    x_max, y_max, z_max = np.max(pc, axis=0)

    random_points = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(num_points, 3),
    )

    pc = np.concatenate([pc, random_points], axis=0)
    return pc


def save_pcd_from_arr(pc: np.ndarray, colors: Optional[np.ndarray]=None, save_fp: str = "./model.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename=save_fp, pointcloud=pcd)


def save_o3dmesh(mesh, save_fp):
    o3d.io.write_triangle_mesh(filename=save_fp, mesh=mesh)


def get_random_colors(N: int, with_normalization: bool = False) -> np.ndarray:
    colors = np.random.randint(0, 256, size=(N, 3), dtype=np.uint8)
    if with_normalization:
        colors = colors.astype(float)
        colors /= 255.0
    return colors


def remove_zero_points(pcd: np.ndarray):
    zero_point_indices = np.all(pcd == 0, axis=-1)
    pcd_ = pcd[~zero_point_indices]
    return pcd_


def smallest_distance(points):
    distances = pdist(points)
    dist_matrix = squareform(distances)
    np.fill_diagonal(dist_matrix, np.inf)
    min_distance = np.min(dist_matrix)
    return min_distance

def combine_meshes(mesh_list):
    combined_vertices = []
    combined_triangles = []
    combined_colors = []

    vertex_offset = 0

    for mesh in mesh_list:
        combined_vertices.append(np.asarray(mesh.vertices))
        combined_triangles.append(np.asarray(mesh.triangles) + vertex_offset)
        if mesh.has_vertex_colors():
            combined_colors.append(np.asarray(mesh.vertex_colors))
        else:
            # If a mesh doesn't have colors, assign a default color (e.g., gray)
            combined_colors.append(np.full((len(mesh.vertices), 3), [0.5, 0.5, 0.5]))

        vertex_offset += len(mesh.vertices)

    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(combined_vertices))
    combined_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(combined_triangles))
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(combined_colors))

    return combined_mesh



def pcd_to_sphere_meshes(pcd: Union[o3d.geometry.PointCloud, np.ndarray], colors: Optional[np.ndarray]=None,
                         sphere_radius: Optional[float] = None, save_fp: str = None, with_viz: bool = False):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd = pcd_from_points(pcd, colors)
    if sphere_radius is None:
        sphere_radius = smallest_distance(pcd.points)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mesh_list = []
    for point, color in zip(points, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        mesh_list.append(sphere)
    pcd_mesh = combine_meshes(mesh_list)
    if with_viz:
        o3d.visualization.draw_geometries([pcd_mesh])
    if save_fp:
        save_o3dmesh(pcd_mesh, save_fp)
    return pcd_mesh


def mask_pcds(pcds_: np.ndarray, point_mask_pct: float,
              is_contiguous: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    pcds = pcds_.copy()
    mask_idx_vals = []
    is_single_frame = False
    if pcds.ndim == 2: # make sure we have 3 dimension, Time, Num_Points, Num_Channels
        pcds = np.array([pcds])
        is_single_frame = True
    num_frames, num_points, num_channels = pcds.shape
    num_samples = int((1-point_mask_pct)*num_points)
    # A separate masked number of values per frame
    masked_pcds = []
    for frame_num in range(num_frames):
        num_points_l = list(range(num_points))
        if not is_contiguous:
            mask_idx = np.array(random.choices(num_points_l, k=num_samples), dtype=int)
        else:
            rand_valid_start = random.randint(0, num_points-num_samples)
            mask_idx = np.array(num_points_l[rand_valid_start:rand_valid_start+num_samples], dtype=int)
        masked_pcds.append(pcds[frame_num, mask_idx, :])
        mask_idx_vals.append(mask_idx)
    if is_single_frame:
        masked_pcds = masked_pcds[0]
        mask_idx_vals = mask_idx_vals[0]
    return np.array(masked_pcds, dtype=pcds_.dtype), np.array(mask_idx_vals, dtype=int)


def pcd_from_points(points: np.ndarray, colors: Optional[np.ndarray]=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if np.max(colors) > 1.0:
            colors_ = colors.astype(float)
            colors_ /= 255.0
        else:
            colors_ = colors.copy()
        pcd.colors = o3d.utility.Vector3dVector(colors_)
    return pcd


def get_pc_bounds(pc: np.ndarray) -> Tuple:
    x_min, x_max = np.min(pc[..., 0]), np.max(pc[..., 0])
    y_min, y_max = np.min(pc[..., 1]), np.max(pc[..., 1])
    z_min, z_max = np.min(pc[..., 2]), np.max(pc[..., 2])
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_rand_mesh_vals(N: int = 500):
    vertices = np.random.rand(N, 3)
    faces = np.random.choice(N, size=(N // 3, 3), replace=False)
    uv_coordinates = np.random.rand(N, 2)
    return vertices, faces, uv_coordinates


def get_xy_grid(width: int, height: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
    coords = np.concatenate([x, y], axis=-1)
    return coords


def get_3d_delaunay_triangulation_faces(points: np.ndarray):
    tri = Delaunay(points)
    faces = tri.convex_hull
    return faces


def get_vertex_radius(vertices: np.ndarray):
    center = np.mean(vertices, axis=0)
    distances = np.linalg.norm(vertices - center, axis=1)
    radius = np.max(distances)
    return radius


def get_rand_o3d_mesh(N: int = 500, use_convex_hull: bool = True,
                      with_display: bool = True):
    points, faces, _ = get_rand_mesh_vals(N)
    vertices =  o3d.utility.Vector3dVector(points)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = vertices
    if use_convex_hull:
        faces = o3d.utility.Vector3iVector(get_3d_delaunay_triangulation_faces(points))
        mesh.triangles = faces
    else:
        mesh.triangles = o3d.utility.Vector3iVector(faces)
    #mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_coordinates)
    mesh.vertex_colors = o3d.utility.Vector3dVector(get_random_colors(N, with_normalization=True))
    #mesh.remove_duplicated_triangles()
    #mesh.remove_unreferenced_vertices()
    if with_display:
        o3d.visualization.draw_geometries([mesh])
    return mesh


def closest_point_idx(pcd_a, pcd_b):
    # for each point in pcd_a, find the closest point in pcd_b
    # should return a tensor of shape N, where N is the number of points in pcd_a
    tree = cKDTree(pcd_b)
    _, closest_point_indices = tree.query(pcd_a)
    return closest_point_indices


@torch.no_grad()
def closest_point_idx_pt(pcd_a, pcd_b, to_gpu: bool = True,
                         device: int = 0):
    if to_gpu:
        pcd_a = torch.tensor(pcd_a, device=device) if not isinstance(pcd_a, torch.Tensor) else pcd_a.to(device)
        pcd_b = torch.tensor(pcd_b, device=device) if not isinstance(pcd_b, torch.Tensor) else pcd_b.to(device)
    else:
        pcd_a = torch.tensor(pcd_a) if not isinstance(pcd_a, torch.Tensor) else pcd_a
        pcd_b = torch.tensor(pcd_b) if not isinstance(pcd_b, torch.Tensor) else pcd_b
    distances = torch.cdist(pcd_a.unsqueeze(0), pcd_b.unsqueeze(0)).squeeze(0)
    closest_point_indices = torch.argmin(distances, dim=1).cpu()
    return closest_point_indices


def closest_point_idx_pt_batched(pcd_a, pcd_b, batch_size=1024, to_gpu=True, device=0):
    if to_gpu:
        dtype = torch.float16 if pcd_a.dtype == torch.float16 or pcd_b.dtype == torch.float16 else torch.float32
        pcd_a = torch.tensor(pcd_a, dtype=dtype, device=device) if not isinstance(pcd_a, torch.Tensor) else pcd_a.to(
            device, dtype=dtype)
        pcd_b = torch.tensor(pcd_b, dtype=dtype, device=device) if not isinstance(pcd_b, torch.Tensor) else pcd_b.to(
            device, dtype=dtype)
    else:
        dtype = torch.float32  # Default to single precision for CPU
        pcd_a = torch.tensor(pcd_a, dtype=dtype) if not isinstance(pcd_a, torch.Tensor) else pcd_a.to(dtype=dtype)
        pcd_b = torch.tensor(pcd_b, dtype=dtype) if not isinstance(pcd_b, torch.Tensor) else pcd_b.to(dtype=dtype)

    # Initialize closest point indices
    closest_point_indices = torch.zeros(pcd_a.shape[0], dtype=torch.long, device=pcd_a.device)

    # Process pcd_b in batches
    min_distances = torch.full((pcd_a.shape[0],), float('inf'), device=pcd_a.device)
    for start_idx in range(0, pcd_b.shape[0], batch_size):
        end_idx = start_idx + batch_size
        pcd_b_batch = pcd_b[start_idx:end_idx]

        # Compute distances for the current batch
        distances = torch.cdist(pcd_a.unsqueeze(0), pcd_b_batch.unsqueeze(0)).squeeze(0)

        # Update minimum distances and closest indices
        batch_min_distances, batch_min_indices = torch.min(distances, dim=1)
        update_mask = batch_min_distances < min_distances
        min_distances[update_mask] = batch_min_distances[update_mask]
        closest_point_indices[update_mask] = batch_min_indices[update_mask] + start_idx

    return closest_point_indices.cpu()


def o3d_mesh_uvs_to_vertex_color(mesh):
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


def is_zero_pcd(pcd: np.array) -> bool:
    f_pcd = pcd.reshape(-1)
    return np.sum(np.abs(f_pcd)) == 0


def get_pcd_zero_frames(pcd: np.ndarray) -> np.ndarray:
    # Assumes the pcd is of shape num_frames, num_points, num_features
    pcd_sum = np.sum(np.abs(pcd), axis=(-2, -1))
    zero_frames = np.argwhere(pcd_sum == 0)
    return zero_frames.flatten()


def get_pcd_non_zero_frames(pcd: np.ndarray) -> np.ndarray:
    # Assumes the pcd is of shape num_frames, num_points, num_features
    pcd_sum = np.sum(np.abs(pcd), axis=(-2, -1))
    zero_frames = np.argwhere(pcd_sum != 0)
    return zero_frames.flatten()


def remove_zero_pcd_frames(pcd: np.ndarray) -> np.ndarray:
    # Assumes the pcd is of shape num_frames, num_points, num_features
    non_zero_frames = get_pcd_non_zero_frames(pcd)
    pcd_filt = pcd[non_zero_frames]
    return pcd_filt


def kabsch_pca_person_normalize(pcd: np.ndarray) -> np.ndarray:
    centered_vertices = pcd - np.mean(pcd, axis=0)
    covariance_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    to_align = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    try:
        rot_matr, _ = scp.spatial.transform.Rotation.align_vectors(to_align, eigenvectors.T)
        rot_matr = rot_matr.as_matrix()
    except:  # Fallback in case of failure
        return pcd

    transformed_pcd = (rot_matr @ centered_vertices.T).T
    transformed_pcd = transformed_pcd - np.mean(transformed_pcd, axis=0)

    covar_matr2 = np.cov(transformed_pcd, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covar_matr2)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    eigen2, eigen3 = eigenvectors[:, 1], eigenvectors[2]
    #axis_swap = np.linalg.norm(eigen2-np.array([1.0, 0.0, 0.0])) > np.linalg.norm(eigen3-np.array([1.0, 0.0, 0.0]))
    axis_swap = np.argmax(np.abs(eigen2)) != 0
    if axis_swap:
        eigenvectors[:, [0, 2]] = eigenvectors[:, [2, 0]]
        transformed_pcd[:, [0, 2]] = transformed_pcd[:, [2, 0]]

    if np.dot(eigenvectors[:, 0], [0, 1, 0]) < 0:
        transformed_pcd[:, 0] *= -1

    if np.dot(eigenvectors[:, 1], [1, 0, 0]) < 0:
        transformed_pcd[:, 1] *= -1

    if np.dot(eigenvectors[:, 2], [0, 0, -1]) > 0:
        transformed_pcd[:, 2] *= -1

    return transformed_pcd


def normalize_pcd_unit_sphere(pcd: np.ndarray) -> np.ndarray:
    centroid = pcd.mean(axis=0)
    points = pcd - centroid
    scale = np.linalg.norm(points, axis=1).max()
    return points / (scale+1e-6)


def normalize_batch_pcd_unit_sphere(pcd: np.ndarray) -> np.ndarray:
    centroid = pcd.mean(axis=1, keepdims=True)
    centered = pcd - centroid
    scale = np.linalg.norm(centered, axis=2, keepdims=True).max(axis=1, keepdims=True)
    return centered / (scale + 1e-8)


def get_mesh_connected_components(faces: np.ndarray):
    # Step 1: Build adjacency list using a single pass
    adjacency = defaultdict(list)
    for face in faces:
        adjacency[face[0]].extend([face[1], face[2]])
        adjacency[face[1]].extend([face[0], face[2]])
        adjacency[face[2]].extend([face[0], face[1]])

    # Step 2: Convert lists to sets
    for key in adjacency:
        adjacency[key] = set(adjacency[key])

    # Step 3: Find connected components using a non-recursive DFS
    visited = set()
    components = []

    def explore(start):
        """Non-recursive DFS to find all connected vertices."""
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(adjacency[node] - visited)
        return component

    # Iterate through all vertices
    for vertex in adjacency:
        if vertex not in visited:
            components.append(np.array(explore(vertex), dtype=np.int32))

    return components


if __name__ == "__main__":
    ...