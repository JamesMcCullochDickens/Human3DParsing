

from collections import defaultdict
from typing import *
import sys
import data_fps
sys.path.append(data_fps.data_fps["M2FP_Parsing_path"])
sys.path.append(data_fps.data_fps["M2FP_Parsing_path"]+"/detectron2")

from OpenGL.GL import *
import glm 
import glfw
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import hdbscan
import M2FP.itw_inference as itw_inference
import torch

import src.PyOpenGL.pyopengl_utils as pyogl_utils
import utils.pc_utils as pc_utils
import utils.frame_utils as frame_utils
import src.Sapiens.part_seg_api as part_seg_api
import utils.pc_vis as pc_vis



sapiens_semantic_mappingv1 = {
    0: 0,    # Background -> Background
    1: 1,    # Apparel -> General_Clothing
    2: 2,    # Face_Neck -> Face
    3: 3,    # Hair -> Hair
    4: 4,    # Left_Foot -> Left_Foot
    5: 5,    # Left_Hand -> Left_Hand
    6: 6,    # Left_Lower_Arm -> Left_Arm
    7: 7,    # Left_Lower_Leg -> Left_Leg
    8: 4,    # Left_Shoe -> Left_Foot
    9: 4,    # Left_Sock -> Left_Foot
    10: 6,   # Left_Upper_Arm -> Left_Arm
    11: 7,   # Left_Upper_Leg -> Left_Leg
    12: 8,   # Lower_Clothing -> Lower_Clothing
    13: 9,   # Right_Foot -> Right_Foot
    14: 10,  # Right_Hand -> Right_Hand
    15: 11,  # Right_Lower_Arm -> Right_Arm
    16: 12,  # Right_Lower_Leg -> Right_Leg
    17: 9,   # Right_Shoe -> Right_Foot
    18: 9,   # Right_Sock -> Right_Foot
    19: 11,  # Right_Upper_Arm -> Right_Arm
    20: 12,  # Right_Upper_Leg -> Right_Leg
    21: 13,  # Torso -> Torso
    22: 14,  # Upper_Clothing -> Upper_Clothing
    23: 2,   # Lower_Lip -> Face
    24: 2,   # Upper_Lip -> Face
    25: 2,   # Lower_Teeth -> Face
    26: 2,   # Upper_Teeth -> Face
    27: 2    # Tongue -> Face
}


sapiens_semantic_mappingv2 = {
    0: 0,    # Background -> Background
    1: 1,    # Apparel -> General_Clothing
    2: 2,    # Face_Neck -> Face
    3: 3,    # Hair -> Hair
    4: 4,    # Left_Foot -> Left_Foot
    5: 5,    # Left_Hand -> Left_Hand
    6: 6,    # Left_Lower_Arm -> Left_Arm
    7: 7,    # Left_Lower_Leg -> Left_Leg
    8: 4,    # Left_Shoe -> Left_Foot
    9: 4,    # Left_Sock -> Left_Foot
    10: 6,   # Left_Upper_Arm -> Left_Arm
    11: 7,   # Left_Upper_Leg -> Left_Leg
    12: 8,   # Lower_Clothing -> Lower_Clothing
    13: 9,   # Right_Foot -> Right_Foot
    14: 10,  # Right_Hand -> Right_Hand
    15: 11,  # Right_Lower_Arm -> Right_Arm
    16: 12,  # Right_Lower_Leg -> Right_Leg
    17: 9,   # Right_Shoe -> Right_Foot
    18: 9,   # Right_Sock -> Right_Foot
    19: 11,  # Right_Upper_Arm -> Right_Arm
    20: 12,  # Right_Upper_Leg -> Right_Leg
    21: 13,  # Torso -> Torso
    22: 14,  # Upper_Clothing -> Upper_Clothing
    23: 15,   # Lower_Lip -> Lip
    24: 15,   # Upper_Lip -> Lip
    25: 16,   # Lower_Teeth -> Teeth
    26: 16,   # Upper_Teeth -> Teeth
    27: 17    # Tongue -> Tongue
}



# New semantic categories after merging
new_sapiens_label_namesv1 = {
    0: "Background",
    1: "Apparel",
    2: "Face_Neck",
    3: "Hair",
    4: "Left_Foot_Complete",
    5: "Left_Hand",
    6: "Left_Arm",
    7: "Left_Leg",
    8: "Lower_Clothing",
    9: "Right_Foot_Complete",
    10: "Right_Hand",
    11: "Right_Arm",
    12: "Right_Leg",
    13: "Torso",
    14: "Upper_Clothing"
}

new_sapiens_label_namesv2 = {
    0: "Background",
    1: "Apparel",
    2: "Face_Neck",
    3: "Hair",
    4: "Left_Foot_Complete",
    5: "Left_Hand",
    6: "Left_Arm",
    7: "Left_Leg",
    8: "Lower_Clothing",
    9: "Right_Foot_Complete",
    10: "Right_Hand",
    11: "Right_Arm",
    12: "Right_Leg",
    13: "Torso",
    14: "Upper_Clothing",
    15: "Lip",
    16: "Teeth",
    17: "Tongue",
}


cihp_label_names = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Gloves", 4: "Sunglasses",
    5: "UpperClothes", 6: "Dress", 7: "Coat", 8: "Socks", 9: "Pants",
    10: "Torso-skin", 11: "Scarf", 12: "Skirt", 13: "Face",
    14: "Left-arm", 15: "Right-arm", 16: "Left-leg", 17: "Right-leg",
    18: "Left-shoe", 19: "Right-shoe"
}


def get_m2fp_inference_on_im(im: Union[np.ndarray, str], with_resize:bool = False,
                             with_visualize: bool = False,  resize_tuple: Optional[Tuple[int, int]] = None,
                             device_id: int = 0) -> np.ndarray:
    
    d = itw_inference.inference_on_im(im, with_resize=with_resize,
                                      resize_tuple=resize_tuple,
                                      device_id=device_id)
    part_mask = d["part_mask"]
    if with_visualize:
        color_im = itw_inference.mask_to_color_im(part_mask)
        itw_inference.show_im(color_im)
    return part_mask


def assign_vertex_to_triangle_id(num_vertices, faces) -> np.ndarray:
    N = num_vertices
    F = faces.shape[0]
    ids = np.full(N, np.iinfo(np.int32).max, dtype=np.int32)  # Initialize with max value
    flat_faces = faces.ravel()
    face_indices = np.repeat(np.arange(F), 3)
    np.minimum.at(ids, flat_faces, face_indices)
    return ids.astype(np.uint32)


def assign_vertices_to_visible_triangles(faces, visible_faces):
    visible_vertex_indices = faces[visible_faces].flatten()
    visible_vertices = np.unique(visible_vertex_indices)
    is_vertex_visible = np.zeros(faces.max() + 1, dtype=bool)
    is_vertex_visible[visible_vertices] = True
    is_face_visible = np.any(is_vertex_visible[faces], axis=1)
    visible_vertices_connected = np.unique(faces[is_face_visible])
    return visible_vertices_connected


def assign_vertex_labels(seg_label_mask, triangle_im_ids, vertex_triangle_ids):
    vertex_triangle_ids = vertex_triangle_ids.astype(int)
    seg_label_mask_f = seg_label_mask.ravel()
    triangle_im_ids_f = triangle_im_ids.ravel()

    non_zero_part_mask = seg_label_mask_f != 0
    seg_label_mask_f = seg_label_mask_f[non_zero_part_mask]
    triangle_im_ids_f = triangle_im_ids_f[non_zero_part_mask]

    # Get unique triangle IDs and their corresponding labels
    unique_triangle_ids, first_indices = np.unique(triangle_im_ids_f, return_index=True)
    triangle_labels = seg_label_mask_f[first_indices]
    triangle_d = {}
    for i, id in enumerate(unique_triangle_ids):
        triangle_d[id] = triangle_labels[i]

    num_vertices = vertex_triangle_ids.shape[0]
    vertex_labels = np.full(num_vertices, 0, dtype=int)
    for i in range(num_vertices):
        triangle_idx = vertex_triangle_ids[i]
        if triangle_idx in triangle_d.keys():
            vertex_labels[i] = triangle_d[triangle_idx]

    return vertex_labels


def vote_based_vertex_labeling(all_seg_masks, visibility_textures, vertex_triangle_ids, num_vertices, num_classes):
    vertex_labels = np.zeros(num_vertices, dtype=np.int32)
    vertex_votes = defaultdict(lambda: np.zeros(num_classes + 1, dtype=np.int32))
    for seg_mask, vis_texture in zip(all_seg_masks, visibility_textures):
        valid_mask = vis_texture.ravel() >= 0
        if not np.any(valid_mask):
            continue
            
        valid_triangles = vis_texture.ravel()[valid_mask]
        valid_labels = seg_mask.ravel()[valid_mask]
        
        keep_mask = (valid_labels > 0) & (valid_triangles >= 0)
        if not np.any(keep_mask):
            continue
            
        valid_triangles = valid_triangles[keep_mask]
        valid_labels = valid_labels[keep_mask]
        
        unique_pairs, counts = np.unique(
            np.stack([valid_triangles, valid_labels]),
            axis=1, return_counts=True
        )
        
        batch_size = 1000
        for start_idx in range(0, len(unique_pairs[0]), batch_size):
            end_idx = start_idx + batch_size
            batch_triangles = unique_pairs[0, start_idx:end_idx]
            batch_labels = unique_pairs[1, start_idx:end_idx]
            batch_counts = counts[start_idx:end_idx]
            
            vertices_mask = np.isin(vertex_triangle_ids, batch_triangles)
            vertices_indices = np.where(vertices_mask)[0]
            
            triangle_to_label = dict(zip(batch_triangles, zip(batch_labels, batch_counts)))
            
            for vertex_idx in vertices_indices:
                triangle_idx = vertex_triangle_ids[vertex_idx]
                if triangle_idx in triangle_to_label:
                    label, count = triangle_to_label[triangle_idx]
                    vertex_votes[vertex_idx][label] += count
    
    for vertex_idx in range(num_vertices):
        if vertex_idx in vertex_votes:
            votes = vertex_votes[vertex_idx]
            if np.any(votes):
                vertex_labels[vertex_idx] = np.argmax(votes)
    
    return vertex_labels


def vote_based_vertex_labeling_np(all_seg_masks, visibility_textures, vertex_triangle_ids, num_vertices, num_classes):
    # Initialize vote storage as a NumPy array instead of a dictionary
    vertex_votes = np.zeros((num_vertices, num_classes + 1), dtype=np.int32)

    # Flatten triangle ID array for faster lookup
    vertex_triangle_ids = np.array(vertex_triangle_ids)

    for mask_num, (seg_mask, vis_texture) in enumerate(zip(all_seg_masks, visibility_textures)):
        valid_mask = vis_texture.ravel() >= 0
        if not np.any(valid_mask):
            continue

        valid_triangles = vis_texture.ravel()[valid_mask]
        valid_labels = seg_mask.ravel()[valid_mask]

        keep_mask = (valid_labels > 0) & (valid_triangles >= 0)
        if not np.any(keep_mask):
            continue

        valid_triangles = valid_triangles[keep_mask]
        valid_labels = valid_labels[keep_mask]

        # Count occurrences of each (triangle, label) pair
        unique_pairs, counts = np.unique(
            np.stack([valid_triangles, valid_labels]),
            axis=1, return_counts=True
        )

        # Create mapping of triangle to label votes
        triangle_to_label_counts = {t: (l, c) for t, l, c in zip(*unique_pairs, counts)}

        # Optimize vertex voting process using NumPy indexing
        valid_vertices_mask = np.isin(vertex_triangle_ids, unique_pairs[0])
        valid_vertices = np.where(valid_vertices_mask)[0]

        # Assign votes in batch
        for vertex_idx in valid_vertices:
            triangle_idx = vertex_triangle_ids[vertex_idx]
            if triangle_idx in triangle_to_label_counts:
                label, count = triangle_to_label_counts[triangle_idx]
                vertex_votes[vertex_idx, label] += count

    # Assign labels based on maximum votes
    vertex_labels = np.argmax(vertex_votes, axis=1)

    return vertex_labels


def vote_based_vertex_labeling_torch(all_seg_masks, visibility_textures, vertex_triangle_ids, num_vertices, num_classes, device='cuda'):
    # Move data to the GPU
    vertex_triangle_ids = torch.tensor(vertex_triangle_ids, device=device, dtype=torch.int32)
    vertex_votes = torch.zeros((num_vertices, num_classes + 1), device=device, dtype=torch.int32)

    for seg_mask, vis_texture in zip(all_seg_masks, visibility_textures):
        vis_texture_torch = torch.tensor(vis_texture.ravel(), device=device, dtype=torch.int32)
        seg_mask_torch = torch.tensor(seg_mask.ravel(), device=device, dtype=torch.int32)

        valid_mask = vis_texture_torch >= 0
        if not valid_mask.any():
            continue

        valid_triangles = vis_texture_torch[valid_mask]
        valid_labels = seg_mask_torch[valid_mask]

        keep_mask = (valid_labels > 0) & (valid_triangles >= 0)
        if not keep_mask.any():
            continue

        valid_triangles = valid_triangles[keep_mask]
        valid_labels = valid_labels[keep_mask]

        # Count occurrences of each (triangle, label) pair
        stacked_pairs = torch.stack([valid_triangles, valid_labels], dim=0)
        unique_pairs, indices, counts = torch.unique(stacked_pairs, dim=1, return_counts=True, return_inverse=True)

        # Map triangles to their labels and counts
        triangle_to_label_counts = dict(zip(unique_pairs[0].tolist(), zip(unique_pairs[1].tolist(), counts.tolist())))

        # Find relevant vertices in batch
        valid_vertices_mask = torch.isin(vertex_triangle_ids, unique_pairs[0])
        valid_vertices = torch.nonzero(valid_vertices_mask, as_tuple=True)[0]

        # Assign votes using PyTorch indexing
        for vertex_idx in valid_vertices:
            triangle_idx = vertex_triangle_ids[vertex_idx].item()
            if triangle_idx in triangle_to_label_counts:
                label, count = triangle_to_label_counts[triangle_idx]
                vertex_votes[vertex_idx, label] += count

    # Assign labels based on max votes
    vertex_labels = torch.argmax(vertex_votes, dim=1)

    return vertex_labels.cpu().numpy()



def detect_clusters(points, eps=0.02, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_


def detect_clusters_h(points, min_cluster_size=50, min_samples=None):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(points)
    return clusterer.labels_


def detect_clusters_ro(points, min_cluster_size=50, min_samples=None):
    """
    Detect clusters using HDBSCAN with error handling for small point sets.
    
    Args:
        points: numpy array of points to cluster
        min_cluster_size: minimum cluster size
        min_samples: number of samples in neighborhood
    
    Returns:
        numpy array of cluster labels
    """
    # If we have very few points, return all points as noise (-1)
    if len(points) <= min_cluster_size:
        return np.full(len(points), -1)
    
    # Adjust min_samples if needed
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Ensure min_samples isn't larger than number of points
    min_samples = min(min_samples, len(points) - 1)
    
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(min_cluster_size, len(points)),
            min_samples=min_samples
        )
        clusterer.fit(points)
        return clusterer.labels_
    except ValueError as e:
        print(f"Warning: Clustering failed with error: {str(e)}")
        print(f"Number of points: {len(points)}")
        # Return all points as noise in case of failure
        return np.full(len(points), -1)


def denoise_nn(vertices_l, vertices_nl, labels_l):
    """Safe nearest neighbor denoising that handles empty arrays"""
    if len(vertices_l) == 0 or len(vertices_nl) == 0:
        # Return array of zeros if either input is empty
        return np.zeros(len(vertices_nl), dtype=int)
    
    tree = cKDTree(vertices_l)
    _, indices = tree.query(vertices_nl, k=1)
    nearest_labels = labels_l[indices]
    return nearest_labels


def denoise_knn(vertices_l, vertices_nl, labels_l, k_neighbors=5):
    # Check if there are any labeled vertices
    if len(vertices_l) == 0:
        # Return array of zeros with same shape as vertices_nl
        return np.zeros(len(vertices_nl), dtype=int)
   
    tree = cKDTree(vertices_l)
    # Get k nearest neighbors
    distances, indices = tree.query(vertices_nl, k=k_neighbors)
    
    # Get labels for all k neighbors
    neighbor_labels = labels_l[indices]

    # Use majority voting along axis 1
    from scipy.stats import mode
    final_labels = mode(neighbor_labels, axis=1)[0].flatten()
    
    return final_labels


def denoising_by_label_dbscan_new(vertices, vertex_labels, eps=0.05, min_samples=10,
                                  k_neighbors=5, with_sapiens=False, is_sapiens_v2=False):
    """
    Denoise semantic labels using DBSCAN clustering for each label class.
    
    Args:
        vertices: Nx3 array of vertex positions
        vertex_labels: N array of semantic labels
        eps: DBSCAN epsilon parameter (cluster distance threshold)
        min_samples: DBSCAN min_samples parameter
        k_neighbors: Number of neighbors for KNN label propagation
        
    Returns:
        tuple: (vertices, cleaned vertex_labels)
    """
    # Input validation
    if vertices.shape[0] != vertex_labels.shape[0]:
        raise ValueError("Vertices and labels must have same length")
    
    unique_labels = np.unique(vertex_labels)

    if with_sapiens:
        if is_sapiens_v2:
            label_names = sapiens_semantic_mappingv2
        else:
            label_names = sapiens_semantic_mappingv1
    else:
        label_names = cihp_label_names

    # Process each semantic label
    for target_label in unique_labels:
        # Extract points for current label
        target_label_indices = np.where(vertex_labels == target_label)[0]
        target_label_points = vertices[target_label_indices]
        
        # Skip processing if too few points
        if len(target_label_indices) < min_samples + 1:
            print(f"\nLabel {target_label} ({label_names.get(target_label, 'Unknown')}): "
                  f"Only {len(target_label_indices)} points, skipping")
            continue
        
        # Detect clusters within current label
        clusters = detect_clusters(target_label_points, eps=eps, min_samples=min_samples)
        unique_clusters = np.unique(clusters)
        
        # Count points per cluster
        cluster_sizes = []
        for c in unique_clusters:
            if c != -1:  # Skip noise points which are labeled as -1
                size = np.sum(clusters == c)
                cluster_sizes.append((c, size))

        # Report cluster statistics
        if not cluster_sizes:
            continue
            
        largest_cluster_id = max(cluster_sizes, key=lambda x: x[1])[0]
        
        # Process smaller clusters
        for cluster_id in unique_clusters:
            if cluster_id == largest_cluster_id or cluster_id == -1:
                continue

            # Get cluster points and indices
            cluster_mask = clusters == cluster_id
            cluster_points = target_label_points[cluster_mask]
            cluster_global_indices = target_label_indices[cluster_mask]

            # Create a mask for all vertices NOT in the current cluster
            non_cluster_mask = np.ones(len(vertices), dtype=bool)
            non_cluster_mask[cluster_global_indices] = False
            non_cluster_points = vertices[non_cluster_mask]
            non_cluster_labels = vertex_labels[non_cluster_mask]

            # Get new labels from KNN
            denoised_labels = denoise_knn(
                non_cluster_points, cluster_points, 
                non_cluster_labels, k_neighbors=k_neighbors)
            
            # Apply new labels
            vertex_labels[cluster_global_indices] = denoised_labels

        # Process noise points
        noise_mask = clusters == -1
        noise_points = target_label_points[noise_mask]
        noise_global_indices = target_label_indices[noise_mask]

        if len(noise_points) > 0:
            # Create mask for non-noise points
            non_noise_mask = np.ones(len(vertices), dtype=bool)
            non_noise_mask[noise_global_indices] = False
            non_noise_points = vertices[non_noise_mask]
            non_noise_labels = vertex_labels[non_noise_mask]

            # Get new labels from KNN
            denoised_noise_labels = denoise_knn(
                non_noise_points, noise_points, 
                non_noise_labels, k_neighbors=k_neighbors)

            # Apply new labels
            for idx, new_label in zip(noise_global_indices, denoised_noise_labels):
                vertex_labels[idx] = new_label
                
            # Simple report of changes
            unique_noise_labels = np.unique(denoised_noise_labels)
            print(f"    Relabeled to: ", end="")
            for l in unique_noise_labels:
                count = np.sum(denoised_noise_labels == l)
                print(f"{label_names.get(l, 'Unknown')}({l}): {count:,} ", end="")
            print()
    
    # M2FP-specific processing: Handle small label groups and socks
    if not with_sapiens:
        # Handle small label groups (less than 500 points)
        special_labels = [1, 3, 4, 6, 9, 10, 11, 12, 14, 15, 16, 17]
        print("\nProcessing small label groups:")
        
        for label in special_labels:
            if label in unique_labels:
                label_indices = np.where(vertex_labels == label)[0]
                label_count = len(label_indices)
                
                if 0 < label_count < 500:
                    print(f"  Small label group: {label} ({label_names.get(label, 'Unknown')}) - {label_count:,} points")
                    
                    # Extract points
                    special_points = vertices[label_indices]
                    other_mask = np.ones(len(vertex_labels), dtype=bool)
                    other_mask[label_indices] = False
                    other_points = vertices[other_mask]
                    other_labels = vertex_labels[other_mask]
                    
                    # Apply KNN to reassign labels
                    majority_labels = denoise_knn(
                        other_points, special_points,
                        other_labels, k_neighbors=k_neighbors
                    )
                    
                    # Apply new labels
                    vertex_labels[label_indices] = majority_labels
                    
                    # Simple report of changes
                    unique_relabels = np.unique(majority_labels)
                    print(f"    Relabeled to: ", end="")
                    for l in unique_relabels:
                        count = np.sum(majority_labels == l)
                        print(f"{label_names.get(l, 'Unknown')}({l}): {count:,} ", end="")
                    print()


         # Specifically handle socks (label 8)
        if 8 in unique_labels:
            sock_indices = np.where(vertex_labels == 8)[0]
            sock_count = len(sock_indices)
            
            if sock_count > 0:
                print(f"\nProcessing socks (8): {sock_count:,} points")
                
                # Extract sock points
                sock_points = vertices[sock_indices]
                
                # Create masks for left/right shoes and legs
                left_shoe_mask = vertex_labels == 18
                right_shoe_mask = vertex_labels == 19
                left_leg_mask = vertex_labels == 16
                right_leg_mask = vertex_labels == 17
                
                # Check if we have any of these reference points
                has_left_refs = np.any(left_shoe_mask) or np.any(left_leg_mask)
                has_right_refs = np.any(right_shoe_mask) or np.any(right_leg_mask)
                
                if has_left_refs or has_right_refs:
                    # Combine reference points
                    left_ref_mask = left_shoe_mask | left_leg_mask
                    right_ref_mask = right_shoe_mask | right_leg_mask
                    
                    # Get reference points and their indices
                    left_ref_points = vertices[left_ref_mask]
                    right_ref_points = vertices[right_ref_mask]
                    
                    # Prepare arrays to store distances
                    left_distances = np.full(sock_count, np.inf)
                    right_distances = np.full(sock_count, np.inf)
                    
                    # Calculate distances
                    if len(left_ref_points) > 0:
                        tree_left = cKDTree(left_ref_points)
                        left_distances, _ = tree_left.query(sock_points, k=1)
                    
                    if len(right_ref_points) > 0:
                        tree_right = cKDTree(right_ref_points)
                        right_distances, _ = tree_right.query(sock_points, k=1)
                    
                    # Determine left vs right based on distances
                    left_mask = left_distances < right_distances
                    
                    # Assign new labels
                    new_sock_labels = np.full(sock_count, 19)  # Default to right shoe
                    new_sock_labels[left_mask] = 18  # Left shoe where appropriate
                    
                    # Apply new labels
                    vertex_labels[sock_indices] = new_sock_labels
                    
                    # Report changes
                    left_count = np.sum(left_mask)
                    right_count = sock_count - left_count
                    print(f"    Relabeled to: Left-shoe(18): {left_count:,} Right-shoe(19): {right_count:,}")
                else:
                    print("    No shoe or leg references found - socks not relabeled")

    else:
        ...

    return vertices, vertex_labels


def generate_human_specific_viewpoints_dense(camera_distance):
    elevations = [0, 30, -30]  # 0 is horizontal, positive is above
    azimuths = [0, 90, 180, 270]  # 0 is front, 90 is right side
    
    viewpoints = []
    for elev in elevations:
        for azim in azimuths:
            # Convert to radians
            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)
            
            # Calculate 3D position
            x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
            y = camera_distance * np.sin(elev_rad)
            z = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
            
            viewpoints.append(np.array([x, y, z]))
    
    return np.array(viewpoints)


def generate_human_specific_viewpoints_sparse(camera_distance):
    """Generate 8 high-coverage human-centric viewpoints."""
    elevations = [0, 30]  # Horizontal and slightly above
    azimuths = [0, 90, 180, 270]  # Front, sides, back

    viewpoints = []
    for elev in elevations:
        for azim in azimuths:
            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)

            x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
            y = camera_distance * np.sin(elev_rad)
            z = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)

            viewpoints.append(np.array([x, y, z]))

    return np.array(viewpoints)



def normalize_point_cloud_scale(vertices):
    """Normalize point cloud to have consistent scale."""
    # Center the vertices
    centered_vertices = vertices - np.mean(vertices, axis=0)
    # Compute the scale (maximum distance from center)
    scale = np.max(np.linalg.norm(centered_vertices, axis=1))
    # Normalize by this scale
    normalized_vertices = centered_vertices / scale
    return normalized_vertices


""" use this to get hands from sapines """
def integrate_hand_segmentation(primary_masks, secondary_masks):
    """
    1. Extract hands from secondary model (Sapiens)
    2. Remap sock labels to corresponding shoes
    
    Args:
        primary_masks: List of segmentation masks from M2FP model
        secondary_masks: List of segmentation masks from Sapiens model
    
    Returns:
        List of enhanced segmentation masks with hand segmentation
    """
    # Define new labels for hands (using values not present in primary segmentation)
    LEFT_HAND_LABEL = 21  # New label for left hand
    RIGHT_HAND_LABEL = 22  # New label for right hand
    
    print("\n=== Integrating Hand Segmentation from Sapiens Model ===")
    
    enhanced_masks = []
    
    for i, (primary_mask, secondary_mask) in enumerate(zip(primary_masks, secondary_masks)):
        # Create a copy of the primary mask to modify
        enhanced_mask = primary_mask.copy()


        # 2. HAND SEGMENTATION: Extract hand regions from secondary mask
        # Label 5 is Left_Hand, label 14 is Right_Hand in Sapiens model
        left_hand_mask = secondary_mask == 5   # Left_Hand in Sapiens
        right_hand_mask = secondary_mask == 14  # Right_Hand in Sapiens
        
        # Get arm regions from primary mask
        # Label 14 is Left-arm, label 15 is Right-arm in M2FP model
        left_arm_mask = primary_mask == 14
        right_arm_mask = primary_mask == 15
        
        # Calculate statistics for reporting
        left_hand_pixels = np.sum(left_hand_mask)
        right_hand_pixels = np.sum(right_hand_mask)
        left_arm_pixels = np.sum(left_arm_mask)
        right_arm_pixels = np.sum(right_arm_mask)
        
        print(f"\nView {i+1}:")
        print(f"  Left arm pixels: {left_arm_pixels:,}")
        print(f"  Right arm pixels: {right_arm_pixels:,}")
        print(f"  Left hand pixels from Sapiens: {left_hand_pixels:,}")
        print(f"  Right hand pixels from Sapiens: {right_hand_pixels:,}")
        
        # Apply the hand segmentation by directly replacing corresponding pixels
        # Only use hand pixels where arms are already detected for robustness
        if left_hand_pixels > 50:
            # Overlay left hand only where it intersects with left arm for robustness
            valid_left_hand = np.logical_and(left_hand_mask, left_arm_mask)
            if np.sum(valid_left_hand) > 20:  # At least 20 pixels of overlap
                enhanced_mask[valid_left_hand] = LEFT_HAND_LABEL
                print(f"  - Added left hand segmentation ({np.sum(valid_left_hand):,} pixels)")
        
        if right_hand_pixels > 50:
            # Overlay right hand only where it intersects with right arm for robustness
            valid_right_hand = np.logical_and(right_hand_mask, right_arm_mask)
            if np.sum(valid_right_hand) > 20:  # At least 20 pixels of overlap
                enhanced_mask[valid_right_hand] = RIGHT_HAND_LABEL
                print(f"  - Added right hand segmentation ({np.sum(valid_right_hand):,} pixels)")
        
        enhanced_masks.append(enhanced_mask)
    
    return enhanced_masks


""" use this to dictate from sapiens """
def remap_socks_and_extract_shoes(primary_masks, secondary_masks):
    """
    1. Remap sock labels to corresponding shoe labels in Sapiens model
    2. Extract shoe segmentations from Sapiens model after remapping
    
    In the Sapiens model:
    - 8: Left_Shoe
    - 9: Left_Sock
    - 17: Right_Shoe
    - 18: Right_Sock
    
    In the M2FP model (primary):
    - 18: Left_shoe
    - 19: Right_shoe
    
    Args:
        primary_masks: List of segmentation masks from M2FP model
        secondary_masks: List of segmentation masks from Sapiens model
    
    Returns:
        List of enhanced segmentation masks with Sapiens shoes
    """
    print("\n=== Remapping Socks and Extracting Shoes from Sapiens Model ===")
    
    # Define labels for shoe mapping
    M2FP_LEFT_SHOE = 18   # Target label in M2FP model
    M2FP_RIGHT_SHOE = 19  # Target label in M2FP model
    
    enhanced_masks = []
    
    # First remap socks to shoes in the Sapiens masks
    remapped_secondary_masks = []
    for i, mask in enumerate(secondary_masks):
        # Create a copy of the mask to modify
        remapped_mask = mask.copy()
        
        # Count original pixels before remapping
        left_sock_pixels = np.sum(mask == 9)
        right_sock_pixels = np.sum(mask == 18)
        left_shoe_pixels_orig = np.sum(mask == 8) 
        right_shoe_pixels_orig = np.sum(mask == 17)
        
        # Define the mappings: sock → shoe in Sapiens model
        mapping = {
            9: 8,    # Left_Sock → Left_Shoe
            18: 17,  # Right_Sock → Right_Shoe
        }
        
        # Apply the mappings
        for sock_label, shoe_label in mapping.items():
            sock_mask = mask == sock_label
            if np.any(sock_mask):
                remapped_mask[sock_mask] = shoe_label
        
        # Count remapped pixels
        remapped_left_shoe_pixels = np.sum(remapped_mask == 8)
        remapped_right_shoe_pixels = np.sum(remapped_mask == 17)
        
        print(f"\nView {i+1} sock remapping:")
        print(f"  Left: {left_sock_pixels:,} sock pixels → {remapped_left_shoe_pixels:,} shoe pixels (added {left_sock_pixels:,} pixels)")
        print(f"  Right: {right_sock_pixels:,} sock pixels → {remapped_right_shoe_pixels:,} shoe pixels (added {right_sock_pixels:,} pixels)")
        
        remapped_secondary_masks.append(remapped_mask)
    
    # Now extract shoes from remapped Sapiens masks and integrate into primary masks
    for i, (primary_mask, secondary_mask) in enumerate(zip(primary_masks, remapped_secondary_masks)):
        # Create a copy of the primary mask to modify
        enhanced_mask = primary_mask.copy()
        
        # Extract shoe regions from remapped Sapiens mask
        # Label 8 is Left_Shoe, label 17 is Right_Shoe in Sapiens model
        sapiens_left_shoe_mask = secondary_mask == 8  
        sapiens_right_shoe_mask = secondary_mask == 17
        
        # Count pixels for reporting
        m2fp_left_shoe_pixels = np.sum(primary_mask == M2FP_LEFT_SHOE)
        m2fp_right_shoe_pixels = np.sum(primary_mask == M2FP_RIGHT_SHOE)
        sapiens_left_shoe_pixels = np.sum(sapiens_left_shoe_mask)
        sapiens_right_shoe_pixels = np.sum(sapiens_right_shoe_mask)
        
        print(f"\nView {i+1} shoe comparison:")
        print(f"  M2FP Left shoe pixels: {m2fp_left_shoe_pixels:,}")
        print(f"  M2FP Right shoe pixels: {m2fp_right_shoe_pixels:,}")
        print(f"  Sapiens Left shoe pixels: {sapiens_left_shoe_pixels:,}")
        print(f"  Sapiens Right shoe pixels: {sapiens_right_shoe_pixels:,}")
        
        # Apply the shoe segmentation if there are enough pixels
        if sapiens_left_shoe_pixels > 100:
            enhanced_mask[sapiens_left_shoe_mask] = M2FP_LEFT_SHOE
            print(f"  - Replaced left shoe segmentation with Sapiens data ({sapiens_left_shoe_pixels:,} pixels)")
            
        if sapiens_right_shoe_pixels > 100:
            enhanced_mask[sapiens_right_shoe_mask] = M2FP_RIGHT_SHOE
            print(f"  - Replaced right shoe segmentation with Sapiens data ({sapiens_right_shoe_pixels:,} pixels)")
        
        enhanced_masks.append(enhanced_mask)
    
    return enhanced_masks


def print_mask_label_stats(masks, model_type="M2FP",
                           is_sapiens_v2: bool = False):
    """Print statistics about the segmentation masks"""
    print("\n" + "="*50)
    print(f"{model_type} REPROJECTION RESULTS - BEFORE PROCESSING")
    print("="*50)
    
    # Get the labels dictionary based on model type
    if model_type == "Sapiens":
        if is_sapiens_v2:
            label_dict = sapiens_semantic_mappingv2
        else:
            label_dict = sapiens_semantic_mappingv1
    else:
        label_dict = cihp_label_names
    
    # Count labels across all masks
    combined_counts = {}
    for i, mask in enumerate(masks):
        unique_labels, counts = np.unique(mask, return_counts=True)
        print(f"\nView {i+1} labels:")
        for label, count in zip(unique_labels, counts):
            if label == 0:  # Skip background
                continue
            print(f"  Label {label} ({label_dict.get(label, 'Unknown')}): {count:,} pixels")
            combined_counts[label] = combined_counts.get(label, 0) + count
    
    # Print total counts
    print("\nTotal counts across all views:")
    for label, count in sorted(combined_counts.items()):
        if label == 0:  # Skip background
            continue
        print(f"  Label {label} ({label_dict.get(label, 'Unknown')}): {count:,} pixels")
    print("="*50)


def mesh_triangle_visibility_and_seg(mesh_d, height: int, width: int,
                                     with_sapiens: bool = False, is_sapiens_v2: bool = False):
    visibility_v_shad = """
    #version 140
    in vec3 position;
    in float vertexID; // Pass the triangle ID as a per-vertex attribute

    flat out float triangleID;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
        triangleID = vertexID; // Forward the triangle ID to the fragment shader
    }
    """

    # Replace visibility_f_shad
    visibility_f_shad = """
    #version 140
    flat in float triangleID;   // Receive the triangle ID
    out float fragColor;

    void main() {
        fragColor = float(triangleID);
    }
    """

    # Replace render_v_shad
    render_v_shad = """
    #version 140
    in vec3 position;
    in vec2 texCoords; // UV coordinates

    out vec2 fragment_uv;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
        fragment_uv = texCoords;
    }
    """

    # Replace render_f_shad
    render_f_shad = """
    #version 140
    in vec2 fragment_uv;

    out vec4 fragColor;

    uniform sampler2D textureSampler;

    void main() {
        fragColor = texture2D(textureSampler, fragment_uv);
    }
    """

    vertices, faces = mesh_d["vertices"], mesh_d["faces"]
    uv_coords = mesh_d["uv_coords"]
    texture_arr = mesh_d["texture"]
    #texture_arr =  np.flipud(texture)[..., :3].astype(np.uint8)
    num_vertices = vertices.shape[0]
    num_triangles = faces.shape[0]
    vertex_ids = assign_vertex_to_triangle_id(num_vertices, faces)
    vertex_ids += 1
    vertices = pyogl_utils.np_to_fp32(vertices)
    vertex_ids = pyogl_utils.np_to_fp32(vertex_ids)
    faces = pyogl_utils.np_to_uint32(faces)
    uv_coords = pyogl_utils.np_to_fp32(uv_coords)

    # Window Creation
    window = pyogl_utils.create_window(width=width, height=height, is_hidden=True)

    # *** Visibility FrameBuffer ***
    vis_fbo = glGenFramebuffers(1)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if (status != GL_FRAMEBUFFER_COMPLETE):
        raise RuntimeError(f"Framebuffer not complete: {status}")
    glBindFramebuffer(GL_FRAMEBUFFER, vis_fbo)
    vis_texture = glGenTextures(1)
    if not vis_texture:
        raise RuntimeError("Failed to create color texture for frame buffer.")
    glBindTexture(GL_TEXTURE_2D, vis_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vis_texture, 0)
    vis_depth_rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, vis_depth_rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, vis_depth_rbo)

    # *** Rendering FrameBuffer ***
    render_fbo, render_texture, render_rbo = pyogl_utils.create_fbo(width, height)
    texture_id = pyogl_utils.numpy_array_to_texture_id(texture_arr)

    # Matrices
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    model_matr = pyogl_utils.get_model_matrix_from_trs(translation=-1 * center)
    model_radius = pc_utils.get_vertex_radius(vertices)
    camera_distance = 2.6 * model_radius
    
    # Generate spherical viewpoints
    if with_sapiens:
        viewpoints = generate_human_specific_viewpoints_dense(camera_distance)
    else:
        viewpoints = generate_human_specific_viewpoints_sparse(camera_distance)
    aspect_ratio = width / height
    projection_matr = pyogl_utils.get_perspective_matr(fov=45, aspect_ratio=aspect_ratio, near_clip=0.01, far_clip=1000)

    # Create Visibility VAO and VBOs
    visibility_vao = glGenVertexArrays(1)
    glBindVertexArray(visibility_vao)
    vertex_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    vertex_id_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_id_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_ids.nbytes, vertex_ids, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)

    triangle_ebo1 = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_ebo1)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    # global visibility_v_shad, visibility_f_shad
    visibility_shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(visibility_v_shad, visibility_f_shad)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create Render VAO and VBOs
    render_vao = glGenVertexArrays(1)
    glBindVertexArray(render_vao)
    vertex_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    uv_coords_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, uv_coords_vbo)
    glBufferData(GL_ARRAY_BUFFER, uv_coords.nbytes, uv_coords, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)

    triangle_ebo2 = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_ebo2)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    # global render_v_shad, render_f_shad
    render_shader_program = pyogl_utils.compile_and_link_vertex_and_frag_shader(render_v_shad, render_f_shad)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Get visibility
    pyogl_utils.configure_opengl_state(width=width, height=height, window=window)
    glEnable(GL_DEPTH_TEST)

    visibility_textures = []
    all_visible_triangles = []
    all_visible_vertices = []
    color_ims = []
    depth_ims = []
    camera_positions = []
    primary_masks = []
 
    for i, viewpoint in enumerate(viewpoints):
        camera_pos = viewpoint
        view_matr = pyogl_utils.get_view_matrix(camera_pos=camera_pos, camera_target=center)
        
        # Visibility Computation
        glBindVertexArray(visibility_vao)
        glUseProgram(visibility_shader_program)
        glUniformMatrix4fv(glGetUniformLocation(visibility_shader_program, "model"),
                        1, GL_FALSE, glm.value_ptr(model_matr))
        glUniformMatrix4fv(glGetUniformLocation(visibility_shader_program, "view"),
                        1, GL_FALSE, glm.value_ptr(view_matr))
        glUniformMatrix4fv(glGetUniformLocation(visibility_shader_program, "projection"),
                        1, GL_FALSE, glm.value_ptr(projection_matr))
        glBindFramebuffer(GL_FRAMEBUFFER, vis_fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_triangles * 3, GL_UNSIGNED_INT, None)
        glfw.poll_events()
        frag_vis_im = np.zeros((height, width), dtype=np.float32)
        glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, frag_vis_im)
        frag_vis_im = np.flipud(frag_vis_im)
        frag_vis_im = frag_vis_im.astype(int)-1
        visibility_textures.append(frag_vis_im)
        visible_triangles = np.unique(frag_vis_im).astype(int)
        visible_triangles = visible_triangles.tolist()
        if -1 in visible_triangles:
            visible_triangles.remove(-1)
        all_visible_triangles.append(visible_triangles)
        visible_vertices = assign_vertices_to_visible_triangles(faces, visible_triangles)
        all_visible_vertices.append(visible_vertices)

        # Rendering Computation
        glBindVertexArray(render_vao)
        glUseProgram(render_shader_program)
        glUniformMatrix4fv(glGetUniformLocation(render_shader_program, "model"),
                        1, GL_FALSE, glm.value_ptr(model_matr))
        glUniformMatrix4fv(glGetUniformLocation(render_shader_program, "view"),
                        1, GL_FALSE, glm.value_ptr(view_matr))
        glUniformMatrix4fv(glGetUniformLocation(render_shader_program, "projection"),
                        1, GL_FALSE, glm.value_ptr(projection_matr))
        texture_loc = glGetUniformLocation(render_shader_program, "texture_sampler")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(texture_loc, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, num_triangles * 3, GL_UNSIGNED_INT, None)
        glfw.poll_events()
        color_buffer_im = pyogl_utils.get_color_buffer(0, 0, width, height)
        depth_buffer_im = pyogl_utils.get_depth_buffer(0, 0, width, height)
        color_ims.append(color_buffer_im)
        depth_ims.append(depth_buffer_im)
        current_camera_pos = pyogl_utils.get_camera_position_from_view_matrix(view_matr)
        camera_positions.append(current_camera_pos)

        glfw.swap_buffers(window)

        if not with_sapiens:
            primary_mask = get_m2fp_inference_on_im(
                color_buffer_im,  # Use color_buffer_im instead of color_im
                with_resize=True
            )
            primary_mask[primary_mask == 7] = 5  # Apply post-processing
        else:
            primary_mask = part_seg_api.get_part_seg_on_im(color_buffer_im)
            remapped_mask = np.zeros_like(primary_mask)
            label_dict = sapiens_semantic_mappingv2 if is_sapiens_v2 else sapiens_semantic_mappingv1
            for old_label, new_label in label_dict.items():
                remapped_mask[primary_mask == old_label] = new_label
            primary_mask = remapped_mask

        # Sanity check
        #frame_utils.show_seg_mask(primary_mask)
        #debug = "debug"

        primary_masks.append(primary_mask)

    all_seg_masks = primary_masks
    model_type = "Sapiens" if with_sapiens else "M2FP"
    print_mask_label_stats(all_seg_masks, model_type, is_sapiens_v2)

    # Delete Textures
    glDeleteTextures(1, [vis_texture])
    glDeleteTextures(1, [render_texture])
    glDeleteTextures(1, [texture_id])
    glDeleteRenderbuffers(1, [vis_depth_rbo])
    glDeleteRenderbuffers(1, [render_rbo])
    glDeleteFramebuffers(1, [vis_fbo])
    glDeleteFramebuffers(1, [render_fbo])
    glDeleteVertexArrays(1, [visibility_vao])
    glDeleteVertexArrays(1, [render_vao])
    glDeleteBuffers(1, [vertex_vbo])
    glDeleteBuffers(1, [vertex_id_vbo])
    glDeleteBuffers(1, [uv_coords_vbo])
    glDeleteBuffers(1, [triangle_ebo1])
    glDeleteBuffers(1, [triangle_ebo2])
    glDeleteProgram(visibility_shader_program)
    glDeleteProgram(render_shader_program)
    glfw.destroy_window(window)
    glfw.terminate()

    vertex_ids -= 1

    if not with_sapiens:
        print("\nChecking for potential leg-to-shoe relabeling...")

        # Calculate average leg pixels across views
        left_leg_counts = [np.sum(mask == 16) for mask in all_seg_masks]
        right_leg_counts = [np.sum(mask == 17) for mask in all_seg_masks]

        # Filter out views with zero counts to avoid skewing the average
        left_leg_nonzero = [count for count in left_leg_counts if count > 0]
        right_leg_nonzero = [count for count in right_leg_counts if count > 0]

        # Calculate averages (with handling for empty lists)
        left_leg_avg = sum(left_leg_nonzero) / len(left_leg_nonzero) if left_leg_nonzero else 0
        right_leg_avg = sum(right_leg_nonzero) / len(right_leg_nonzero) if right_leg_nonzero else 0

        print(f"average Left Leg (16) pixels across all views: {left_leg_avg:,}")
        print(f"average Right Leg (17) pixels across all views: {right_leg_avg:,}")

        # Threshold for relabeling (5000 pixels)
        if 5500 > left_leg_avg > 0:
            print(f"Left Leg has only {left_leg_avg:,} pixels - relabeling to Left Shoe (18)")
            for mask in all_seg_masks:
                left_leg_pixels = mask == 16
                if np.any(left_leg_pixels):
                    mask[left_leg_pixels] = 18

        if 5500 > right_leg_avg > 0:
            print(f"Right Leg has only {right_leg_avg:,} pixels - relabeling to Right Shoe (19)")
            for mask in all_seg_masks:
                right_leg_pixels = mask == 17
                if np.any(right_leg_pixels):
                    mask[right_leg_pixels] = 19


    # Get the maximum class label for initializing the voting array
    max_class = max(np.max(mask) for mask in all_seg_masks)

    vertex_labels = vote_based_vertex_labeling_np(
        all_seg_masks,
        visibility_textures,
        vertex_ids,
        num_vertices,
        num_classes=max_class,
    )

    if not with_sapiens:
        dress_count = np.sum(vertex_labels == 6)
        upper_clothes_count = np.sum(vertex_labels == 5)
        pants_count = np.sum(vertex_labels == 9)
        skirt_count = np.sum(vertex_labels == 12)

        # Rule 1: If dress is present with significant coverage, remove upper clothes, pants and skirts
        total_clothing_vertices = dress_count + upper_clothes_count + pants_count + skirt_count
        if total_clothing_vertices > 0:
            dress_percentage = dress_count / total_clothing_vertices * 100
            skirt_percentage = skirt_count / total_clothing_vertices * 100
    
            if dress_percentage > 25:

                # Find upper clothes vertices to be relabeled
                upper_clothes_indices = np.where(vertex_labels == 5)[0]
                if len(upper_clothes_indices) > 0:
                    vertex_labels[upper_clothes_indices] = 6

                # Find pants vertices to be relabeled
                pants_indices = np.where(vertex_labels == 9)[0]
                if len(pants_indices) > 0:
                    vertex_labels[pants_indices] = 6

                # Find skirt vertices to be relabeled
                skirt_indices = np.where(vertex_labels == 12)[0]
                if len(skirt_indices) > 0:
                    vertex_labels[skirt_indices] = 6

            # Rule 2: If skirt is present with significant coverage, remove pants and dress
            elif skirt_percentage > 20 and skirt_percentage > dress_percentage:

                # Find pants vertices to be relabeled
                pants_indices = np.where(vertex_labels == 9)[0]
                if len(pants_indices) > 0:
                    vertex_labels[pants_indices] = 12

                # Find dress vertices to be relabeled
                dress_indices = np.where(vertex_labels == 6)[0]
                if len(dress_indices) > 0:
                    vertex_labels[dress_indices] = 12
    else:
        left_leg_idx, right_leg_idx = 7, 12
        left_foot_idx, right_foot_idx = 4, 9
        if np.sum(vertex_labels==left_leg_idx) < 1500:
            vertex_labels[left_leg_idx] = left_foot_idx
        if np.sum(vertex_labels==right_leg_idx) < 1500:
            vertex_labels[right_foot_idx] = right_foot_idx

    v_normalized = normalize_point_cloud_scale(vertices)
    valid_mask = vertex_labels != 0
    vertices_l = v_normalized[valid_mask]
    labels_l = vertex_labels[valid_mask]
    vertices_nl = v_normalized[~valid_mask]
    vertices_nl_labels = denoise_nn(vertices_l, vertices_nl, labels_l)
    vertex_labels[~valid_mask] = vertices_nl_labels
    v_normalized, vertex_labels = denoising_by_label_dbscan_new(v_normalized, vertex_labels,
                                eps=0.03, min_samples=100, k_neighbors=40, with_sapiens=with_sapiens,
                                                                is_sapiens_v2=is_sapiens_v2)

    return vertex_labels

if __name__ == "__main__":
    ...