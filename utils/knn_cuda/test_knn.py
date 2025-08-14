import torch
import open3d as o3d
import knn_cuda

def knn_cpu(query, support, k):
    B, N, C = query.shape
    _, M, _ = support.shape
    idx = []

    for b in range(B):
        q = query[b]         # [N, C]
        s = support[b]       # [M, C]
        # Compute [N, M] distance matrix
        dist = torch.cdist(q, s, p=2)  # [N, M]
        knn_idx = dist.topk(k, largest=False).indices  # [N, k]
        idx.append(knn_idx)

    return torch.stack(idx, dim=0)


def compare_knn_indices(idx_cuda, idx_cpu):
    """
    Compare two [B, N, K] index tensors and report mismatches.
    Assumes both are on the same device.
    """
    mismatch = idx_cuda != idx_cpu
    num_total = idx_cuda.numel()
    num_diff = mismatch.sum().item()

    print(f"Total elements: {num_total}")
    print(f"Differing elements: {num_diff}")
    if num_diff == 0:
        print("✅ All indices match exactly.")
        return

    # Find some differing indices
    mismatched_locs = torch.nonzero(mismatch, as_tuple=False)
    for i in range(min(10, len(mismatched_locs))):  # Show up to 10 mismatches
        b, n, k = mismatched_locs[i].tolist()
        print(f"Batch {b}, Point {n}, Neighbor {k} -- CUDA: {idx_cuda[b, n, k].item()} vs CPU: {idx_cpu[b, n, k].item()}")



"""
xyz1 = torch.randn(8, 100000, 3).to(0)
xyz2 = torch.randn(8, 1000, 3).to(0)
idx1 = knn_cuda.knn_query(xyz1, xyz2, 3)
idx2 = knn_cuda.knn_query(xyz2, xyz1, 3)
idx3 = knn_cuda.self_knn_query(xyz1, 3)
debug = "debug"

idx1_cpu = knn_cpu(xyz1.cpu(), xyz2.cpu(), 3).to(0)
idx2_cpu = knn_cpu(xyz2.cpu(), xyz1.cpu(), 3).to(0)

# Check correctness
compare_knn_indices(idx1_cpu, idx1)
compare_knn_indices(idx2_cpu, idx2)
"""
pcd1 = torch.rand(1, 100000, 3).to(0)
knn_idx = knn_cuda.self_knn_query(pcd1, 30).long().cpu()
debug = "debug"
#print(knn_idx)
