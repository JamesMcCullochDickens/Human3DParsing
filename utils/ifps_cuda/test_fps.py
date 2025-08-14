import torch
import farthest_point_sampling as fps

B, N, K = 2, 32, 5
points = torch.rand(B, N, 3).cuda()

def cpu_fps(points: torch.Tensor, K: int):
    B, N, _ = points.shape
    indices = torch.zeros(B, K, dtype=torch.long)
    dists = torch.full((B, N), float('inf'))

    farthest = torch.zeros(B, dtype=torch.long)
    batch_indices = torch.arange(B)

    for i in range(K):
        indices[:, i] = farthest
        centroid = points[batch_indices, farthest]  # (B, 3)
        dist = torch.sum((points - centroid[:, None, :]) ** 2, dim=2)  # (B, N)
        dists = torch.minimum(dists, dist)
        farthest = torch.max(dists, dim=1).indices

    return indices


with torch.no_grad():
    indices = fps.farthest_point_sampling(points, K)
    cpu_indices = cpu_fps(points.cpu(), K)
    print("Sampled indices:", indices)
    print("Unique?", torch.unique(indices[0]).numel() == K)
    print("Sampled indices:", cpu_indices)
    print("Unique?", torch.unique(cpu_indices[0]).numel() == K)