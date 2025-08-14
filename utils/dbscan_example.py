import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate synthetic 2D point cloud
X = np.random.randn(100, 2)
X[:30] += 5
X[30:60] += 10
X[60:100] += 20

# Apply DBSCAN
dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("DBSCAN Clustering")
plt.show()