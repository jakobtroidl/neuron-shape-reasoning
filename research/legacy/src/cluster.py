import torch
from sklearn import cluster

def agglomerative(distance_matrix: torch.tensor, distance_threshold = 0.2):
    """
    Perform agglomerative clustering on the input distance matrix.
    """

    distance_matrix_np = distance_matrix.cpu().numpy()
    clustering = cluster.AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=distance_threshold, linkage='average')
    clustering.fit(distance_matrix_np)

    labels = clustering.labels_
    labels = torch.tensor(labels, dtype=torch.int64).to(distance_matrix.device)

    return labels

def dbscan(distance_matrix: torch.tensor, eps = 0.2, min_samples = 5):
    """
    Perform DBSCAN clustering on the input distance matrix.
    """

    distance_matrix_np = distance_matrix.cpu().numpy()
    clustering = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix_np)

    labels = torch.tensor(labels, dtype=torch.int64).to(distance_matrix.device)

    return labels