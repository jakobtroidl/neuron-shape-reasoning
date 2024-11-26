
import torch
import numpy as np

from sklearn.metrics import mutual_info_score
from sklearn import cluster
from scipy.stats import entropy
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import pyvoi
from skimage.metrics import adapted_rand_error


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

def entropy_of_clustering(labels):
    """Compute the entropy of a clustering."""
    label_counts = np.bincount(labels.astype(int))
    probs = label_counts / len(labels)
    return entropy(probs)

def voi(labels_true, labels_pred):
    """Compute the variation of information between two clusterings."""
    vi, vi_split, vi_merge = pyvoi.VI(labels_true, labels_pred, torch=False)
    return vi, vi_split, vi_merge

def adjusted_rand_score(labels_true, labels_pred):
    """Compute the adjusted rand index between two clusterings."""
    return ARI(labels_true.astype(int), labels_pred.astype(int))

def adjusted_rand_error(labels_true, labels_pred):
    """Compute the adjusted rand index between two clusterings."""
    are, are_prec, are_recall = adapted_rand_error(labels_true.astype(int), labels_pred.astype(int))
    return are, are_prec, are_recall