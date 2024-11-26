import torch
import numpy as np
import pyvoi

from sklearn.metrics import average_precision_score, top_k_accuracy_score
from scipy.stats import entropy
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from skimage.metrics import adapted_rand_error


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


def confusion_matrix(pred_label_matrix: torch.tensor, gt_label_matrix: torch.tensor):
    """
    Compute metrics based on the predicted and ground truth label matrices.
    """
    tp = torch.sum(pred_label_matrix * gt_label_matrix)
    fp = torch.sum(pred_label_matrix * (1 - gt_label_matrix))
    fn = torch.sum((1 - pred_label_matrix) * gt_label_matrix)
    tn = torch.sum((1 - pred_label_matrix) * (1 - gt_label_matrix))

    return tp, fp, fn, tn

def mAP(y_true, y_pred):
    """
    Compute mean average precision.
    """
    map_score = average_precision_score(y_true, y_pred)
    return map_score

def top_k_error(y_true, y_pred, k=5):
    """
    Compute top k error.
    """
    top_k_score = top_k_accuracy_score(y_true, y_pred, k=k)
    error = 1 - top_k_score
    return error