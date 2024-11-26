import torch
from sklearn.metrics import average_precision_score, top_k_accuracy_score


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
    map_score = average_precision_score(y_true, y_pred)  # or 'micro' depending on your preference
    return map_score

def top_k_error(y_true, y_pred, k=5):
    """
    Compute top k error.
    """
    top_k_score = top_k_accuracy_score(y_true, y_pred, k=k)
    error = 1 - top_k_score
    return error