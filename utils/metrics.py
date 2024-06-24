from typing import cast

import numpy as np
import pandas as pd
from mapie._typing import NDArray
from sklearn.utils.validation import column_or_1d


def compute_iou_matrix(annotation_boxes, pred_boxes):
    x11, y11, x12, y12 = np.split(annotation_boxes, 4, axis=1)
    x21, y21, x22, y22 = np.split(pred_boxes, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def numpy_2d_argmax(a: np.ndarray):
    """Returns the index of the minimum value in a 2D numpy array
    along both axis.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    Tuple[int, int]
        A tuple with the position of the minimum value in both x and y axis

    Examples
    --------
    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> numpy_2d_argmin(x)
    (1, 1)
    """
    if a.ndim != 2:
        raise ValueError(
            "Dimension error, only 2D numpy arrays are supported"
            "in this function."
        )
    x_idx, y_idx = np.unravel_index(np.argmax(a, axis=None), a.shape)
    return (int(x_idx), int(y_idx))


def match_annotations_and_predictions(
    annotations, predictions, gt_classes=None, pred_classes=None, min_iou=.3
):
    """Match annotations and predictions by distance."""

    matched_annot = []
    matched_pred = []
    if len(annotations) > 0 and len(predictions) > 0:
        iou_matrix = compute_iou_matrix(annotations, predictions)
        class_matrix = create_class_matrix(
            gt_classes, pred_classes, len(annotations),
            len(predictions)
        )
        iou_matrix = np.multiply(iou_matrix, class_matrix)

        # dist matrix: each row is associated to an annotation and each
        # column is associated to a prediction.
        if max(iou_matrix.shape) > 0:
            best_annot, best_pred = numpy_2d_argmax(iou_matrix)
            while iou_matrix[best_annot, best_pred] > min_iou:
                matched_annot.append(best_annot)
                matched_pred.append(best_pred)
                iou_matrix[best_annot, :] = - np.float32('inf')
                iou_matrix[:, best_pred] = - np.float32('inf')
                best_annot, best_pred = numpy_2d_argmax(iou_matrix)

    return matched_annot, matched_pred


def create_class_matrix(gt_classes, pred_classes, n_annot, n_pred):
    if True: # pred_classes is None:
        return np.ones((n_annot, n_pred))
    else:
        num_gt_classes = len(gt_classes)
        num_pred_classes = len(pred_classes)
        class_matrix = np.zeros((num_gt_classes, num_pred_classes))

        for i in range(num_gt_classes):
            for j in range(num_pred_classes):
                if gt_classes[i] == pred_classes[j]:
                    class_matrix[i, j] = 1

        return class_matrix


def compute_recall(
    lambdas: NDArray,
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    In `MapieMultiLabelClassifier` when `metric_control=recall`,
    compute the recall per observation for each different
    thresholds lambdas.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    lambdas: NDArray of shape (n_lambdas, )
        Threshold that permit to compute recall.

    Returns
    -------
    NDArray of shape (n_samples, n_labels, n_lambdas)
        Risks for each observation and each value of lambda.
    """
    if y_pred_proba.ndim != 3:
        raise ValueError(
            "y_pred_proba should be a 3d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if y.ndim != 2:
        raise ValueError(
            "y should be a 2d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError(
            "y and y_pred_proba could not be broadcast."
        )
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(
        y_pred_proba,
        n_lambdas,
        axis=2
    )
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    recall = (
        _true_positive(y_pred_th, y_repeat) /
        y.sum(axis=1)[:, np.newaxis]
    )
    return recall


def compute_precision(
    lambdas: NDArray,
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    In `MapieMultiLabelClassifier` when `metric_control=precision`,
    compute the precision per observation for each different
    thresholds lambdas.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    lambdas: NDArray of shape (n_lambdas, )
        Threshold that permit to compute precision score.

    Returns
    -------
    NDArray of shape (n_samples, n_labels, n_lambdas)
        Risks for each observation and each value of lambda.
    """
    if y_pred_proba.ndim != 3:
        raise ValueError(
            "y_pred_proba should be a 3d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if y.ndim != 2:
        raise ValueError(
            "y should be a 2d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError(
            "y and y_pred_proba could not be broadcast."
        )
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(
        y_pred_proba,
        n_lambdas,
        axis=2
    )
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    with np.errstate(divide='ignore', invalid="ignore"):
        precision = _true_positive(y_pred_th, y_repeat)/y_pred_th.sum(axis=1)
    precision = np.nan_to_num(precision, nan=1)
    return precision


def _true_positive(
    y_pred_th: NDArray,
    y_repeat: NDArray
) -> NDArray:
    """
    Compute the number of true positive.

    Parameters
    ----------
    y_pred_proba : NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    Returns
    -------
    tp: float
        The number of true positive.
    """
    tp = (y_pred_th * y_repeat).sum(axis=1)
    return tp


def compute_2D_precision(lambdas_obj, lambdas_depths, y_pred_proba, y):
    """Compute the precision for each observation and each
    thresholds.

    Parameters
    ----------
    lambdas_obj: NDArray of shape (n_lambdas_obj, )
        Threshold that permit to compute precision score for
        objective.

    lambdas_depths: NDArray of shape (n_lambdas_depths, )
        Threshold that permit to compute precision score for
        depths.

    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    Returns
    -------
    precision_obj: NDArray of shape (n_samples, n_lambdas_obj)
        Precision for each observation and each value of lambda
        for objective.

    precision_depths: NDArray of shape (n_samples, n_lambdas_depths)
        Precision for each observation and each value of lambda
        for depths.
    """
    precisions = np.zeros((len(y), len(lambdas_obj), len(lambdas_depths)))
    for i, lambda_depth in enumerate(lambdas_depths):
        y_pred_proba_rations = (
            (y_pred_proba[:, :, 1] <= lambda_depth) * y_pred_proba[:, :, 0]
        )
        precision = compute_precision(
            lambdas_obj,
            y_pred_proba_rations[:, :, np.newaxis],
            y
        )
        precisions[:, :, i] = precision
    return precisions


def compute_2D_recall(lambdas_obj, lambdas_depths, y_pred_proba, y):
    """Compute the precision for each observation and each
    thresholds.

    Parameters
    ----------
    lambdas_obj: NDArray of shape (n_lambdas_obj, )
        Threshold that permit to compute precision score for
        objective.

    lambdas_depths: NDArray of shape (n_lambdas_depths, )
        Threshold that permit to compute precision score for
        depths.

    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    Returns
    -------
    precision_obj: NDArray of shape (n_samples, n_lambdas_obj)
        Precision for each observation and each value of lambda
        for objective.

    precision_depths: NDArray of shape (n_samples, n_lambdas_depths)
        Precision for each observation and each value of lambda
        for depths.
    """
    recalls = np.zeros((len(y), len(lambdas_obj), len(lambdas_depths)))
    for i, lambda_depth in enumerate(lambdas_depths):
        y_pred_proba_rations = (
            (y_pred_proba[:, :, 1] <= lambda_depth) * y_pred_proba[:, :, 0]
        )
        recall = compute_recall(
            lambdas_obj,
            y_pred_proba_rations[:, :, np.newaxis],
            y
        )
        recalls[:, :, i] = recall
    return recalls


def get_slide_index(slide_name, results):
    for i in range(len(results["all"]["true"]["sld_name"])):
        if slide_name in results["all"]["true"]["sld_name"][i]:
            return i


def get_p_r_df(cm, id_for_cm):
    p_r_df = {}

    for follicle in id_for_cm.keys():
        p_r_df[follicle] = {}
        p_r_df[follicle]["precision"] = (
            cm[id_for_cm[follicle], id_for_cm[follicle]] / np.sum(cm[:, id_for_cm[follicle]])
        )
        p_r_df[follicle]["recall"] = cm[id_for_cm[follicle], id_for_cm[follicle]] / np.sum(cm[id_for_cm[follicle], :])
    p_r_df = pd.DataFrame(p_r_df).T
    return p_r_df
