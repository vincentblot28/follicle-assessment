from typing import Union, cast, Optional, Tuple, List, Any

import numpy as np
from mapie._typing import NDArray
from mapie.utils import check_alpha
from scipy.stats import binom
from sklearn.model_selection import KFold
from tqdm import tqdm

from utils.metrics import match_annotations_and_predictions, compute_precision, compute_nD_precision, compute_nD_recall


def _select_boxes_classe(boxes, classes, class_, scores=None, other_variables_values=None):
    index_class = np.array([i in class_ for i in classes])
    other_variables_values_class = {}
    if len(index_class) > 0:
        boxes = boxes[index_class]
        classes = classes[index_class]
        if scores is not None:
            scores = scores[index_class]
        if other_variables_values is not None:
            for ov in other_variables_values.keys():
                other_variables_values_class[ov] = other_variables_values[ov][index_class]
    else:
        boxes = []
        classes = []
        if scores is not None:
            scores = []
        if other_variables_values is not None:
            for ov in other_variables_values.keys():
                other_variables_values_class[ov] = []
    if scores is not None:
        return boxes, classes, scores, other_variables_values_class
    else:
        return boxes, classes


def _create_dataset(
        annotations, predictions, slide_name,
        roi_name, iou_th, variables, classes="all"
):
    n_variables = len(variables)
    gt_boxes = np.array(annotations[slide_name][roi_name]["bboxes"])
    gt_classes = np.array(annotations[slide_name][roi_name]["classes"])
    pred_boxes = np.array(predictions[slide_name][roi_name]["bboxes"])
    pred_classes = np.array(predictions[slide_name][roi_name]["classes"])
    pred_scores = np.array(predictions[slide_name][roi_name]["scores"])
    pred_classes = np.array(predictions[slide_name][roi_name]["classes"])
    variables_values = {}
    for variable in variables:
        variables_values[variable] = np.array(predictions[slide_name][roi_name][variable])
    if classes != "all":
        gt_boxes, gt_classes = _select_boxes_classe(
            gt_boxes, gt_classes, classes
        )
        pred_boxes, pred_classes, pred_scores, variables_values = _select_boxes_classe(
            pred_boxes, pred_classes, classes, pred_scores, variables_values
        )

    matched_annots, matched_preds = match_annotations_and_predictions(
        gt_boxes, pred_boxes, gt_classes, pred_classes, iou_th
    )
    n_unmatched_gt = len(gt_boxes) - len(matched_annots)

    data_temp = np.zeros((len(pred_boxes), n_variables + 1))

    for i, ov in enumerate(variables):
        data_temp[:, i] = variables_values[ov]
    data_temp[matched_preds, -1] = 1
    data_unmatched = np.zeros((n_unmatched_gt,  n_variables + 1))
    data_unmatched[:, -1] = 1
    data = np.concatenate((data_temp, data_unmatched), axis=0)
    X = data[:, :n_variables].tolist()
    y = data[:, n_variables].tolist()
    assert sum(y) == len(gt_boxes)
    return X, y


def create_x_y(
        sld_names, annotations, predictions,
        iou_th, variables, classes="all"
):
    n_variables = len(variables)
    X, y = [], []
    for slide_name in sld_names:
        for roi_name in predictions[slide_name].keys():
            X_, y_ = _create_dataset(
                annotations, predictions, slide_name, roi_name,
                iou_th=iou_th, classes=classes, variables=variables
            )
            X.append(X_)
            y.append(y_)
    max_len_y = 0
    for y_ in y:
        if len(y_) > max_len_y:
            max_len_y = len(y_)

    for y_ in y:
        y_.extend([0] * (max_len_y - len(y_)))
    for x in X:
            x.extend([[0] * (n_variables)] * (max_len_y - len(x)))
    return np.array(X), np.array(y)

def run_ltt_cv(
        annotations, predictions,
        target_precision, iou_th,
        ths_dict, classes,
        delta, correction="bonferroni"
):

    kf = KFold(n_splits=len(predictions))

    slide_names = list(predictions.keys())
    results = {}

    for cal_index, test_index in tqdm(
        kf.split(slide_names), total=kf.get_n_splits()
    ):
        sld_cal = np.array(slide_names)[cal_index]
        sld_test = np.array(slide_names)[test_index][0]
        results[sld_test] = {}
        X_cal, y_cal = create_x_y(
            sld_cal, annotations, predictions, iou_th,
            ths_dict.keys(), classes
        )
        cal_precisions = compute_nD_precision(
            lambdas=ths_dict.values(),
            y_pred_proba=X_cal, y=y_cal
        )
        cal_precisions = np.nanmean(cal_precisions, axis=0)
        cal_recalls = compute_nD_recall(
            lambdas=ths_dict.values(),
            y_pred_proba=X_cal, y=y_cal
        )
        cal_recalls = np.nanmean(cal_recalls, axis=0)
        best_ths = run_ltt(cal_precisions, cal_recalls, y_cal, target_precision, delta, ths_dict.values(), correction=correction)

        for i, k in enumerate(ths_dict.keys()):
            results[sld_test]["best_th_" + k] = best_ths[i]
    return results



def run_ltt(
        precisions, recalls, y, target_precision, delta,
        lambdas, correction="bonferroni"
):
    """
    Run the LTT procedure to find the best threshold for the objectness and depth.
    """
    alpha = [1 - target_precision]
    n_obs = len(y)
    original_shape = precisions.shape
    risks = 1 - np.array(precisions)
    if correction == "bonferroni":
        risks = risks.ravel()
        valid_index, _ = ltt_bonferoni(risks, alpha, delta, n_obs)
        valid_index = valid_index[0]
    elif correction == "fst":
        if  len(lambdas) == 1:
            valid_index = ltt_fst_univariate(risks, alpha, delta, n_obs)
        else:
            valid_index = ltt_fst_multivariate(risks, alpha, delta, n_obs)
    else:
        raise ValueError(
            "Invalid correction: correction must be either 'bonferroni' or 'fst'."
        )

    matrix_valid = np.zeros(original_shape)
    if correction == "fst":
        for c in valid_index:
            if isinstance(c, np.int64):
                matrix_valid[c] = 1
            else:
                matrix_valid[tuple(c)] = 1
    elif correction == "bonferroni":
        matrix_valid[
            np.unravel_index(
                np.array(valid_index),
                (matrix_valid.shape)
            )
        ] = 1
    else:
        raise ValueError(
            "Invalid correction: correction must be either 'bonferroni' or 'fst'."
        )
    matrix_valid_recall = matrix_valid * recalls
    best_recalls_index = np.unravel_index(
        np.argmax(matrix_valid_recall),
        (matrix_valid.shape)
    )
    
    return [lambda_[best_recalls_index[i]] for i, lambda_ in enumerate(lambdas)]


def ltt_bonferoni(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: Optional[float],
    n_obs: int,
) -> Tuple[List[List[Any]], NDArray]:

    if delta is None:
        raise ValueError(
            "Invalid delta: delta cannot be None while"
            + " controlling precision with LTT. "
        )
    p_values = compute_hoeffdding_bentkus_p_value(r_hat, n_obs, alpha_np)
    valid_index = []
    N = len(p_values)
    for i in range(len(alpha_np)):
        l_index = np.where(p_values[:, i] <= delta/N)[0].tolist()
        valid_index.append(l_index)

    return valid_index, p_values


def compute_hoeffdding_bentkus_p_value(
    r_hat: NDArray,
    n_obs: int,
    alpha: Union[float, NDArray]
) -> NDArray:

    alpha_np = cast(NDArray, check_alpha(alpha))
    alpha_np = alpha_np[:, np.newaxis]
    r_hat_repeat = np.repeat(
        np.expand_dims(r_hat, axis=1),
        len(alpha_np),
        axis=1
    )
    alpha_repeat = np.repeat(
        alpha_np.reshape(1, -1),
        len(r_hat),
        axis=0
    )
    hoeffding_p_value = np.exp(
        -n_obs * _h1(
            np.where(
                r_hat_repeat > alpha_repeat,
                alpha_repeat,
                r_hat_repeat
            ),
            alpha_repeat
        )
    )
    bentkus_p_value = np.e * binom.cdf(
        np.ceil(n_obs * r_hat_repeat), n_obs, alpha_repeat
    )
    hb_p_value = np.where(
        bentkus_p_value > hoeffding_p_value,
        hoeffding_p_value,
        bentkus_p_value
    )
    return hb_p_value

def _h1(
    r_hats: NDArray,
    alphas: NDArray
) -> NDArray:
    """
    This function allow us to compute
    the tighter version of hoeffding inequality.
    This function is then used in the
    hoeffding_bentkus_p_value function for the
    computation of p-values.

    Parameters
    ----------
    r_hats: NDArray of shape (n_lambdas, n_alpha).
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision
        making and therefore empirical risk.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    alphas: NDArray of shape (n_lambdas, n_alpha).
        Contains the different alphas control level.
        In other words, empirical risk must be less
        than each alpha in alphas.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    Returns
    -------
    NDArray of shape a(n_lambdas, n_alpha).
    """
    elt1 = r_hats * np.log(r_hats/alphas)
    elt2 = (1-r_hats) * np.log((1-r_hats)/(1-alphas))
    return elt1 + elt2

def ltt_fst_univariate(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: Optional[float],
    n_obs: int,
) -> Tuple[List[List[Any]], NDArray]:

    if delta is None:
        raise ValueError(
            "Invalid delta: delta cannot be None while"
            + " controlling precision with LTT. "
        )
    p_values = compute_hoeffdding_bentkus_p_value(r_hat, n_obs, alpha_np)
    valid_index = []
    J = np.arange(0, len(r_hat), 10)
    for i in range(len(alpha_np)):
        for j in J:
            if j in valid_index:
                continue
            while (j < len(r_hat)) and (p_values[j, i] <= delta/len(J)):
                valid_index.append(j)
                j += 1

    return valid_index


def ltt_fst_multivariate(
    r_hat: NDArray,
    alpha: float,
    delta: Optional[float],
    n_obs: int,
) -> Tuple[List[List[Any]], NDArray]:

    if delta is None:
        raise ValueError(
            "Invalid delta: delta cannot be None while"
            + " controlling precision with LTT. "
        )
    r_hat_1D = r_hat.ravel()
    p_values_1D = compute_hoeffdding_bentkus_p_value(r_hat_1D, n_obs, alpha)
    p_values = p_values_1D.reshape(r_hat.shape)
    valid_index = []
    N = int(r_hat.size / r_hat.shape[0])
    for j in range(N):
        j_unraveled = tuple(np.unravel_index(j, r_hat.shape[1:]))
        indices_p_values = (range(r_hat.shape[0]), ) + j_unraveled
        p_values_j = p_values[indices_p_values][::-1]
        for k, p in enumerate(p_values_j):
            if p <= delta/N:
                k_ordered = len(p_values_j) - k - 1
                valid_index.append([k_ordered] + list(j_unraveled))
            else:
                break

    return valid_index


def run_naive_precision_contorl(
        annotations, predictions,
        target_precision, iou_th,
        ths_obj, classes
):

    kf = KFold(n_splits=len(predictions))

    slide_names = list(predictions.keys())
    results = {}

    for cal_index, test_index in tqdm(
        kf.split(slide_names), total=kf.get_n_splits()
    ):
        sld_cal = np.array(slide_names)[cal_index]
        sld_test = np.array(slide_names)[test_index][0]
        results[sld_test] = {
            "best_th_obj": [],
            "best_th_depth": [],
        }
        X_cal, y_cal = create_x_y(
            sld_cal, annotations, predictions, iou_th,
            variables=["scores"], classes=classes
        )

        cal_precisions = compute_precision(
            lambdas=ths_obj,
            y_pred_proba=X_cal, y=y_cal
        )
        cal_precisions = np.nanmean(cal_precisions, axis=0)
        best_th = ths_obj[np.argmin(np.abs(cal_precisions - target_precision))]

        results[sld_test]["best_th_obj"].append(best_th)

    return results