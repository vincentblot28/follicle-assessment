import numpy as np
from mapie.control_risk.ltt import ltt_procedure

from utils.metrics import match_annotations_and_predictions


def run_ltt(precisions, delta, y, target_precision, with_depth, ths_obj, ths_depth=None, cal_recalls=None):
    """
    Run the LTT procedure to find the best threshold for the objectness and depth.
    """
    alpha = [1 - target_precision]
    n_obs = len(y)
    risks = 1 - np.array(precisions)
    if with_depth:
        risks = risks.ravel()
    valid_index, _ = ltt_procedure(risks, alpha, delta, n_obs)
    if with_depth:

        matrix_valid = np.zeros((len(ths_obj), len(ths_depth)))
        matrix_valid[
            np.unravel_index(
                np.array(valid_index[0]),
                (matrix_valid.shape)
            )
        ] = 1
        matrix_valid_recall = matrix_valid * cal_recalls
        best_recall = np.unravel_index(
            np.argmax(matrix_valid_recall),
            (matrix_valid.shape)
        )
        best_th_obj, best_th_depth = ths_obj[best_recall[0]], \
            ths_depth[best_recall[1]]
        return best_th_obj, best_th_depth
    else:
        valid_index = valid_index[0][0]

        best_th = ths_obj[valid_index]
        return best_th


def select_boxes_classe(boxes, classes, class_, scores=None, depths=None):
    index_class = np.array([i in class_ for i in classes])
    if len(index_class) > 0:
        boxes = boxes[index_class]
        classes = classes[index_class]
        if scores is not None:
            scores = scores[index_class]
            depths = depths[index_class]
    else:
        boxes = []
        classes = []
        if scores is not None:
            scores = []
            depths = []
    if scores is not None:
        return boxes, classes, scores, depths
    else:
        return boxes, classes


def create_dataset(
        annotations, predictions, slide_name,
        roi_name, iou_th, with_depth=True, classes="all"
):

    gt_boxes = np.array(annotations[slide_name][roi_name]["bboxes"])
    gt_classes = np.array(annotations[slide_name][roi_name]["classes"])
    pred_boxes = np.array(predictions[slide_name][roi_name]["bboxes"])
    pred_classes = np.array(predictions[slide_name][roi_name]["classes"])
    pred_scores = np.array(predictions[slide_name][roi_name]["scores"])
    pred_classes = np.array(predictions[slide_name][roi_name]["classes"])
    depths = np.array(predictions[slide_name][roi_name]["depths"])
    if classes != "all":
        gt_boxes, gt_classes = select_boxes_classe(
            gt_boxes, gt_classes, classes
        )
        pred_boxes, pred_classes, pred_scores, depths = select_boxes_classe(
            pred_boxes, pred_classes, classes, pred_scores, depths
        )

    matched_annots, matched_preds = match_annotations_and_predictions(
        gt_boxes, pred_boxes, gt_classes, pred_classes, iou_th
    )
    n_unmatched_gt = len(gt_boxes) - len(matched_annots)
    data_temp = np.zeros((len(pred_boxes), 2 + with_depth))
    data_temp[:, 0] = pred_scores
    if with_depth:
        data_temp[:, 1] = depths
    data_temp[matched_preds, 1 + with_depth] = 1
    data_unmatched = np.zeros((n_unmatched_gt, 2 + with_depth))
    data_unmatched[:, 1 + with_depth] = 1
    data = np.concatenate((data_temp, data_unmatched), axis=0)
    if with_depth:
        X = data[:, :2].tolist()
    else:
        X = data[:, 0].tolist()
    y = data[:, 1 + with_depth].tolist()
    assert sum(y) == len(gt_boxes)
    return X, y


def create_x_y(
        sld_names, annotations, predictions,
        iou_th, classes="all", with_depth=True
):
    X, y = [], []
    for slide_name in sld_names:
        for roi_name in predictions[slide_name].keys():
            X_, y_ = create_dataset(
                annotations, predictions, slide_name, roi_name,
                iou_th=iou_th, classes=classes, with_depth=with_depth
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
        if with_depth:
            x.extend([[0, 0]] * (max_len_y - len(x)))
        else:
            x.extend([0] * (max_len_y - len(x)))

    return np.array(X), np.array(y)
