import json
import warnings

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from utils.ltt_utils import create_x_y, run_ltt
from utils.metrics import (
    compute_precision,
    compute_2D_precision, compute_2D_recall
)

warnings.filterwarnings("ignore")
MODEL_NAME = "efficientdet"
ANNOT_PATH = "data/01_ovary_cuts/roi_annotation_united/all_annotations.json"
PRED_PATH = f"data/04_model_predictions/{MODEL_NAME}/results.json"
SAVE_APTH = f"data/05_LTT_results/{MODEL_NAME}/"
THRESHOLDS_OBJ = np.linspace(.45, .8, 30)
THRESHOLDS_DEPTH = np.linspace(.3, .8, 20)
TARGET_PRECISION = .5
DELTA = 0.001
IOU_TH = .4


def run_ltt_cv(
        annotations, predictions,
        target_precision, iou_th,
        ths_obj, ths_depth, classes,
        with_depth, delta
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
            classes, with_depth
        )
        if with_depth:
            cal_precisions = compute_2D_precision(
                lambdas_obj=ths_obj, lambdas_depths=ths_depth,
                y_pred_proba=X_cal, y=y_cal
            )
            cal_precisions = np.nanmean(cal_precisions, axis=0)
            cal_recalls = compute_2D_recall(
                lambdas_obj=ths_obj, lambdas_depths=ths_depth,
                y_pred_proba=X_cal, y=y_cal
            )
            cal_recalls = np.nanmean(cal_recalls, axis=0)
            best_th_obj, best_th_depth = run_ltt(
                precisions=cal_precisions, delta=delta, y=y_cal,
                target_precision=target_precision, with_depth=with_depth,
                ths_obj=ths_obj, ths_depth=ths_depth, cal_recalls=cal_recalls
            )
            results[sld_test]["best_th_obj"].append(best_th_obj)
            results[sld_test]["best_th_depth"].append(best_th_depth)
        else:
            cal_precisions = compute_precision(
                lambdas=ths_obj,
                y_pred_proba=X_cal[:, :, np.newaxis], y=y_cal
            )
            cal_precisions = np.nanmean(cal_precisions, axis=0)
            best_th = run_ltt(
                precisions=cal_precisions, delta=delta, y=y_cal,
                target_precision=target_precision, with_depth=with_depth,
                ths_obj=ths_obj
            )
            results[sld_test]["best_th_obj"].append(best_th)

    return results


if __name__ == "__main__":
    with open(ANNOT_PATH) as f:
        annotations = json.load(f)

    with open(PRED_PATH) as f:
        predictions = json.load(f)

    results_depth = run_ltt_cv(
        annotations, predictions,
        TARGET_PRECISION, IOU_TH, THRESHOLDS_OBJ,
        THRESHOLDS_DEPTH, with_depth=True, classes="all",
        delta=DELTA
    )
    results_classical = run_ltt_cv(
        annotations, predictions,
        TARGET_PRECISION, IOU_TH, THRESHOLDS_OBJ,
        THRESHOLDS_DEPTH, with_depth=False, classes="all",
        delta=DELTA
    )

    with open(f"{SAVE_APTH}results_depth.json", "w") as f:
        json.dump(results_depth, f)
    with open(f"{SAVE_APTH}results_classical.json", "w") as f:
        json.dump(results_classical, f)
