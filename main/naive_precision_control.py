import warnings

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from utils.ltt_utils import create_x_y
from utils.metrics import (
    compute_precision
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


def run_dumb_precision_contorl(
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
            classes, with_depth=False
        )

        cal_precisions = compute_precision(
            lambdas=ths_obj,
            y_pred_proba=X_cal[:, :, np.newaxis], y=y_cal
        )
        cal_precisions = np.nanmean(cal_precisions, axis=0)
        best_th = ths_obj[np.argmin(np.abs(cal_precisions - target_precision))]

        results[sld_test]["best_th_obj"].append(best_th)

    return results
