import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/ubuntu/follicle-assessment/")

import cv2
import numpy as np
from tqdm import tqdm

from utils.metrics import compute_iou_matrix

ANNOT_PATH = "/home/ubuntu/folcon/01_ovary_cuts/roi_annotation_united/all_annotations.json"
PRED_PATH = "/home/ubuntu/folcon/04_model_predictions/yolo/results_train.json"
IOU_TH = .3
OVARY_PATH = "/home/ubuntu/folcon/01_ovary_cuts/ovaries_images"
BBOX_SIZE = 512
IMG_SAVE_PATH = "/home/ubuntu/folcon/02_model_input_classif/yolo/images2/"


with open(ANNOT_PATH, "r") as f:
    annotations = json.load(f)

with open(PRED_PATH, "r") as f:
    predictions = json.load(f)


labels = {}
for slide_name in tqdm(list(predictions.keys())):
    for roi_name in predictions[slide_name].keys():
        gt_boxes = np.array(annotations[slide_name][roi_name]["bboxes"])
        pred_boxes = np.array(predictions[slide_name][roi_name]["bboxes"])
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)
            matched_boxes = iou_matrix.max(axis=0) >= IOU_TH
            cut_name = f"{slide_name}__{roi_name}.tif"
            cut = cv2.imread(os.path.join(OVARY_PATH, slide_name, cut_name))
            predictions_list_temp = []
            for i, pred_box in enumerate(pred_boxes):
                x0, y0, x1, y1 = pred_box
                center = (x0 + x1) // 2, (y0 + y1) // 2
                x0, y0 = center[0] - BBOX_SIZE // 2, center[1] - BBOX_SIZE // 2
                x1, y1 = x0 + BBOX_SIZE, y0 + BBOX_SIZE
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                patch = cut[y0:y1, x0:x1]

                if patch.shape != (BBOX_SIZE, BBOX_SIZE, 3):
                    print("mishaped patch")
                    continue
                labels[f"{slide_name}_{roi_name}_{i}"] = int(matched_boxes[i])
                # assert predictions[slide_name][roi_name]["scores"][i] == X[count][i]

                cv2.imwrite(os.path.join(IMG_SAVE_PATH, f"{slide_name}_{roi_name}_{i}.tif"), patch)

with open("/home/ubuntu/folcon/02_model_input_classif/yolo/labels.json", "w") as f:
    json.dump(labels, f)