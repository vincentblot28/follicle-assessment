import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/ubuntu/follicle-assessment/")

import cv2
from tqdm import tqdm

from utils.ltt_utils import create_x_y

ANNOT_PATH = "/home/ubuntu/folcon/01_ovary_cuts/roi_annotation_united/all_annotations.json"
PRED_PATH = "/home/ubuntu/folcon/04_model_predictions/yolo/results_train.json"
IOU_TH = .2
OVARY_PATH = "/home/ubuntu/folcon/01_ovary_cuts/ovaries_images"
BBOX_SIZE = 512
IMG_SAVE_PATH = "/home/ubuntu/folcon/02_model_input_classif/yolo/images2/"


with open(ANNOT_PATH, "r") as f:
    annotations = json.load(f)

with open(PRED_PATH, "r") as f:
    predictions = json.load(f)


X, y = create_x_y(
    predictions.keys(), annotations, predictions,
    iou_th=IOU_TH, classes="all", with_depth=False
)

count = 0
labels = {}
for slide_name in tqdm(list(predictions.keys())):
    for roi_name in predictions[slide_name].keys():
        cut_name = f"{slide_name}__{roi_name}.tif"
        cut = cv2.imread(os.path.join(OVARY_PATH, slide_name, cut_name))
        predictions_list_temp = []
        for i, pred_box in enumerate(predictions[slide_name][roi_name]["bboxes"]):
            x0, y0, x1, y1 = pred_box
            center = (x0 + x1) // 2, (y0 + y1) // 2
            x0, y0 = center[0] - BBOX_SIZE // 2, center[1] - BBOX_SIZE // 2
            x1, y1 = x0 + BBOX_SIZE, y0 + BBOX_SIZE
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            patch = cut[y0:y1, x0:x1]

            if patch.shape != (BBOX_SIZE, BBOX_SIZE, 3):
                print("mishaped patch")
                continue
            labels[f"{slide_name}_{roi_name}_{i}"] = y[count][i]
            assert predictions[slide_name][roi_name]["scores"][i] == X[count][i]


            cv2.imwrite(os.path.join(IMG_SAVE_PATH, f"{slide_name}_{roi_name}_{i}.tif"), patch)

        count += 1

with open("/home/ubuntu/folcon/02_model_input_classif/yolo/labels.json", "w") as f:
    json.dump(labels, f)