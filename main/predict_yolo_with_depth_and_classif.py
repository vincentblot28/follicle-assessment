import json
import os
import re
import warnings
from typing import Any, Dict

import sys
sys.path.append("/home/ubuntu/follicle-assessment/")

import cv2
import numpy as np
import patchify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from ultralytics import YOLO

from utils.depths_utils import compute_prediction_depths
from utils.patch_utils import is_not_white, create_img_to_classif_from_box


warnings.filterwarnings("ignore")

MODEL_PATH = "/home/ubuntu/yolo_ultralytics/runs/detect/train5/weights/best.pt"
PATCH_SIZE = 1000
OVARY_PATH = "/home/ubuntu/folcon/01_ovary_cuts/ovaries_images"
CLASS_NAME_VALUES = {0: "PMF", 1: "Primary", 2: "Secondary"}
BBOXES_SIZE_PARAMS = {
    "PMF": {"width": 200, "height": 200},
    "Primary": {"width": 350, "height": 350},
    "Secondary": {"width": 450, "height": 450}
}
IMG_CLASSIF_SIZE = 512
SAVE_PREDICTIONS_PATH = "/home/ubuntu/folcon/04_model_predictions/yolo/results_classif_2.json"
DATA_SPLIT_PATH = "/home/ubuntu/folcon/02_model_input/data_split.json"
VGG = models.vgg16(pretrained=False)
num_features = VGG.classifier[6].in_features
VGG.classifier[6] = nn.Linear(num_features, 2)
VGG.load_state_dict(torch.load("/home/ubuntu/follicle-assessment/runs/2024-06-23-02-04_lr=1e-06_wd=0.01/2024-06-23-02-04_lr=1e-06_wd=0.01.pth"))
VGG.eval()
test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def predict_efficientdet(model_path,
                         patch_size: int,
                         ovary_path: str,
                         data_split: Dict,
                         class_name_values: Dict[int, str],
                         bboxes_size_params
                         ) -> Dict[str, Dict[str, Any]]:

    model = YOLO(model_path)
    predictions = {}
    # for ovary_id in tqdm(os.listdir(os.path.join(os.getcwd(), ovary_path))):
    for ovary_id in tqdm(data_split["test"]):
        predictions[ovary_id] = {}
        for cut_name in os.listdir(os.path.join(ovary_path, ovary_id)):
            roi_name = re.findall(r"roi\d+", cut_name)[0]
            predictions[ovary_id][roi_name] = {"bboxes": [], "scores": [], "classes": []}
            cut = cv2.imread(os.path.join(ovary_path, ovary_id, cut_name))
            patches = patchify.patchify(cut, (patch_size, patch_size, 3), step=patch_size)
            boxes_to_classify = []
            for ix in range(patches.shape[0]):
                for iy in range(patches.shape[1]):
                    if is_not_white(patches[ix, iy, 0, :, :, :]):
                        patch = patches[ix, iy, 0, :, :, :]

                        patch_row = ix
                        patch_col = iy

                        result = model(patch, verbose=False)[0]
                        pred_classes = result.boxes.cls.detach().cpu().numpy()
                        pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
                        pred_confs = result.boxes.conf.detach().cpu().numpy()
                        
                        if len(pred_boxes) > 0:
                            for i in range(len(pred_boxes)):
                                x1 = pred_boxes[i][0] + patch_col * patch_size
                                y1 = pred_boxes[i][1] + patch_row * patch_size

                                class_label = pred_classes[i]
                                class_name = class_name_values[int(class_label)]
                                new_bbox_width = bboxes_size_params[class_name]["width"]
                                new_bbox_height = bboxes_size_params[class_name]["height"]
                                new_bbox = np.zeros(4)
                                new_bbox[0] = x1
                                new_bbox[1] = y1
                                new_bbox[2] = x1 + new_bbox_width
                                new_bbox[3] = y1 + new_bbox_height
                                predictions[ovary_id][roi_name]["bboxes"].append(new_bbox.tolist())
                                predictions[ovary_id][roi_name]["scores"].append(float(pred_confs[i]))
                                predictions[ovary_id][roi_name]["classes"].append(class_name)
                                boxes_to_classify.append(create_img_to_classif_from_box(new_bbox, cut, IMG_CLASSIF_SIZE))
            if len(boxes_to_classify) > 0:
                boxes_to_classify = torch.stack([test_transform(box) for box in boxes_to_classify])
                with torch.no_grad():
                    outputs = F.softmax(VGG(boxes_to_classify))[:, 1].numpy()
                    
                
            depths = compute_prediction_depths(cut, np.array(predictions[ovary_id][roi_name]["bboxes"]), resolution=10)
            predictions[ovary_id][roi_name]["depths"] = depths.tolist()
            predictions[ovary_id][roi_name]["scores_classif"] = outputs.tolist()
        # break
    return predictions


if __name__ == "__main__":
    with open(DATA_SPLIT_PATH, "r") as f:
        data_split = json.load(f)

    predictions = predict_efficientdet(
        model_path=MODEL_PATH,
        patch_size=PATCH_SIZE,
        ovary_path=OVARY_PATH,
        data_split=data_split,
        class_name_values=CLASS_NAME_VALUES,
        bboxes_size_params=BBOXES_SIZE_PARAMS
    )
    with open(SAVE_PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f)
