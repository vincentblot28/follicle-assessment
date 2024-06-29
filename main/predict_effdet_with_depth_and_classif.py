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

from utils.depths_utils import compute_prediction_depths
from utils.efficientdet.dataset import get_valid_transforms
from utils.efficientdet.efficientdet import EfficientDetModel
from utils.patch_utils import is_not_white, create_img_to_classif_from_box


warnings.filterwarnings("ignore")

MODEL_PATH = "/home/ubuntu/follicle-assessment/model_weights/effdet.ckpt"

MODEL_PARAMS = {
    "learning_rate": 0.0001,
    "prediction_confidence_threshold": 0.45,
    "wbf_iou_threshold": 0.2,
    "model_architecture": "tf_efficientdet_d2",
    "num_workers": 4,
    "batch_size": 16,
    "max_epochs": 100,
    "num_classes": 90,  # Artefact during the training
    "img_size": 768,
    "lr_warmup_epoch": 1
}
PATCH_SIZE = 1000
OVARY_PATH = "/home/ubuntu/ssd/folcon/01_ovary_cuts/ovaries_images"
CLASS_NAME_VALUES = {1: "PMF", 2: "Primary", 3: "Secondary"}
BBOXES_SIZE_PARAMS = {
    "PMF": {"width": 200, "height": 200},
    "Primary": {"width": 350, "height": 350},
    "Secondary": {"width": 450, "height": 450}
}
SAVE_PREDICTIONS_PATH = "/home/ubuntu/ssd/folcon/04_model_predictions/efficientdet/results_depth_classif.json"
DATA_SPLIT_PATH = "/home/ubuntu/ssd/folcon/02_model_input/data_split.json"
IMG_CLASSIF_SIZE = 512
VGG = models.vgg16(pretrained=False)
num_features = VGG.classifier[6].in_features
VGG.classifier[6] = nn.Linear(num_features, 2)
VGG.load_state_dict(torch.load("/home/ubuntu/follicle-assessment/model_weights/vgg16.pth"))
VGG.eval()
test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_efficientdet(data_split,
                         model_path,
                         model_params: Dict[str, Any],
                         patch_size: int,
                         ovary_path: str,
                         class_name_values: Dict[int, str],
                         bboxes_size_params
                         ) -> Dict[str, Dict[str, Any]]:

    model = EfficientDetModel.load_from_checkpoint(model_path, **model_params)
    if torch.cuda.is_available():
        model.eval().cuda(device=0)
        VGG.cuda()
    else:
        model.eval()

    model.inference_tfms = get_valid_transforms(with_ground_truth=False)

    predictions = {}
    # for ovary_id in tqdm(os.listdir(os.path.join(os.getcwd(), ovary_path))):
    for ovary_id in tqdm(data_split["test"]):
        predictions[ovary_id] = {}
        for cut_name in os.listdir(os.path.join(ovary_path, ovary_id)):
            roi_name = re.findall(r"roi\d+", cut_name)[0]
            predictions[ovary_id][roi_name] = {"bboxes": [], "scores": [], "classes": []}
            cut = cv2.imread(os.path.join(ovary_path, ovary_id, cut_name))
            cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
            patches = patchify.patchify(cut, (patch_size, patch_size, 3), step=patch_size)
            boxes_to_classify = []
            for ix in range(patches.shape[0]):
                for iy in range(patches.shape[1]):
                    if is_not_white(patches[ix, iy, 0, :, :, :]):
                        patch = patches[ix, iy, 0, :, :, :]

                        patch_row = ix
                        patch_col = iy

                        pred_boxes, pred_classes, pred_confs, _ = model.predict([patch])
                        if len(pred_boxes) > 0:
                            for i in range(len(pred_boxes[0])):
                                mean_x = (pred_boxes[0][i][0] + pred_boxes[0][i][2]) / 2  # predicted_bboxes[0][i][2]
                                mean_y = (pred_boxes[0][i][1] + pred_boxes[0][i][3]) / 2  # predicted_bboxes[0][i][3]
                                predicted_center = [mean_x, mean_y]  # predicted_bboxes[0][i]
                                predicted_center[0] = predicted_center[0] + patch_row * patch_size
                                predicted_center[1] = predicted_center[1] + patch_col * patch_size
                                class_label = pred_classes[0][i]
                                class_name = class_name_values[int(class_label)]
                                new_bbox_width = bboxes_size_params[class_name]["width"]
                                new_bbox_height = bboxes_size_params[class_name]["height"]
                                new_bbox = np.zeros(4)
                                new_bbox[1] = predicted_center[0] - new_bbox_width / 2
                                new_bbox[0] = predicted_center[1] - new_bbox_height / 2
                                new_bbox[3] = predicted_center[0] + new_bbox_width / 2
                                new_bbox[2] = predicted_center[1] + new_bbox_height / 2
                                predictions[ovary_id][roi_name]["bboxes"].append(new_bbox.tolist())
                                predictions[ovary_id][roi_name]["scores"].append(pred_confs[0][i])
                                predictions[ovary_id][roi_name]["classes"].append(class_name)
                                boxes_to_classify.append(
                                    create_img_to_classif_from_box(
                                        new_bbox, cut, IMG_CLASSIF_SIZE, to_rgb=False
                                    )
                                )

            if len(boxes_to_classify) > 0:
                boxes_to_classify = torch.stack([test_transform(box) for box in boxes_to_classify])
                if torch.cuda.is_available():
                    boxes_to_classify = boxes_to_classify.cuda()
                with torch.no_grad():
                    n_boxes = len(boxes_to_classify)
                    boxes_to_classify1 = boxes_to_classify[:n_boxes // 2]
                    boxes_to_classify2 = boxes_to_classify[n_boxes // 2:]
                    outputs1 = F.softmax(VGG(boxes_to_classify1))[:, 1].cpu().numpy()
                    outputs2 = F.softmax(VGG(boxes_to_classify2))[:, 1].cpu().numpy()
                    outputs = np.concatenate([outputs1, outputs2], axis=0)
            depths = compute_prediction_depths(cut, np.array(predictions[ovary_id][roi_name]["bboxes"]), resolution=10)
            predictions[ovary_id][roi_name]["depths"] = depths.tolist()
            predictions[ovary_id][roi_name]["scores_classif"] = outputs.tolist()
        # break
    return predictions


if __name__ == "__main__":
    with open(DATA_SPLIT_PATH, "r") as f:
        data_split = json.load(f)

    predictions = predict_efficientdet(
        data_split=data_split,
        model_path=MODEL_PATH,
        model_params=MODEL_PARAMS,
        patch_size=PATCH_SIZE,
        ovary_path=OVARY_PATH,
        class_name_values=CLASS_NAME_VALUES,
        bboxes_size_params=BBOXES_SIZE_PARAMS
    )
    with open(SAVE_PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f)

