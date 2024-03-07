import json
import gc
import os

import numpy as np

from utils.depths_utils import binarize_image, get_contours, get_depths
from utils.img_read_utils import read_image


IMG_PATH = "data/01_ovary_cuts/ovaries_images"
PREDS_PATH = "data/04_model_predictions/efficientdet/results.json"


def compute_depth(img_paths, pred_path, resolution=1):
    with open(pred_path, "r") as f:
        preds = json.load(f)
    n_images = sum([len(preds[slide_name]) for slide_name in preds.keys()])
    count_image = 1
    with open(pred_path, "r") as f:
        preds = json.load(f)
    for slide_name in preds.keys():
        for roi_name in preds[slide_name].keys():
            
            print(f"Processing image {count_image}/{n_images}")
            # Get predictions
            img_path = os.path.join(
                img_paths, slide_name, f"{slide_name}__{roi_name}.tif"
            )
            print("Reading image")
            img = read_image(img_path, downsacle_factor=resolution)
            print("Reading ok")
            img_bin = binarize_image(img)
            mask_filled, contour_max = get_contours(img_bin)
            preds_roi_complete = np.array(
                    preds[slide_name][roi_name]["bboxes"]
                ) / resolution
            if len(preds_roi_complete) > 0:
                preds_roi_center = np.concatenate(
                    [
                        [(preds_roi_complete[:, 1] + preds_roi_complete[:, 3]) / 2],
                        [(preds_roi_complete[:, 0] + preds_roi_complete[:, 2]) / 2]
                    ],
                    axis=0
                ).T.astype(int)
                preds_roi_center = np.maximum(preds_roi_center - 1, 0)
                preds_roi_center[:, 0] = np.minimum(preds_roi_center[:, 0], img.shape[0] - 1)
                preds_roi_center[:, 1] = np.minimum(preds_roi_center[:, 1], img.shape[1] - 1)

                # Get thedepth of each prediction
                depths = get_depths(
                    contour_max, mask_filled,
                    preds_roi_center, factor=int(100 / resolution)
                )
            else:
                depths = np.array([])
            preds[slide_name][roi_name]["depths"] = depths.tolist()
            # Trigger garbage collection
            gc.collect()
            count_image += 1

    return preds


if __name__ == "__main__":
    preds_augmented = compute_depth(
        IMG_PATH, PREDS_PATH
    )
    with open(PREDS_PATH, "w") as f:
        json.dump(preds_augmented, f)
