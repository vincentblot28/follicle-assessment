import os
import json

import cv2
import numpy as np
import patchify

from utils.patch_utils import is_annot_center_in_patch, is_not_white

PATCH_SIZE = 1000
ROIS_PATH = "/mnt/folcon/01_ovary_cuts/ovaries_images"
ANNOTATIONS_PATH = "/mnt/folcon/01_ovary_cuts/roi_annotation"
PATCHES_SAVE_PATH = "/mnt/folcon/02_model_input/patches_stride_train_val"
PATCHES_ANNOTATIONS_PATH = "/mnt/folcon/02_model_input/patches_annotation_stride_train_val.json"
DATA_SLIT_PATH = "/mnt/folcon/02_model_input/annotations/data_split.json"


def generate_patches_and_annotation(
        patch_size: int,
        rois_path: str,
        annotations,
        patches_save_path: str,
        data_split
):
    patch_id = 0
    patches_annotations = []

    for slide_name in annotations.keys():
        if slide_name not in data_split["test"]:
            for roi_name in annotations[slide_name].keys():
                roi_file = f"{slide_name}__{roi_name}.tif"
                roi = cv2.imread(os.path.join(rois_path, slide_name, roi_file))
                roi_annot = annotations[slide_name][roi_name]

                patches = patchify.patchify(roi, (patch_size, patch_size, 3), step=patch_size // 2)

                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):

                        new_annotation = []

                        if is_not_white(patches[i, j, 0, :, :, :]):

                            target_file_path = f'{slide_name}/{slide_name}__{roi_name}_{i}_{j}'
                            for element in roi_annot['annotations']:

                                xmin, ymin, w, h = element['bbox']

                                xmin_patch = xmin - (patch_size * j / 2)
                                ymin_patch = ymin - (patch_size * i / 2)
                                valid_bbox = is_annot_center_in_patch(
                                    xmin_patch + w // 2, ymin_patch + h // 2, patch_size
                                )

                                # if bbox is valid
                                if valid_bbox:

                                    # add annotation
                                    new_annotation.append(
                                        {
                                            'bbox': [xmin_patch, ymin_patch, xmin_patch + w, ymin_patch + h],
                                            'category': element['category'],
                                            'MarkerGuid': element['MarkerGuid'],
                                            'CategoryGuid': element['CategoryGuid'],
                                        }
                                    )

                            if (len(new_annotation) > 0) or (np.random.rand() > .95):
                                if not os.path.exists(os.path.join(patches_save_path, slide_name)):
                                    os.makedirs(os.path.join(patches_save_path, slide_name))
                                patches_annotations.append(
                                    {
                                        'filepath': target_file_path,
                                        'annotation': new_annotation.copy(),
                                        'id': patch_id,
                                        'image': f"{slide_name}__{roi_name}_{i}_{j}",
                                        'slide': slide_name,
                                    }
                                )
                                patch_id += 1

                                cv2.imwrite(
                                    os.path.join(patches_save_path, target_file_path + ".png"),
                                    patches[i, j, 0, :, :, :],
                                )

    return patches_annotations


if __name__ == "__main__":
    with open(DATA_SLIT_PATH, "r") as f:
        data_split = json.load(f)
    patches_annotations = generate_patches_and_annotation(
        PATCH_SIZE, ROIS_PATH, ANNOTATIONS_PATH, PATCHES_SAVE_PATH, data_split
    )
    with open(PATCHES_ANNOTATIONS_PATH, "w") as f:
        json.dump(patches_annotations, f)
