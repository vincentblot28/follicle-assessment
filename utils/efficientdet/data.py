from typing import List, Any, Dict
import random


import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2

random.seed(2023)


def cut_bbox(all_bbox: List[List], shape: int) -> List:
    """Cut bbox so coordinates are part of images

    Parameters
    ----------
    all_bbox : List[List]
        List of bboxes with the following format: [x_min, y_min, x_max, y_max].

    Returns
    -------
    List[List]
        List of bboxes with valid coordinates within [0, shape]
    """
    for bbox in all_bbox:
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(shape, bbox[2])
        bbox[3] = min(shape, bbox[3])

    return all_bbox


def get_samples(
    annotations: List[Dict[str, Any]],
    positive_mix: float = 0,
    include_all_samples: bool = False,
) -> List[Dict[str, Any]]:
    """Get the samples from an annotation list"""

    if include_all_samples:
        return annotations
    else:
        # add positive examples
        samples = []
        for annot in annotations:
            nb_bbox = [len(element["bbox"]) for element in annot["annotation"]]
            if sum(nb_bbox) > 0:
                samples.append(annot)

        # now that we know how many positive examples there are
        # let's add the 1 - positive_mix % of negative ones
        if positive_mix < 1:
            PERCENTAGE_POSITIVES = len(samples) / len(annotations)
            PERCENTAGE_NEGATIVES = PERCENTAGE_POSITIVES * (1 / positive_mix - 1)

            for annot in annotations:
                nb_bbox = [len(element["bbox"]) for element in annot["annotation"]]
                if sum(nb_bbox) == 0 and random.random() < PERCENTAGE_NEGATIVES:
                    samples.append(annot)
        return samples


def get_train_transforms() -> Compose:
    """Transformations to apply to the
    training images.

    Returns
    -------
    Compose
        Transformation of the train image.
    """
    return A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms() -> Compose:
    """Transformations to apply to the
    validation images.

    Returns
    -------
    Compose
        Transformation of the val image.
    """
    return A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
