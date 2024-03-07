"""Utils functions for the depths computation."""
import time

import cv2
import numpy as np


def binarize_image(img, th=245):
    """
    Binarizes an image using a given threshold.

    Args:
        img (numpy.ndarray): The input image.
        th (int): The threshold value. Pixels below this value
            will be set to 0, and pixels above or equal to this value
            will be set to 1. Default is 245.

    Returns:
        numpy.ndarray: The binarized image.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = (gray_img < th).astype(np.uint8)
    thresh = cv2.medianBlur(thresh, ksize=3)

    return thresh


def get_area(contour):
    """
    Calculates the area of a contour.
    """
    width = np.max(contour[:, 0, 0]) - np.min(contour[:, 0, 0])
    height = np.max(contour[:, 0, 1]) - np.min(contour[:, 0, 1])

    return width * height


def get_contours(img_th):
    """
    Finds and returns the contours of a binary image.

    Parameters:
    img_th (numpy.ndarray): The binary image.

    Returns:
    mask_filled (numpy.ndarray): The filled mask of the contour
         with the largest area.
    contour_max (numpy.ndarray): The contour with the largest area.
    """
    contours, _ = cv2.findContours(
        img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = {
        get_area(contours[i]):
        contours[i] for i in range(len(contours)) if
        contours[i][:, 0, :].shape[0] > 5
    }
    contour_max = areas[max(areas.keys())]

    mask_filled = np.zeros_like(img_th)
    cv2.drawContours(mask_filled, [contour_max], -1, (1, 1, 1), cv2.FILLED)

    return mask_filled, contour_max


def get_depths(contour_vect, mask, box, factor=100):

    # Initialize matrix that will be used after
    contour_temp = np.zeros_like(mask)
    contour = np.zeros_like(mask)

    # Initialize depth and count for all box to np.inf
    depths = np.ones(len(box)) * np.inf
    time0 = time.time()
    contour_vect = [contour_vect]

    while (contour[box[:, 0], box[:, 1]] == 0).any():
        time1 = time.time()
        for cont in contour_vect:
            contour = cv2.drawContours(contour, [cont], -1, (1, 1, 1), factor)
        contour = cv2.bitwise_and(contour, mask)
        contour = cv2.bitwise_or(contour, contour_temp)

        contour_temp = contour.copy()
        time2 = time.time()

        depth = np.sum(contour) / np.sum(mask)
        depths[(contour[box[:, 0], box[:, 1]] == 1) * (depths > depth)] = depth

        if (contour == mask).all():
            depths[(depths > depth)] = 1
            break

        contour_vects, _ = cv2.findContours(
            contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_vects = sorted(
            contour_vects, key=lambda x: get_area(x), reverse=True
        )
        contour_vect = contour_vects[1:]
        contour = np.zeros_like(mask)

        # logging
        log = "DrawContour time: " + str(time2 - time1) + \
            " ; depth: " + str(depth)
        print(log, end="\r")

    print(log)
    print("Total time: " + str(time.time() - time0))

    return depths


def compute_prediction_depths(img, predictions, resolution):
    img = cv2.resize(img, (img.shape[1] // resolution, img.shape[0] // resolution))
    img_bin = binarize_image(img)
    mask_filled, contour_max = get_contours(img_bin)
    if len(predictions) > 0:
        preds_roi_center = np.concatenate(
            [
                [(predictions[:, 1] + predictions[:, 3]) / 2],
                [(predictions[:, 0] + predictions[:, 2]) / 2]
            ],
            axis=0
        ).T.astype(int) // resolution
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

    return depths
