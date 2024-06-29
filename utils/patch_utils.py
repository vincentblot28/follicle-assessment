import cv2
import numpy as np


def is_not_white(image, threshold=0.95):
    """
    Check if image is not mostly white (reddish).

    Parameters:
    ----------
    image : numpy array
        Image data in the form of a numpy array.
    threshold : float
        Threshold value to determine maximum allowed amount of white pixels.

    Returns:
    -------
    bool
        True if the image is not mostly white, False otherwise.
    """
    n_white_pix = np.sum(image >= 245)
    is_above_threshold = n_white_pix/image.size < threshold

    return is_above_threshold


def is_annot_center_in_patch(x, y, size):
    """
    Check whether a box center lies within an image.

    Parameters:
    ----------
    x : float
        The x-coordinate of the box center.
    y : float
        The y-coordinate of the box center.
    size : int
        The size of the image in pixels.

    Returns:
    -------
    bool
        True if the box center lies within the image, False otherwise.
    """
    return ((0 < x) and (0 < y) and (x < size) and (y < size))


def create_img_to_classif_from_box(pred_box, cut, box_size, to_rgb=True):
    x0, y0, x1, y1 = np.array(pred_box).astype(int)
    center = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = center[0] - box_size // 2, center[1] - box_size // 2
    x1, y1 = x0 + box_size, y0 + box_size
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(cut.shape[1], x1)
    y1 = min(cut.shape[0], y1)

    img_to_classif = cut[y0:y1, x0:x1]
    if img_to_classif.shape != (box_size, box_size, 3):
        print(img_to_classif.shape)
        img_to_classif = cv2.resize(img_to_classif, (box_size, box_size))
    if to_rgb:
        img_to_classif = cv2.cvtColor(img_to_classif, cv2.COLOR_BGR2RGB)
    
    return img_to_classif