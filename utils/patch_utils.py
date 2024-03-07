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
