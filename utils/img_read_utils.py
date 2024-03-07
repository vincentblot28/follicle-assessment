"""read utils module."""
import cv2


def read_image(file_path, downsacle_factor=200):
    """
    Reads an image from a file path with rasterio and
    returns it as a numpy array.
    """
    img = cv2.imread(file_path)
    img = cv2.resize(img, (img.shape[1] // downsacle_factor, img.shape[0] // downsacle_factor))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
