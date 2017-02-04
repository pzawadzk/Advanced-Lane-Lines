import cv2
import numpy as np


def get_transform_matrix(inverse=False):
    """Returns perspective transform matrix.
    """
    # Source points
    src = np.float32([
        [275, 680],
        [1045, 680],
        [734, 480],
        [554, 480]
    ])
    # Destination points
    dst = np.float32([
        [350, 700],
        [950, 700],
        [950, 200],
        [350, 200]
    ])
    if inverse:
        return cv2.getPerspectiveTransform(dst, src)
    else:
        return cv2.getPerspectiveTransform(src, dst)


def warp_image(image):
    """Applies perspective transform and returns transformed image.
    """
    M = get_transform_matrix()
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped
