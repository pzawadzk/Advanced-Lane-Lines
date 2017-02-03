import numpy as np
import cv2


def calibrate_camera(files, nx=9, ny=6):
    """Returns camera calibration parameters.
    """
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # Loop over calibration files
    for file in files:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    cv2.destroyAllWindows()
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    params = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return params
