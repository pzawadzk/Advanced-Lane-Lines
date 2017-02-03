import numpy as np
import cv2


def detect_line(image):
    """Applies S-channel threshold and Sobel absolute x-direction gradient threshold.
    Returns binary image.
    """
    s_binary = hsv_threshold(image, channel=2, thresh=(160, 255))
    gradx = abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=200)

    combined = np.zeros_like(s_binary)
    combined[(gradx == 1) | (s_binary == 1)] = 1
    return combined


def abs_sobel_thresh(img, orient='x', sobel_kernel=9,
                     thresh_min=0, thresh_max=255):
    """Applies Sobel absolute gradient threshold.
    Returns binary image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(
            cv2.Sobel(
                gray,
                cv2.CV_64F,
                1,
                0,
                ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(
            cv2.Sobel(
                gray,
                cv2.CV_64F,
                0,
                1,
                ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) &
                  (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=9, mag_thresh=(0, 255)):
    """Applies Sobel gradient magnitude threshold.
    Returns binary image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=9, thresh=(0, np.pi / 2)):
    """Applies Sobel gradient direction threshold.
    Returns binary image.
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_threshold(img, channel=2, thresh=(0, np.pi / 2)):
    """Applies HLS-channel threshold.
    Returns binary image.
    """
    # Extract H or L or S channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    img_channel = hsv[:, :, channel]
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(img_channel)
    binary_output[(img_channel >= thresh[0]) & (img_channel <= thresh[1])] = 1

    # Return the binary image
    return binary_output
