import numpy as np
import warp
import line
import cv2


class Line():

    def __init__(self, line_type, n_iter, yvals):
        """Initialize Line object that represents left or right line of the road lane.

           Keyword arguments:
           line_type -- str, ('R' or 'L')
               Right (R) or Left (L) line of the leane
           n_iter -- int
               Number of frames retained for line averaging
           yvals -- numpy array
               y-coordinates of the line
        """

        self.n_iter = n_iter
        self.line_type = line_type
        self.yvals = yvals

        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []

        # current value of lane curvature
        self.current_radius = None
        # current value of lane relative position wrt the image center (car
        # center)
        self.current_relative_line_pos = None

    def guess_center(self, image, offset=100, image_frac=4):
        """Returns initial guess for the line position in the image.

           Keyword arguments:
           image -- numpy array
               Binary road image with detected lines
           offset -- int (optional, px)
               Parameter to trims image sides
           image_frac -- int (optional, fraction of image)
               Trims image top
        """
        # Distribution of line pixels along x-direction
        histogram = np.sum(image[image.shape[0] / image_frac:, :], axis=0)
        # Image midpoint
        mid_point = int((histogram.shape[0] - 2 * offset) / 2)

        # Select x-indexes for the left or right line
        if self.line_type == 'L':
            index = np.arange(offset, mid_point).astype(int)
        elif self.line_type == 'R':
            index = np.arange(
                mid_point,
                histogram.shape[0] -
                offset).astype(int)
        # Calculate the position of peak in the histogram using weighted
        # average
        center = np.average(index, weights=histogram[index]).astype(int)
        return center

    def gen_fit_data(self, image, peak_width=50, nbins=10):
        """Scan the image using a square window and return x, y coordinates of the line.

           Keyword arguments:
           image -- numpy array
               Binary road image with detected lines
           peak_width -- int (optional, px)
               Half width of squared window in x-direction
           nbins -- int (optional)
               Number of bins for y-direction scan, defines widow size in y-direction
        """

        # Get approximate index of the line center
        center = self.guess_center(image)

        # x-direction window start and end
        x_start = center - peak_width
        x_end = center + peak_width

        imsize = image.shape
        # y-direction widow size
        bin_size = int(imsize[0] / nbins)

        # Array indexes of line data points
        relevant_idx = []

        Y, X = np.nonzero(image)
        Xy_idx = np.arange(len(Y))
        # Scan image along y-direction
        for nbin in range(nbins):
            index = np.arange(max(0, x_start), min(x_end, imsize[1]))
            # y-direction window start and end
            y_end = imsize[0] - nbin * bin_size
            y_start = y_end - bin_size
            # Distribution of line pixels along x-direction contained in
            # y-direction bin
            histogram = np.sum(image[y_start:y_end, :], axis=0)
            # Calculate line center using weighted average
            try:
                center = int(np.average(index, weights=histogram[index]))
            except:
                pass
            # Update scanning window
            x_start = center - peak_width
            x_end = center + peak_width
            # Select indexes of data points contained in the scare window
            idx = Xy_idx[
                (X >= x_start) & (
                    X < x_end) & (
                    Y >= y_start) & (
                    Y < y_end)]
            relevant_idx.append(idx)
        # Concatenate data points found in y-direction scan
        relevant_idx = np.concatenate(relevant_idx)
        return X[relevant_idx], Y[relevant_idx]

    def fit_line(self, coord):
        """Fit 2nd order polynomial to data points and return its parameters.
        """
        X, Y = coord
        return np.polyfit(Y, X, 2)

    def poly2(self, fit):
        """Calculate the position of lane with 2nd order polynomial.
        """
        return fit[0] * self.yvals**2 + fit[1] * self.yvals + fit[2]

    def calcualte_curvature_and_position(self, coord, imsize):
        """Calculate line curvature and relative position wrt image center.
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Image size
        yval, xval = imsize

        X, Y = coord
        # Fit new polynomials to x,y in world space
        # x = A*y**2 + B*y + C
        fit_cr = np.polyfit(Y * ym_per_pix, X * xm_per_pix, 2)
        # Calculate the new radii of curvature in meters
        curverad = ((1 + (2 * fit_cr[0] * yval * ym_per_pix +
                          fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])

        # Calculate the line position in meters
        line_pos = fit_cr[0] * (yval * ym_per_pix)**2 + \
            fit_cr[1] * yval * ym_per_pix + fit_cr[2]
        # Calculate the center position in meters
        center_pos = xval * xm_per_pix / 2
        relative_line_pos = line_pos - center_pos
        # Return the radii of curvature and the relative position
        return curverad, relative_line_pos

    def update_line(self, image, cutoff=10000.):
        """Update the line parameters.

           Keyword arguments:
           image -- numpy array
               Binary road image with detected lines
           cutoff -- int (optional)
               Cutoff for accepting new fit parameters
        """
        # Get coordinates of line pixels
        coord = self.gen_fit_data(image)
        # Fit 2nd order polynomial
        current_fit = self.fit_line(coord)
        # Calculate the radii of curvature and the relative position
        current_radius, current_relative_line_pos = self.calcualte_curvature_and_position(
            coord, image.shape)

        # Calculate change between best and current fit parameters
        if len(self.current_fit) > 1:
            delta_fit = self.best_fit - current_fit
        else:
            delta_fit = np.zeros(3)

        # Update line parameters if the change in squared sum of fit parameters
        # is smaller than the cutoff
        if np.dot(delta_fit, delta_fit) < cutoff:
            self.current_fit.append(current_fit)
            # Keep only self.n_iter parameters
            if len(self.current_fit) > self.n_iter:
                self.current_fit.pop(0)
            # Average fit parameters
            self.best_fit = np.average(self.current_fit, axis=0)
            # Calculate x-coordinates of the line using the averaged parameters
            self.bestx = self.poly2(self.best_fit)
            # Calculate current radius
            self.current_radius = current_radius
            # Calculate line position wrt the center
            self.current_relative_line_pos = current_relative_line_pos


class Lane(Line):

    def __init__(self, mtx, dist, n_iter=10, ysize=720):
        """Initialize Lane object that represents left and right line of the road lane.

           Keyword arguments:
           mtx -- numpy array
               Camera matrix
           dist -- numpy array
               Distortion coefficients
           n_iter -- int
               Number of frames retained for line averaging
           yvals -- int
               Image size in y-direction
        """
        self.yvals = np.linspace(0, ysize, 20)
        # Initialize left line
        self.left_line = Line('L', n_iter, self.yvals)
        # Initialize right line
        self.right_line = Line('R', n_iter, self.yvals)
        self.mtx = mtx
        self.dist = dist

    def process_image(self, image):
        """Pre-process image: undistort, detect and warp line.
        Return undistorted image and binary perspective transform of detect image lines.
        """
        # Undistort image
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        # Detect lines
        imgline = line.detect_line(undist)
        # Perspective transform
        warped = warp.warp_image(imgline)
        warped[warped > 0] = 1
        return undist, warped

    def update_lane(self, image, debug=False):
        """Update lane and return image with drawn lane.
        """
        # Get undistorted image and warped binary lane image
        undist, warped = self.process_image(image)

        # Update left and right line parameters
        self.left_line.update_line(warped)
        self.right_line.update_line(warped)

        # Draw lane
        Minv = warp.get_transform_matrix(inverse=True)
        result = self.draw_lane(undist, warped, Minv)
        # Write lane curvature and car position wrt lane center
        self.write_curvature_and_position(result)

        return result

    def draw_lane(self, undist, warped, Minv):
        """Overlay lane on road image.
        """

        left_fitx = self.left_line.bestx
        right_fitx = self.right_line.bestx

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.yvals]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, self.yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def write_curvature_and_position(self, image):
        """Write curvature radius and car position wrt lane center.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Average radius of the left and right lane
        radius = (self.left_line.current_radius +
                  self.right_line.current_radius) / 2.
        # Car position wrt lane center
        dist_center = (self.left_line.current_relative_line_pos +
                       self.right_line.current_relative_line_pos) / 2.
        radius_str = 'Radius of curvature = %d(m)' % radius
        dist_center_str = 'Vehicle is %.2f m left of center' % dist_center
        cv2.putText(image, radius_str, (50, 75),
                    font, 1.5, (255, 255, 255), 2)
        cv2.putText(image, dist_center_str, (50, 150),
                    font, 1.5, (255, 255, 255), 2)
