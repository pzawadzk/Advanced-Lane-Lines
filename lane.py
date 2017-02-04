import numpy as np
import warp
import line
import cv2


class Line():

    def __init__(self, line_type, n_iter):
        self.n_iter = n_iter
        self.line_type = line_type
        self.yvals = np.linspace(0, 720, 20)

        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []

        self.current_radius = None
        self.current_relative_line_pos = None

    def guess_center(self, image, offset=100, image_frac=4):
        histogram = np.sum(image[image.shape[0] / image_frac:, :], axis=0)
        mid_point = int((histogram.shape[0] - 2 * offset) / 2)
        if self.line_type == 'L':
            index = np.arange(offset, mid_point).astype(int)
        elif self.line_type == 'R':
            index = np.arange(
                mid_point,
                histogram.shape[0] -
                offset).astype(int)
        return np.average(index, weights=histogram[index]).astype(int)

    def gen_fit_data(self, image, peak_width=80, nbins=10):

        peak_width = 50
        imsize = image.shape
        center = self.get_center(image)
        x_start = center - peak_width
        x_end = center + peak_width
        bin_size = int(imsize[0] / nbins)
        relevant_idx = []
        Y, X = np.nonzero(image)
        Xy_idx = np.arange(len(Y))
        for nbin in range(nbins):
            index = np.arange(max(0, x_start), min(x_end, imsize[1]))
            y_end = imsize[0] - nbin * bin_size
            y_start = y_end - bin_size
            histogram = np.sum(image[y_start:y_end, :], axis=0)
            try:
                center = int(np.average(index, weights=histogram[index]))
            except:
                pass
            x_start = center - peak_width
            x_end = center + peak_width
            idx = Xy_idx[
                (X >= x_start) & (
                    X < x_end) & (
                    Y >= y_start) & (
                    Y < y_end)]
            relevant_idx.append(idx)
        relevant_idx = np.concatenate(relevant_idx)
        return X[relevant_idx], Y[relevant_idx]

    # TODO: get better startgin point
    def get_center(self, image):
        return self.guess_center(image)

    def fit_line(self, coord):
        X, Y = coord
        return np.polyfit(Y, X, 2)

    def poly2(self, fit):
        return fit[0] * self.yvals**2 + fit[1] * self.yvals + fit[2]

    def update_line(self, image, cutoff=10000.):
        coord = self.gen_fit_data(image)
        current_fit = self.fit_line(coord)
        current_radius, current_relative_line_pos = self.calcualte_radius(
            coord)

        if len(self.current_fit) > 1:
            delta_fit = self.best_fit - current_fit
        else:
            delta_fit = np.zeros(3)

        if np.dot(delta_fit, delta_fit) < cutoff:
            self.current_fit.append(current_fit)
            if len(self.current_fit) > self.n_iter:
                self.current_fit.pop(0)
            self.best_fit = np.average(self.current_fit, axis=0)
            self.bestx = self.poly2(self.best_fit)
            self.current_radius = current_radius
            self.current_relative_line_pos = current_relative_line_pos

    def calcualte_radius(self, coord):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        X, Y = coord
        yval = 720
        xval = 1280
        # Fit new polynomials to x,y in world space
        # x = a*y**2 + b*y + c
        fit_cr = np.polyfit(Y * ym_per_pix, X * xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * yval * ym_per_pix +
                          fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])

        line_pos = fit_cr[0] * (yval * ym_per_pix)**2 + \
            fit_cr[1] * yval * ym_per_pix + fit_cr[2]
        center_pos = xval * xm_per_pix / 2
        relative_line_pos = line_pos - center_pos

        return curverad, relative_line_pos


class Lane(Line):

    def __init__(self, mtx, dist, n_iter):
        self.left_line = Line(line_type='L', n_iter)
        self.right_line = Line(line_type='R', n_iter)
        self.mtx = mtx
        self.dist = dist

    def process_image(self, image):

        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        imgline = line.detect_line(undist)
        warped = warp.warp_image(imgline)
        warped[warped > 0] = 1
        return undist, warped

    def update_lane(self, image, debug=False):

        undist, warped = self.process_image(image)
        #plt.imshow(warped, cmap='Greys_r')

        self.left_line.update_line(warped)
        self.right_line.update_line(warped)

        Minv = warp.get_transform_matrix(inverse=True)
        result = self.draw_lines(undist, warped, Minv)

        font = cv2.FONT_HERSHEY_SIMPLEX
        radius = (self.left_line.current_radius +
                  self.right_line.current_radius) / 2.
        dist_center = (self.left_line.current_relative_line_pos +
                       self.right_line.current_relative_line_pos) / 2.
        radius_str = 'Radius of curvature = %d(m)' % radius
        dist_center_str = 'Vehicle is %.2f m left of center' % dist_center
        cv2.putText(result, radius_str, (50, 75),
                    font, 1.5, (255, 255, 255), 2)
        cv2.putText(result, dist_center_str, (50, 150),
                    font, 1.5, (255, 255, 255), 2)

        return result

    def draw_lines(self, undist, warped, Minv):
        yvals = np.linspace(0, 720, 20)

        left_fitx = self.left_line.bestx
        right_fitx = self.right_line.bestx

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
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
