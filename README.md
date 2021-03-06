
[//]: # (Image References)

[image1]: ./output_images/original_image.jpg "Original"
[image2]: ./output_images/undistorted_image.jpg "Undistorted"
[image3a]: ./output_images/undistorted_road_image.jpg "Undistorted Road"
[image3b]: ./output_images/original_road_image.jpg "Original Road"
[image4]: ./output_images/thresholded_binary.jpg "Thresholded"
[image5]: ./output_images/straight_lines.jpg "Warp Example"
[image6]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image7]: ./output_images/poly_fit.jpg "Fit Visual"
[image8]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video_with_lane.mp4 "Video"

## Advanced Lane Finding Project

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fourth code cell of the IPython notebook located in "./Line-detection.ipynb" that calls function `calibrate_camera` implemented in the file called `calibrate.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1] ![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I apply the distortion correction to one of the test images. Here is the resulting undistorted image:
![alt text][image3a]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The color threshold  uses the saturation channel of an image in HLS space (hue, lightness, and saturation); the gradient threshold is based on the absolute value of the Sobel x-direction gradient.  These two thresholds are implemented in the file `line.py` (lines 17-50 and 94-106, respectively).   

I apply the thresholds in the function called `detect_line` implemented in `line.py`.
This function returns a binary image where ones correspond to situations where either of the thresholds applies.

Here is an example output (see also 7th cell of the IPyhton notebook):

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 28 through 33 in the file `warp.py`. This function calls `cv2.warpPerspective` and another function called `get_transform_matrix` that hardcodes the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 275, 680      | 350, 200        | 
| 1045, 680      | 950, 700      |
| 734, 480     | 950, 700      |
| 554, 480      | 350, 200        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image (the IPython notebook, code cell 8 through 12).

![alt text][image5]

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
This step is implemented in the file `lane.py` that defines two classes: `class Line()` and `class Lane(Line)`.  
The class `Line` implements methods to generate line data points, fit a 2nd order polynomial  and also to calculate line curvature and position with respect to image center (i.e. car center).  The class `Lane` inherits after left and right `Line` object and implements lane detection pipeline.

To generated lane-line pixels I scan images in y-direction using a rectangular window `[y_start:y_end, x_start:x_end]` and update the window position along x-axis
every time the window is shifted in y-direction:
```
class Line():
    ...
    def gen_fit_data(self, image, peak_width=50, nbins=10):
    ...
    for nbin in range(nbins):
        histogram = np.sum(image[y_start:y_end, :], axis=0)
        center = int(np.average(index, weights=histogram[index]))
        x_start = center - peak_width
        x_end = center + peak_width
    ...
```
Indexes of points belonging to the rectangular window are appended to list called `relevant\_idx`  and are then used to select all non-zero points.
```
    def gen_fit_data(self, image, peak_width=50, nbins=10):
        ...
        Y, X = np.nonzero(image)
        ...
        # y-direction scan
        ...
        return X[relevant_idx], Y[relevant_idx]
```

Once data points are identified I call the function `fit\_line` to fit a 2nd order polynomial.
Code cells 13 thought 16 of the IPython notebook illustrate the final results. 

![alt text][image7]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this step in the function called `calcualte_curvature_and_position_`  (lines 135 through 160 in `lane.py`). Both curvatures and vehicle positions are calculated in meter space.

To calculate line curvature use the following equation: 
<code>
<i>
R(l) = [1+(2A(l)y + B(l))<sup>2</sup>]<sup>3/2</sup> / |2A(l)|
</i>
</code>
where <code><i>A, B, C</i></code> are parameters of the 2nd oder polynomial fit and <code><i>y</i></code> represents the bottom of the image.
The lane curvature is then the average value of the left and right lines curvatures: <code> <i> R = [R(l)+R(r)]/2 </i> </code> (`lane.py` code lines 289-280).

To calculate vehicle positions I first calculate positions of left and right lanes at the bottom of the image with respect to image center (`M`):
<code>
<i>
P(l) = A(l)y<sup>2</sup> + B(l)y + C(l) - M
</i>
</code>
The vehicle positions with respect to lane center are then the average values of the left and right lines positions: <code> <i> P = [P(l)+P(r)]/2 </i> </code> (`lane.py` code lines 292-283).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
Here is an example of my result on a test image (the code cell number 17 of the IPython notebook).

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video file](./project_video.mp4)

[![IMAGE ALT TEXT](http://img.youtube.com/vi/vg3sN9wN-N0/0.jpg)](http://www.youtube.com/watch?v=vg3sN9wN-N0 "Video Title")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I encountered couple of challenges in the project. Here are the three most important ones:

1. Generation of binary lane-line images: I used a try and error approach to test methods to create thresholded binary image. Although, the resulting lane-line detection works well, the whole process was somewhat time consuming. An automated optimization process would allow for a more exhaustive search.

2. Identification of lane-line pixels in binary images. For some images my algorithm was picking up a road edge rather than a lane-line. 
I solved this issues by masking left and right sides of images.  A better approach would be to identify all the lines and then pick the lines that are left and right of the car.
In that we would keep information about all the lanes. 

3. Stability of lane detection. To improve stability of lane detection I used a running average over the last 10 image frames. In situations where binary images had lot of noise or simply line pixels were not present, resulting fit parameters were wrong and skewed the average. To solve this issue I update the average only when fit parameters were not unreasonably different from the current average parameters.

Here are a few potential issues with the implemented line detection algorithm:

1. The current pipeline was tested only on one type of road conditions.  
It is possible that the pipeline will fail under rainy, dark or other type of road conditions.  While using saturation channel which is more illumination independent should improve generalization, more  testing and a more systematic optimization approach to build a binary image could help. 

2. Lane detection may not work properly when lines split or merge e.g. at cross-sections,  acceleration or deceleration lanes etc. Identification of multiple lines and sanity checks based on known ranges of lane parameters could help here.

3. Algorithm may get confused during lane changing.
