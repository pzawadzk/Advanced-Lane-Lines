
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
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3a]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The color threshold  is based on the saturation channel of the image transformed to HLS space (hue, lightness, and saturation); the gradient threshold is based on the absolute value of the Sobel x-direction gradient.  These two thresholds are implemented in the file `line.py` (lines 17-50 and 94-106).   

I apply the thresholds in the function called `detect_line` implemented in the file `line.py`.
This function returns a binary image where ones correspond to situations where either of the thresholds applies.

Here is an example output (see also 7th cell of the IPyhton notebook):

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 28 through 33 in the file `warp.py. The function another function called `get_transform_matrix` that hardcodes the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 275, 680      | 350, 200        | 
| 1045, 680      | 950, 700      |
| 734, 480     | 950, 700      |
| 554, 480      | 350, 200        |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image as implemented in the IPython notebook (code cell 8 through 12).

![alt text][image5]

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)
[![IMAGE ALT TEXT](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE "Video Title")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

