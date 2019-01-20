## Writeup Joshua Urlaub

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./all_writeup/image_calibrate/undistortedcalibration1.jpg "Undistorted Cal Image"
[image6]: ./all_writeup/undistortcal_test1.jpg "Undistorted Image"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./all_writeup/image_transform/thresholdstraight_lines1.jpg "Transform Example"

[image4]: ./all_writeup/image_transform/transformed_2_5_straight_lines1.jpg "Final Warped Points"
[image5]: ./all_writeup/image_transform/transformed_2_1_straight_lines1.jpg "Alternative Warp Example"
[image7]: ./all_writeup/pipeline_test1/PL_1_test7.jpg "Fit Visual"
[image8]: ./all_writeup/pipeline_test1/path_5_straight_lines1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


### Writeup

For this project, I wanted to step away from the jupyter notebook and run the code directly on the machine. The trade-off is that it was harder to visualize and plot the changes to the image files.

The overall project is organized using 3 files and 2 classes. Pipeline is the primary class that is invoked from the command line. Line is the second class that trackes two lines detected in the image. The utils file contains a couple of utility methods.

### Camera Calibration

The calibration code uses a set of methods from the utils.py file.
1. **obtainCalibrationPointsFromFile()** (ln 30 utils.py)
2. **cameraCalibration()** (ln 63 utils.py)

For this, I used the test calibration files provided for the project. Not all the provided images were valid. Essentailly, I learned that the cv2.findChessboardCorners really does fail if the inner & outer corners were not fully visible. Even trying to specify a subset of the corners is not valid. I generated the camera calibration and distortion coefficients and saved to a pickle file. This file is pulled into the Pipeline class when it is instantiated.

Here is a example of a calibrated image:

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This distortion corrected image from the test image folder uses the **Pipeline.undistort** method (ln 66 pipeline.py). The file uses the **cv2.undistort** method.
![alt text][image6]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Color transforms are handled in pipeline.py by the **threshold_hls_bgr()** method (ln 71). I use a sobelx grayscale and combine it with an s_channel threshold.
Here is the output from the testing.

![alt text][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform took a lot of experimentation. I saved some of the notes in the utils file for later reference. The primary transform is in the pipeline.py file called **perspective_warp()** (ln 109). It takes an image and returns an altered image. The altered matrix is saved in the class for later unwarping. The dimension points used for the warping are optimized to that camera mounted position but could be easily changed if necessary.

For example: using the bottom_edge = 22.333 sets the lower bounds of projected overlay just on top of the hood of the car.


```python
       imshape = frame.shape
        imagesize = imshape[1::-1]
        height_seg = imshape[0]/24
        width_seg = imshape[1]/24

        top_right = 12.945 # must be >= top_left
        top_left = 11.069 # must be <= top_right
        top_edge = 15 # 15

        bottom_right = 20.26
        bottom_left = 3.75
        # bottom_edge = 22.333
        bottom_edge = 24


        s = [[  (width_seg * top_left, height_seg * top_edge),
                (width_seg * top_right,  height_seg * top_edge),
                (width_seg * bottom_right, height_seg * bottom_edge),
                (width_seg * bottom_left, height_seg * bottom_edge)
            ]]

        src = np.float32(s)

        # --- transform_2_5_NAME
        v = [[(width_seg * bottom_left, 0),(width_seg * bottom_right, 0 ), (width_seg * bottom_right, imshape[0]), (width_seg * bottom_left,imshape[0] ) ]]

        dst = np.float32(v)

```
The source & destination points can be derived from the above code.

I used a variety of test images to ensure that the transform was working as expected. Interesting points to note is that the output image orientation is depenent on the order of the points.

This is an example of the image using the points in the code
![alt text][image4]

This is alternative example that I considered using. It basically matched the full dimensions of the image shape.
![alt text][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There sliding box method is named **find_lane_pixels()** (ln 155). Additionally, there is a fit to existing lines method **search_around_polynomial()** (ln 247). Both are used to identify new lines. Each sends Lane pixel positions to the *Line* class **add_all_pixels()** (ln 53 line.py). Future refactoring would see both methods cut in half. I would remove duplicate code and create a way to run an individual line with different configuration and locations. This would allow for tracking more road lines.

The polyfit function was moved to the *Line* class **_fit_line_polynomial()** (ln 101 line.py).

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the *Line* class **_fit_line_polynomial()** (ln 123 line.py).  The vehicle postion is found in *pipeline.py line 308-333* in the overlay class. The hard thing was to get the radius to align with the 1km mentioned in the
**tips and tricks** page. The key is probably the conversion from pixels to meters. If you look at **_fit_line_polynomial()** I calculate the curvature data separate from the pixel line fit data. The output values are still low.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The overlay results are plotted in *pipeline.py* **overlay_path_and_calculations()** (ln 278) Here is an example  of the test images generated.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./all_writeup/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?



Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
