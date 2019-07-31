
## Advanced Lane Finding Project


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the first iteration of   `add_lanes()` I make a call to `init_globals()` .  This initializes a camera matrix and distortion coefficients with a call to `get_calibrated_camera`.  

It imports the images from `camera_cal/` which contains images of chessboards from various angles and distances.  For each image, I get the chessboard corners using `cv2.findChessboardCorners` and append the corners to the list `imgpoints`.  I make a corresponding array `objpoints` that contains the list of the chessboard corners in object space, which we hardcoded by adding the `objp` defined by:
```
objp = np.zeros((x * y, 3), np.float32)
objp[:, :2] = np.mgrid[0:y, 0:x].T.reshape(-1, 2)
```
With the two lists were able to calibrate our camera matrix and distortion coefficients using: `cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)`

Now, we can undo the distortion caused by the camera lens with:
```
undistorted_image = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_COEF,
                                      None, CAMERA_MATRIX)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

With the distortion:
![with-distortion](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/test_images/straight_lines1.jpg)

Without the distortion
![without-distortion](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1_1_undistorted.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The following is all done in the `to_color_binary()` function.

I used a combination of the sobel edges generated from the a grayscale image and the saturation channel with thresholds to generate a binary image.  Here is the image transformed into the HLS color space and just showing the saturation channel:
![saturation](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1_2_saturation_channel.jpg)
Here is the image after converted to grayscale and processed through the Sobel x transform to find edges:
![gray-sobel](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1_5_color_binary.jpg)
The color-coded combination of the two outputs the following:'

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform code is performed in `add_lanes()` function.

I first create the perspective matrix using the `get_perspective_matrix()` function and set it to the global variable `PERSPECTIVE_MATRIX`.  I used the following points as source and destination for determining the matrix.
```python
    src = np.float32([
        [598, 451],  # tl
        [683, 451],  # tr
        [1007, 660], # br
        [292, 660],  # bl
    ])
    dest = np.float32([
        [275, 200],  # tl
        [930, 200],  # tr
        [930, 700],  # br
        [275, 700],  # bl
    ])
```

```python
warped = cv2.warpPerspective(
    color_binary, PERSPECTIVE_MATRIX, image_size, flags=cv2.INTER_LINEAR)
```
The transformation does the following:
![warped](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1_6_warped.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I identified lane-line pixels in `find_lane_pixels()`.

It takes the warped image and converts it into a histogram.  It then takes the bottom portion of the histogram and finds the two peaks.  One represents the left lane and the other represents the right lane.

I then specify the number of windows (9) to split up the lane into vertically.  You can see the result here:
![lanes](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1_9_fitted_lines.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

We calculated the radius of the curvature of the lane in `fit_polynomial`.

We fit a second order polynomial to the lane pixels found:
```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

We then used those coefficients in the formula for determining the radius of a curve in radians:
```python
# Calculation of R_curve (radius of curvature)
left_curverad = ((
    1 + (2 * left_fit[0] * y_eval * YM_PER_PIX + left_fit[1])**2) **
                 1.5) / np.absolute(2 * left_fit[0])
right_curverad = ((
    1 + (2 * right_fit[0] * y_eval * YM_PER_PIX + right_fit[1])**2) **
                  1.5) / np.absolute(2 * right_fit[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for painting a whole green is in `get_painted_lane`
![painted](https://raw.githubusercontent.com/NathanBWaters/CarND-Advanced-Lane-Lines/master/output_images/straight_lines2_10_painted_lane.jpg)


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/NathanBWaters/CarND-Advanced-Lane-Lines/blob/master/output_images/project_video_output.mp4)

You might have to download it in order for it to work.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There is still the problem that we have assumed that the left lane and right lane are on the left and right hand side of the image.  This is not the case when we are changing lanes.

The algorithm also confused some other lines with the lanes.  So if other imagery such as stripes on the side of another car appeared it would confuse my algorithm.

The lines are calculated in 2D space (xz), but in reality roads also move up and down (xyz).  So my algorithm would fail with hilly roads.
