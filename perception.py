#!/usr/bin/env python
import glob
import numpy as np
import cv2
from scipy import stats
import os
from moviepy.editor import VideoFileClip
from PIL import ImageFont, ImageDraw, Image

CAMERA_MATRIX = None
DISTORTION_COEF = None
PERSPECTIVE_MATRIX = None
INV_PERSPECTIVE_MATRIX = None
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 1280  # meters per pixel in x dimension


def grayscale(img):
    '''
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    '''
    Applies the Canny transform
    '''
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    '''
    Applies a Gaussian Noise kernel
    '''
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    '''

    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    '''
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on
    # the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    '''
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    '''
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    '''
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    '''
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def normalize_lines(grouped_lines, image_height):
    '''
    Given a set of lines, average them and extend them to the bottom of the screen
    '''
    lines_matrix = np.asarray(grouped_lines)
    f = np.matrix.flatten(lines_matrix)
    xi = f[::2]
    yi = f[1::2]

    slope, b, r_value, p_value, std_err = stats.linregress(xi, yi)

    middle_height = (image_height / 2) + 50
    image_middle_x = (middle_height - b) / slope
    image_bottom_x = (image_height - b) / slope

    return np.asarray([
        int(image_middle_x),
        int(middle_height),
        int(image_bottom_x), image_height
    ])


def draw_lines(img, lines, color=[0, 0, 255], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once
    you want to average/extrapolate the line segments you detect to map out the
    full extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lines = [l[0] for l in lines]
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        if slope > 0:
            left_lines.append(line)
        else:
            right_lines.append(line)

    line_a = normalize_lines(left_lines, img.shape[0])
    line_b = normalize_lines(right_lines, img.shape[0])

    for x1, y1, x2, y2 in [line_a, line_b]:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def init_globals():
    '''
    Initializes global variables
    '''
    global CAMERA_MATRIX, DISTORTION_COEF, PERSPECTIVE_MATRIX, INV_PERSPECTIVE_MATRIX
    if CAMERA_MATRIX is None:
        print('Initializing camera matrix')
        CAMERA_MATRIX, DISTORTION_COEF = get_calibrated_camera(9, 6)

    if PERSPECTIVE_MATRIX is None:
        print('Initializing perspective matrix')
        PERSPECTIVE_MATRIX = get_perspective_matrix()
        INV_PERSPECTIVE_MATRIX = np.linalg.inv(PERSPECTIVE_MATRIX)


def get_perspective_matrix():
    '''
    Returs the perspective matrix

    These points were retrieved via 'straight_lines1.jpg'
    '''
    src = np.float32([
        [598, 451],  # tl
        [683, 451],  # tr
        [1007, 660],  # br
        [292, 660],  # bl
    ])
    dest = np.float32([
        [275, 200],  # tl
        [930, 200],  # tr
        [930, 700],  # br
        [275, 700],  # bl
    ])

    return cv2.getPerspectiveTransform(src, dest)


def find_lane_pixels(binary_warped, image_path=''):
    '''
    Finds the lanes using a sliding window moving across a histogram
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    # if 'test4' in image_path:
    #     import pdb; pdb.set_trace()
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint: 1100]) + midpoint
    print('leftx_base: ', leftx_base)
    print('rightx_base: ', rightx_base)

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, image_path=''):
    '''
    Fits a polynomial to the image
    '''
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(
        binary_warped, image_path=image_path)

    # store the curves for our fit to be shown on the image
    pixel_left_fit = np.polyfit(lefty, leftx, 2)
    pixel_right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_pixel_points = pixel_left_fit[0] * ploty**2 + pixel_left_fit[1] * ploty + pixel_left_fit[2]
    right_pixel_points = pixel_right_fit[0] * ploty**2 + pixel_right_fit[1] * ploty + pixel_right_fit[2]

    # Fit a second order polynomial to each using `np.polyfit` in meter space
    left_fit = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    y_eval = np.max(out_img)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((
        1 + (2 * left_fit[0] * y_eval * YM_PER_PIX + left_fit[1])**2) **
                     1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((
        1 + (2 * right_fit[0] * y_eval * YM_PER_PIX + right_fit[1])**2) **
                      1.5) / np.absolute(2 * right_fit[0])

    # [-4.01757621e-04 -2.42389445e-02  5.01017705e+02]
    print('left_fit: ', left_fit)
    # [ 4.02548309e-04 -8.65974458e-01  1.39781836e+03]
    print('right_fit: ', right_fit)

    # single number like 1246.5374760353297
    print('left_curverad: ', left_curverad)
    # single number like 1246.5374760353297
    print('right_curverad: ', right_curverad)
    return (
        out_img,
        left_pixel_points,
        right_pixel_points,
        left_fit,
        right_fit,
        left_curverad,
        right_curverad)


def to_color_binary(image, image_path=''):
    '''
    Converts an image to its color binary
    '''
    ##########################################
    # S Channel - good for yellow lines
    ##########################################
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    if image_path:
        write_output(s_channel, image_path, '2_1_saturation_channel')

    # Threshold color channel
    s_thresh_min = 140
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    if image_path:
        write_output(s_binary * 255, image_path, '2_2_saturation_binary')

    ##########################################
    # Grayscale
    ##########################################
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(
        sobelx
    )  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    if image_path:
        write_output(sobelx, image_path, '3_1_gray_channel')

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    if image_path:
        write_output(sxbinary * 255, image_path, '3_2_gray_binary')

    ##########################################
    # L Channel - good for yellow lines
    ##########################################
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    sobel_l = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    sobel_l = np.uint8(255 * sobel_l / np.max(sobel_l))
    if image_path:
        write_output(sobel_l, image_path, '4_1_sobel_l_channel')

    thresh_min = 25
    thresh_max = 150
    lbinary = np.zeros_like(scaled_sobel)
    lbinary[(sobel_l >= thresh_min) & (sobel_l <= thresh_max)] = 1
    if image_path:
        write_output(lbinary * 255, image_path, '4_2_l_binary')

    ##########################################
    # B Channel - good for yellow lines
    ##########################################
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    sobel_b = np.absolute(cv2.Sobel(b_channel, cv2.CV_64F, 1, 0))
    sobel_b = np.uint8(255 * sobel_b / np.max(sobel_b))
    if image_path:
        write_output(sobel_b, image_path, '5_1_sobel_b_channel')

    thresh_min = 50
    thresh_max = 100
    b_binary = np.zeros_like(scaled_sobel)
    b_binary[(sobel_b >= thresh_min) & (sobel_b <= thresh_max)] = 1
    if image_path:
        write_output(b_binary * 255, image_path, '5_2_b_binary')

    ##########################################
    # Combined
    ##########################################
    # Stack each channel to view their individual contributions in green and
    # blue respectively. This returns a stack of the two binary images, whose
    # components you can see as different colors
    color_binary = np.dstack(
        (s_binary,
         b_binary,
         lbinary,
         )
    ) * 255

    if image_path:
        write_output(color_binary, image_path, '6_color_binary')

    return color_binary


def add_lanes(image, image_path=''):
    '''
    Adds lanes to an image and returns the image
    '''
    init_globals()
    global CAMERA_MATRIX, DISTORTION_COEF, PERSPECTIVE_MATRIX, INV_PERSPECTIVE_MATRIX

    print('\nOn image_path: ', image_path)
    image_size = image.shape[1], image.shape[0]

    # undistort the image
    undistorted_image = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_COEF,
                                      None, CAMERA_MATRIX)
    if image_path:
        write_output(undistorted_image, image_path, '1_undistorted')

    color_binary = to_color_binary(undistorted_image, image_path=image_path)

    # convert the image into a birds-eye view with a perspective matrix
    warped = cv2.warpPerspective(
        color_binary, PERSPECTIVE_MATRIX, image_size, flags=cv2.INTER_LINEAR)
    if image_path:
        write_output(warped, image_path, '6_warped')

    grayscale_warped = grayscale(warped)
    if image_path:
        write_output(grayscale_warped, image_path, '7_grayscale_warped')

    ret, binary_warped = cv2.threshold(grayscale_warped, 1, 255, cv2.THRESH_BINARY)
    if image_path:
        write_output(binary_warped, image_path, '8_binary_warped')

    out_img, left_pixel_points, right_pixel_points, left_curve, right_curve, left_radius, right_radius = (
        fit_polynomial(binary_warped, image_path=image_path))

    # the middle of the lane is the mean of the left x value and the right x
    # value of the two edges
    middle_of_lane = (right_pixel_points[-1] + left_pixel_points[-1]) / 2
    offset = (middle_of_lane - (1280 / 2)) * XM_PER_PIX

    if image_path:
        write_output(out_img, image_path, '9_fitted_lines')

    painted_lane = get_painted_lane(
        undistorted_image,
        binary_warped,
        left_pixel_points,
        right_pixel_points)

    # Add text to the final image with curve radius and offset
    final_output = Image.fromarray(painted_lane)
    draw = ImageDraw.Draw(final_output)
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 50)
    info_text = "Radius: {}.  Offset: {}".format(left_radius, offset)
    draw.text((0, 0), info_text, font=font)

    # Save the image
    if image_path:
        write_output(np.array(final_output), image_path, '10_painted_lane')

    return np.array(final_output)


def get_painted_lane(undistorted_image, binary_warped, left_curve, right_curve):
    '''
    Returns an image of the lane painted
    '''
    image_size = undistorted_image.shape[1], undistorted_image.shape[0]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_curve, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp,
        INV_PERSPECTIVE_MATRIX,
        image_size)

    # Combine the result with the original image
    return cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)


def write_output(image, image_path, step_name,
                 output_folder='./output_images'):
    '''
    Writes an image to a file
    '''
    file_path = image_path.split('/')[-1]
    file_name, extension = file_path.split('.')
    output_file = os.path.join(output_folder, '{}_{}.{}'.format(
        file_name, step_name, extension))
    # print('Writing out {}'.format(output_file))
    cv2.imwrite(output_file, image)


def process_test_images():
    '''
    Given an image, it will add the lanes to the image and will write it
    to the 'test_images_output' directory
    '''
    image_paths = glob.glob('test_images/*.jpg')

    for image_path in image_paths:
        image = cv2.imread(image_path)
        add_lanes(image, image_path)


def get_calibrated_camera(x, y):
    '''
    Returns a calibrated camera matrix and distorion coefficients
    '''
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:y, 0:x].T.reshape(-1, 2)

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, file_name in enumerate(images):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (y, x), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Test undistortion on an image
    example_image = cv2.imread(images[0])
    img_size = (example_image.shape[1], example_image.shape[0])

    # Do camera calibration given object points and image points
    ret, camera_matrix, distortion_coef, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    return camera_matrix, distortion_coef


def process_video(video_path):
    '''
    Adds lanes to a video
    '''
    clip1 = VideoFileClip(video_path)
    clip_output = clip1.fl_image(add_lanes)
    clip_output.write_videofile('output_images/z_project_video_output.mp4', audio=False)


if __name__ == '__main__':
    process_test_images()

    process_video('./project_video.mp4')
