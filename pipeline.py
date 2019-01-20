#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from moviepy.editor import VideoFileClip

from utils import obtainCalibrationPointsFromFile
from utils import save_frame, load_frame
from utils import threshold_hls
from line import Line


class Pipeline(object):

    def __init__(self):
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/800 # 700 # meters per pixel in x dimension


        self.cMatrix, self.dCoeff = obtainCalibrationPointsFromFile()

        # --- set up line instances ---
        self.lf_line = Line(self.ym_per_pix, self.xm_per_pix)
        self.rt_line = Line(self.ym_per_pix, self.xm_per_pix)

        #--- perspective ---
        self.M = None

        self.standard_poly_diff_threshold = 0.0060
        self.standard_margin = 55
        self.standard_pixel = 550
        self.standard_lane = 3.5


        self.margin = 55
        self.left_curverand = None
        self.right_curverad = None
        self.left_curverand_real = None
        self.right_curverad_real = None
        self.avg_curverad = None
        self.left_fitx = None
        self.right_fitx = None

        self.lfit = None
        self.rfit = None
        self.ploty = None

        self.fitcount = 0

        self.curve_diff = 0
        self.poly_diff = 0
        self.poly_skip = False
        self.poly_skip_count = 0

        self.poly_diff_threshold = copy.copy(self.standard_poly_diff_threshold)
        self.window_margin = copy.copy(self.standard_margin)
        self.window_segments = 15
        self.min_pixels = copy.copy(self.standard_pixel)
        self.target_lane = copy.copy(self.standard_lane)

        self.sanity_control = ""
        self.sanity_error_string = ""


    def undistort(self, frame):
        """ calibrations obtained from calibration images and saved to file using examples from past lessons as examples """
        return cv2.undistort(frame, self.cMatrix, self.dCoeff, None, self.cMatrix )


    def threshold_hls_bgr(self, frame, s_thresh=(170, 255), x_gradient_thresh=(20, 100)):
        """From lesson 8.12 -- Combine thresholds

        """
        cp_frame = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- convert to HLS color space ---
        hls = cv2.cvtColor(cp_frame, cv2.COLOR_BGR2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]


        # --- sobel x using gray ---
        # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3) # take the derivative in x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # take the derivative in x
        abs_sobelx = np.absolute(sobelx) # accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # --- threshold x gradient ---
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= x_gradient_thresh[0]) & (s_channel <= x_gradient_thresh[1])] = 1


        # --- threshold color channel ---
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # --- stack  each channel ---
        # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary



    def perspective_warp(self, frame):
        """ taken from a number of lessons: 7.18, 7.16(transform) project 1 --> mask_vertices(imshape)


        WRITEUP- The warping to a birds eye view is complete but I am not sure if it works. We will have to see
        if the next part of the code works.

        """

        # --- source image shape ----
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

        # --- Transform image -----
        self.M = cv2.getPerspectiveTransform(src, dst)
        # print("warp - imagesize:{}".format(imagesize))
        warped = cv2.warpPerspective(frame, self.M, imagesize, flags=cv2.INTER_LINEAR)
        return warped



    def find_lane_pixels(self, warped_frame):
        """ from lesson 9.4

        TODO - Remove visualization
        """
        # take histogram of bottom half of image
        histogram = np.sum(warped_frame[warped_frame.shape[0]//2:,:], axis=0)

        # Find peak of left & right halves
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        # 'hyperparamters' --> 'sliding windows'
        n_windows = self.window_segments
        # width of windows +/- margin
        margin = self.window_margin
        # number to recenter window
        minimum_pixals = self.min_pixels

        # set height of windows
        window_height = np.int(warped_frame.shape[0]//n_windows)
        # id x/y positions of all nonzero pixels
        nonzero = warped_frame.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # current positions to be updated later for each window in n_windows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # empty lists to track left/right pixel indicies
        left_lane_inds = []
        right_lane_inds = []

        # search
        for window in range(n_windows):
            win_y_floor = warped_frame.shape[0] - (window + 1) * window_height
            win_y_ceil = warped_frame.shape[0] - window * window_height

            right_max = warped_frame.shape[1]
            if (leftx_current < margin):
                win_xleft_low = 0
            else:
                win_xleft_low = int(leftx_current - margin)
            win_xleft_high = int(leftx_current + margin)

            win_xright_low = int(rightx_current - margin)
            if (rightx_current > (right_max - margin)):
                win_xright_high = int(right_max)
            else:
                win_xright_high  = int(rightx_current + margin)

            # --- id nonzero pixels within window
            good_left_inds = ((nonzero_y >= win_y_floor) & (nonzero_y < win_y_ceil) & (nonzero_x >= win_xleft_low) &  (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_floor) & (nonzero_y < win_y_ceil) & (nonzero_x >= win_xright_low) &  (nonzero_x < win_xright_high)).nonzero()[0]


            # --- append to lists ---
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)


            # --- recenter if necessery ---
            if (len(good_left_inds) > minimum_pixals):
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))

            if (len(good_right_inds) > minimum_pixals):
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))


        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:

            pass

        # --- extract lane pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]

        # print("find_lane_pixels-shape:{}".format(warped_frame.shape))

        self.lf_line.add_all_pixels(leftx, lefty, warped_frame.shape)
        self.rt_line.add_all_pixels(rightx, righty, warped_frame.shape)



    def search_around_polynomial(self, warped_frame):
        """ lesson 9.5 """
        # self.margin = 100
        nonzero = warped_frame.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_fit = self.lf_line.best_fit
        right_fit = self.rt_line.best_fit

        # set area of search based on activated x-values
        left_lane_inds = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y +
           left_fit[2] - self.margin)) & (nonzero_x < (left_fit[0]*(nonzero_x**2) +
           left_fit[1]*nonzero_x + left_fit[2] + self.margin)))

        right_lane_inds = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y +
           right_fit[2] - self.margin)) & (nonzero_x < (right_fit[0]*(nonzero_y**2) +
           right_fit[1]*nonzero_y + right_fit[2] + self.margin)))

        # --- extract left and right lane pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]


        self.lf_line.add_all_pixels(leftx, lefty, warped_frame.shape)
        self.rt_line.add_all_pixels(rightx, righty, warped_frame.shape)



    def overlay_path_and_calculations(self,  frame):
        """ from project 2.2 tips and tricks """
        # create blank image

        color_warp = np.zeros_like(frame).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        frame_shape = frame.shape
        ploty = np.linspace(0, frame_shape[0]-1, frame_shape[0])

        # print(self.lf_line.bestx)

        #  recast to cv2.fillpoly usable format
        pts_left = np.array([np.transpose(np.vstack([self.lf_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.rt_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # print("overlay_path_and_calculations-bestx:\n{}".format(self.rt_line.bestx))

        # print("colorwarp shape:{}".format(color_warp.shape))
        # draw lane onto blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        newwarp = cv2.warpPerspective(color_warp, self.M, (frame_shape[1], frame_shape[0]), flags=cv2.WARP_INVERSE_MAP)
        overlayed_frame = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)



        # --- calculate offset ---
        # print(frame.shape)
        left_lanex = self.lf_line.line_base_pos # m
        right_lanex = self.rt_line.line_base_pos # m
        # print("xleft::{} xright::{}".format(left_lanex, right_lanex))
        lane_offset = ((right_lanex - left_lanex) / 2) + left_lanex # m
        frame_offset = (frame.shape[1] / 2) * self.xm_per_pix # m
        # print("lane_offset:{} frame_offset:{}".format(lane_offset, frame_offset))
        lane_offset_m = lane_offset
        frame_offset_m = frame_offset
        alignment = frame_offset_m - lane_offset_m

        lane_dist = (right_lanex - left_lanex)
        # print("lane_offset:{}m frame_offset:{}m  -- alignment:{}".format(lane_offset_m, frame_offset_m, alignment))

        if (alignment > 0):
            position = "right"
        else:
            position = "left"
        offset_string = "Vehicle is {:.2}m {} of center ".format(abs(alignment), position)

        # --- add detected radius ----
        # radius_string = "RC:{:}m lf:{:.2f}m rt:{:.2f}m dis:{:.2f}m cd:{:.2f}".format(self.avg_curverad,
        #  self.lf_line.radius_of_curvature, self.rt_line.radius_of_curvature,
        #  lane_dist, self.curve_diff)
        radius_string = "Radius of Curvature:{:}m".format(self.avg_curverad)
        overlayed_frame = cv2.putText(overlayed_frame, radius_string, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2 )
        overlayed_frame = cv2.putText(overlayed_frame, offset_string, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2)


        # --- error handling  message----
        # if(len(self.lf_line.best_fitQ) > 1):
        #     lfdiff = "left_lane: last:{:.8f}  now:{:.8f}".format(self.lf_line.best_fitQ[-2][0],  self.lf_line.best_fitQ[-1][0])
        # else:
        #     lfdiff = "left_lane: now:{:.8f}".format(  self.lf_line.best_fitQ[-1][0])

        # if(len(self.lf_line.best_fitQ) > 1):
        #     rtdiff = "right_lane:last:{:.8f}   now:{:.8f}".format(self.rt_line.best_fitQ[-2][0], self.rt_line.best_fitQ[-1][0])
        # else:
        #     rtdiff = "right_lane:   now:{:.8f}".format( self.rt_line.best_fitQ[-1][0])

        # lfbest = "best:{:.8f}   poly2:{:.8f}".format(self.lf_line.best_fit[0], self.poly_diff)

        # rtbest = "best:{:.8f}   poly_skip_count:{} ".format(self.rt_line.best_fit[0], self.poly_skip_count)
        # overlayed_frame = cv2.putText(overlayed_frame, lfdiff, (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2)
        # overlayed_frame = cv2.putText(overlayed_frame, lfbest, (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2)
        # overlayed_frame = cv2.putText(overlayed_frame, rtbest, (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2)

        # overlayed_frame = cv2.putText(overlayed_frame, rtdiff, (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 235), 2)

        # if(self.sanity_error_string is not ""):
        #         overlayed_frame = cv2.putText(overlayed_frame, self.sanity_error_string, (25, 265), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 105), 2)



        return overlayed_frame


    def sanity_check(self, warped_frame):
        """
        """
        if(self.lf_line.detected is False):
            self.sanity_error_string = "error:lf_line.detected=False"
            self.sanity_update("reset")
            return False
        elif (self.rt_line.detected is False):
            self.sanity_error_string = "error:rt_line.detected=False"
            self.sanity_update("reset")
            return False

       # --- lane width check ---
        left_lanex = self.lf_line.line_base_pos
        right_lanex = self.rt_line.line_base_pos
        lane_dist = abs((right_lanex - left_lanex))
        max_lane = 4.3
        if(lane_dist < self.target_lane):
            self.sanity_error_string = "error:lane: {} < {}".format(lane_dist, self.target_lane)
            return False
        elif(lane_dist > max_lane):
            self.sanity_error_string = "error:lane-reset: {:.2f} < {}".format(lane_dist, max_lane)
            self.sanity_update("reset")
            self.fitcount = self.fitcount + 1
            return "reset"
        else:
            self.sanity_update()
            self.fitcount = 0


        return True

    def sanity_update(self, err_version=None):

        if(err_version is "reset"):
            self.sanity_control = "reset"

        else:
            self.sanity_control = ""
            self.sanity_error_string = ""



    def sanity_check1(self, warped_frame):

        # print("Curvature:------------\nlf:{:.2f}\nrt:{:.2f}".format(self.lf_line.radius_of_curvature, self.rt_line.radius_of_curvature))

        if(self.lf_line.detected is False):
            self.sanity_error_string = "error:lf_line.detected=False"
            return False
        elif (self.rt_line.detected is False):
            self.sanity_error_string = "error:rt_line.detected=False"
            return False


        # --- radius check ---
        lfcurve = self.lf_line.radius_of_curvature
        rtcurve = self.rt_line.radius_of_curvature
        self.curve_diff = rtcurve - lfcurve
        if(abs(self.curve_diff) > 10000):
            self.sanity_error_string = "error:curve_diff>10000"
            return False



        # --- poly Test ----
        should_return_false = False
        lfpoly2 = self.lf_line.best_fitQ[-1][0]
        rtpoly2 = self.rt_line.best_fitQ[-1][0]
        self.poly_diff = rtpoly2 - lfpoly2

        if(abs(self.poly_diff) > self.poly_diff_threshold):
            self.sanity_error_string = "error:poly_diff: {} > {}".format(self.poly_diff, self.poly_diff_threshold)
            #     self.poly_skip = True
            #     self.poly_skip_count = self.poly_skip_count + 1


            #     if(self.poly_skip_count > 10):
            #         self.target_lane = copy.copy(self.standard_lane)
            #         self.window_margin = copy.copy(self.standard_margin)
            #         self.min_pixels = copy.copy(self.standard_pixel)
            #         self.poly_diff_threshold = copy.copy(self.standard_poly_diff_threshold)
            #         self.find_lane_pixels(warped_frame)
            #         self.lf_line.use_starting_values()
            #         self.rt_line.use_starting_values()

            #         self.poly_skip_count = 0
            #         return True
            #     elif(self.poly_skip_count > 6):
            #         self.window_margin = self.window_margin + 50
            #         # self.min_pixels = self.min_pixels + 75
            #         # self.poly_diff_threshold = self.poly_diff_threshold * .9
            #         self.find_lane_pixels(warped_frame)
            #         return True
            #     elif(self.poly_skip_count > 3):
            #         # self.target_lane = 3.2
            #         # self.min_pixels = self.min_pixels - 50
            #         self.find_lane_pixels(warped_frame)
            #         return True



            return False

        # else:
        #     self.poly_skip = False


        # --- lane width check ---
        left_lanex = self.lf_line.line_base_pos
        right_lanex = self.rt_line.line_base_pos
        lane_dist = abs((right_lanex - left_lanex))
        if(lane_dist < self.target_lane):
            self.sanity_error_string = "error:lane: {} < {}".format(lane_dist, self.target_lane)
            return False

        # if (should_return_false):
        #     return False


        # xiLeft = self.lf_line.line_base_pos
        # xiRight = self.rt_line.line_base_pos
        # val = (xiRight-xiLeft)*self.xm_per_pix
        # print("lane width estimate:{:.2f}m & pixels{:.0f}".format(val, xiRight-xiLeft))
        return True

    def process_lane(self, warped_frame):
        """
        for each lane:
            was last lane detected and/valid
            if so, fit using last polynomial
                sanity check --> use fit
                sanity check --> use last fit and track


        """
        if ((self.lf_line.bestx is None) or (self.rt_line.bestx is None)):
            self.find_lane_pixels(warped_frame)
            self.lf_line.use_starting_values()
            self.rt_line.use_starting_values()

        else:
            self.search_around_polynomial(warped_frame)
            # self.find_lane_pixels(warped_frame)



        checked = self.sanity_check(warped_frame)

        if (checked is True ):

            self.lf_line.use_staged_values()
            self.rt_line.use_staged_values()

        elif(checked is "reset"):
            self.find_lane_pixels(warped_frame)
            # self.sanity_update()
            # --- reset to begining values
            if(self.fitcount > 3):
                self.fitcount = 0
                self.lf_line.use_starting_values()
                self.rt_line.use_starting_values()



        self.avg_curverad = int((self.lf_line.radius_of_curvature + self.rt_line.radius_of_curvature) / 2)




    def process_frame(self, frame):

        # --- entire image and both lanes ---
        t1 = self.undistort(frame)
        at1 = self.threshold_hls_bgr(t1)
        at1 = self.perspective_warp(at1)


        # --- individual lanes ---
        # --- sanity check ---
        # look at lines and see if line detected
        self.process_lane(at1)

        # --- entire image ---
        at1 = self.overlay_path_and_calculations( t1)

        return at1


    def extract_and_process_video(self, fName, outFileName):
        """from first project """
        clip = VideoFileClip(fName) # .subclip(41, 45)
        overlay_clip = clip.fl_image(self.process_frame)
        overlay_clip.write_videofile(outFileName, audio=False)






if __name__ == "__main__":


    pl = Pipeline()

    fName = "project_video.mp4"
    # fName = "challenge_video.mp4"
    out = "all_writeup/{}".format(fName)
    pl.extract_and_process_video(fName, out)

    # fName1 = "straight_lines1.jpg"
    # fName = "straight_lines2.jpg"
    # fName = "test2.jpg"
    # fName = "test1.jpg"
    # fName = "test3.jpg"


    # fName1 = "test5.jpg"
    # fName2 = "test6.jpg"
    # fName3 = "test7.jpg"
    # fName = "test6.jpg"



    # --- pipe test
    # load image, process and obtain lines, save to file
    # load 2nd image, fit lines to file

    # fName1 = fName
    # t1 = load_frame("test_images/{}".format(fName1))
    # at1 = pl.process_frame(t1)
    # save_frame(at1, fName1, "all_writeup", "refactor_1_")

    # t2 = load_frame("test_images/{}".format(fName2))
    # at2 = pl.process_frame(t2)
    # save_frame(at2, fName2, "all_writeup", "refactor_1_")

    # t3 = load_frame("test_images/{}".format(fName3))
    # at3 = pl.process_frame(t3)
    # save_frame(at3, fName3, "all_writeup", "refactor_1_")

