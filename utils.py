
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import numpy as np
import cv2


# def cameraCalibration(fileName):

#     cornersx = 9
#     cornersy = 6

#     img = cv2.imread(fileName)

#     print('This image is:', type(img), 'with dimensions:', img.shape)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (cornersx, cornersy), None)

#     if ret == True:
#         cv2.drawChessboardCorners(img, (cornersx, cornersy), corners, ret)

#         cv2.imwrite('test2.jpg', img)
#     else:
#         print("findChessboardCorners did not find enough corners or otherwise failed")

# cameraCalibration("camera_cal/calibration2.jpg")

def obtainCalibrationPointsFromFile(fname="all_writeup/camera_calibrations_pickle1.p"):
    """
    Uses pickle to open/unpack camera calibration values from file.
    Expects pickle with the two following keys:
    'cMatrix'   ==  camera matrix
    'dCoeff'    ==  distortion coefficients

    Assumes fname is a pickled file with no error checking.

    Input: fname - file name of pickle --> default is project file
    return: cMatrix, dCoeff

    """
    with open(fname, "rb") as f:
        dist_pickle = pickle.load(f)

    cMatrix = dist_pickle["cMatrix"]
    dCoeff = dist_pickle["dCoeff"]
    return cMatrix, dCoeff





def calibrateAndUndistort(img, objpoints, ipoints ):
    """ Test method supporting util.cameraCalibration. Not for pipeline.
    """
    img_size = img.shape[1::-1]
    retval, cMatrix, dCoeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, ipoints, img_size, None, None )
    undist = cv2.undistort(img, cMatrix, dCoeff, None, cMatrix)
    print("undistort")
    return undist, cMatrix, dCoeff

def cameraCalibration():
    """  Code sourced from color calibrations notebook and from lessons 11 & 12.

        Obtains the object and image points and saves to file.
        Should be rerun for each camera.

        WRITE_UP:
        Discovered that the find corners CV method will not find fewer corners
        in an image. calibration1.jpg only has 5 rows compared to the others
        which have 6. Ran with 5 on all images and the function took longer and
        ignored all the images except calibration1. Reran with 6 rows and most
        images were identified. The findChessboardCorners function really needs
        to see the whole image for it to work.
    """
    DEBUG = 0

    cornersx = 9
    cornersy = 6
    objp = np.zeros((cornersy*cornersx,3), np.float32)
    objp[:,:2] = np.mgrid[0:cornersx, 0:cornersy].T.reshape(-1,2)

    objpoints = []
    ipoints = []

    files = glob.glob('camera_cal/calibration*.jpg')
    if DEBUG: print("image files found: {}".format(len(files)))

    for idx, fname in enumerate(files):
        if DEBUG: print("{}".format(fname))

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cornersx, cornersy), None)

        if ret == True:
            objpoints.append(objp)
            ipoints.append(corners)
            if DEBUG: print("added {} to points".format(fname))
        else:
            if DEBUG: print("{} corners not found".format(fname))
            pass

    # --- save calibration files to file ---
    save_pickle = {}
    save_pickle["cMatrix"] = cMatrix
    save_pickle["dCoeff"] = dCoeff
    pickle.dump(save_pickle, open("all_writeup/camera_calibrations_pickle1.p", "wb"))

    # --- test calibrating an individual file ---
    # img = cv2.imread('camera_cal/test2.jpg')
    # ajusted_image, cMatrix, dCoeff = calibrateAndUndistort(img, objpoints, ipoints )
    # write_name = 'all_writeup/undistorted'+"test2"+'.jpg'
    # cv2.imwrite(write_name, ajusted_image)

# cameraCalibration()



def load_frame( name):
    return cv2.imread(name)

def save_frame(frame, name="test1.jpg", path=None, process_tag="undistort"):
    name = "{}{}".format(process_tag, name)
    if path:
        name = "{}/{}".format(path, name)

    cv2.imwrite(name, frame)

def threshold_hls(self, frame, s_thresh=(170, 255), x_gradient_thresh=(20, 100)):
    """From lesson 8.12 """
    cp_frame = np.copy(frame)

    # --- convert to HLS color space ---
    hls = cv2.cvtColor(cp_frame, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # --- sobel x ---
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3) # take the derivative in x
    abs_sobelx = np.absolute(sobelx) # accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # --- threshold x gradient ---
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= x_gradient_thresh[0]) & (s_channel <= x_gradient_thresh[1])] = 1


    # --- threshold color channel ---
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # --- stack  each channel ---
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary








def perspective_warp_FOR_REFERENCE(self, frame):
    """ taken from a number of lessons: 7.18, 7.16(transform) project 1 --> mask_vertices(imshape)


    WRITEUP- The warping to a birds eye view is complete but I am not sure if it works. We will have to see
    if the next part of the code works.

    """

    # imagesize = frame.shape[1::-1]
    # --- source image shape ----
    imshape = frame.shape
    imagesize = imshape[1::-1]
    height_seg = imshape[0]/24
    width_seg = imshape[1]/24
    #     print("h:{} w:{} hm:{} wm:{}".format(height_seg, width_seg, height_seg * 20, width_seg * 20))
    # top_right = 13 # must be >= top_left
    # top_left = 11  # must be <= top_right
    top_right = 12.945 # must be >= top_left
    top_left = 11.069 # must be <= top_right
    top_edge = 15 # 15

    bottom_right = 20.26
    bottom_left = 3.75
    bottom_edge = 22.333

    #                      (width ,   height)
    # s = np.array([[(width_seg * top_right,  height_seg * top_edge),
    #                   ( width_seg  * top_left, height_seg * top_edge),
    #                   (width_seg * bottom_left, height_seg * bottom_edge),
    #                   (width_seg * bottom_right, height_seg * bottom_edge)]], dtype=np.int32)
    # s = [[(width_seg * top_right,  height_seg * top_edge),
    #           ( width_seg  * top_left, height_seg * top_edge),
    #           (width_seg * bottom_left, height_seg * bottom_edge),
    #           (width_seg * bottom_right, height_seg * bottom_edge)]]
    s = [[  (width_seg * top_left, height_seg * top_edge),
            (width_seg * top_right,  height_seg * top_edge),
            (width_seg * bottom_right, height_seg * bottom_edge),
            (width_seg * bottom_left, height_seg * bottom_edge)
        ]]

    '''
    (610, 425), (670, 425)
    (100, 720), (1180, 720)
    transformed_N_
    top_edge = 14.75;  top_left = 11;  top_right = 13;
    bottom_right = 22
    bottom_left = 2
    bottom_edge = 24

    top_right = 12.57 # must be >= top_left
    top_left = 11.44 # must be <= top_right
    top_edge = 14.46



    (590, 445), (690, 445)
    (200, 670), (1080, 670)
    top_right = 12.945 # must be >= top_left
    top_left = 11.069 # must be <= top_right
    top_edge = 14.84

    bottom_right = 20.26
    bottom_left = 3.75
    bottom_edge = 22.333
    of = 0
    lf = 0
    v = [[  (0 + of , 0), (imshape[1] - of , 0 ) , (imshape[1] - lf , imshape[0]), (0 + lf , imshape[0])]]




    [[[340. 445.]
      [940. 445.]
      [940. 720.]
      [340. 720.]]]


    [[[440. 445.]
      [840. 445.]
      [840. 720.]
      [440. 720.]]]



    2_2_


    v = [[  (540 , 445), (740 , 445 ) , (740 , 720), (540 , 720)]]


    [[[   0.    0.]
    [1280.    0.]
    [1280.  720.]
    [   0.  720.]]]


    '''

    src = np.float32(s)

    # src = np.float32([v])

    # --- perspective shift ---
    # v = [[(width_seg * bottom_right, height_seg * top_edge),
    #         (width_seg * bottom_left, height_seg * top_edge),
    #         (width_seg * bottom_left, height_seg * bottom_edge),
    #         (width_seg * bottom_right, height_seg * bottom_edge)]]

    of = 0
    lf = 0
    # # ---- (top_right, top_left, bottom_left, bottom_right)

    # v = [[ (0 + of, imshape[0]-of), (0 + of, 0+ of), (imshape[1] - of, 0 + of) , (imshape[1] - of, imshape[0]-of)]]
    # ----
    #   with shape (720,1280,3) --> shape points are in (x,y) order & sequenced ordered from (0,0), (1280,0), (1280,720), (0, 720)
    # v = [[ (0 + of, imshape[0]-of), (0 + of, 0+ of), (imshape[1] - of, 0 + of) , (imshape[1] - of, imshape[0]-of)]]

    # v = [[ (0 + of, imshape[0]-of), (0 + of, 0+ of), (imshape[1] - of, 0 + of) , (imshape[1] - of, imshape[0]-of)]]
    # v = [[  (0 + of, 0+ of), (imshape[1] - of, 0 + of) , (imshape[1] - of, imshape[0]-of), (0 + of, imshape[0]-of)]]
    # v = [[  (0 + of , 0), (imshape[1] - of , 0 ) , (imshape[1] - lf , imshape[0]), (0 + lf , imshape[0])]]



    # ---  transform_2_1_NAME
    # v = [[(0, 0), (imshape[1], 0), (imshape[1], imshape[0]), (0, imshape[0])]]

    # --- transform_2_3_NAME
    # v = [[  (540 , 445), (740 , 445 ) , (740 , 720), (540 , 720)]]

    # --- transform_2_4_NAME
    # v = [[  (200 , 450), (1080 , 450 ) , (1080 , 720), (200 , 720)]]

    # --- transform_2_5_NAME
    v = [[(width_seg * bottom_left, 0),(width_seg * bottom_right, 0 ), (width_seg * bottom_right, imshape[0]), (width_seg * bottom_left,imshape[0] ) ]]



    # ss = np.array(v, dtype=np.int32)


    # v = [[(0, 0), (imshape[1], 0), (imshape[1], imshape[0]), (0, imshape[0])]]
    dst = np.float32(v)
    # print("{}\n{}".format(src, dst))


    # print("imshape:{}, imagesize:{}, src:{}\ndst:{}".format(imshape, imagesize, src, dst))


    # v = [ corners[0][0], corners[nx-1][0], corners[-nx][0], corners[-1][0] ]
    # src = np.float32([v])
    # print(src)
    # x_off = (imagesize[0] * .2) / 2.0
    # y_off = (imagesize[1] * .2) / 2.0

    # # w = [[list(x_off, y_off), list(imagesize[0] - x_off, y_off), list(x_off, imagesize[1]- y_off), list(imagesize[0]-x_off, imagesize[1]-y_off)]]
    # w = [[x_off, y_off], [imagesize[0] - x_off, y_off], [x_off, imagesize[1]- y_off], [imagesize[0]-x_off, imagesize[1]-y_off]]
    # dst = np.float32(w)
        #defining a blank mask to start with


    # mask = np.zeros_like(frame)

    # #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    # if len(frame.shape) > 2:
    #     channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
    #     ignore_mask_color = (255,) * channel_count
    # else:
    #     ignore_mask_color = 255

    # #filling pixels inside the polygon defined by "vertices" with the fill color
    # cv2.fillPoly(mask, ss, ignore_mask_color)

    # #returning the image only where mask pixels are nonzero
    # masked_image = cv2.bitwise_and(frame, mask)
    # return masked_image


    # --- Transform image -----
    self.M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, self.M, imagesize, flags=cv2.INTER_LINEAR)
    return warped


