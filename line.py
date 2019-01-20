
import numpy as np
from collections import deque

QLEN = 8

class Line(object):
    """ from #2.Tips and Tricks for the Project """
    def __init__(self, yp=None, xp=None):

        self.ym_per_pix = yp
        self.xm_per_pix = xp
        # self.frame_shape = fs

        # was the line detected in the last iteration?
        self.detected = False


        # --- self.left_fitx & self.right_fitx
        # x values of the last n fits of the line
        # self.recent_xfitted = []
        self.recent_xfitted = deque(maxlen=QLEN)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        # --- using polyfit self.lfit & self.rfit
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fitQ = deque(maxlen=QLEN)

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        self.diffs_prev = np.array([0,0,0], dtype='float')


        # --- values for a given frame, may not be used ----
        # --- polyfit self.lfit & self.rfit
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None  # m
        #distance in meters of line from edge
        self.line_base_pos = None
        # --- leftx ?
        #x values for detected line pixels
        self.allx = None
        # --- lefty ?
        #y values for detected line pixels
        self.ally = None
        self.line_fitx = None
        # ---


    def add_all_pixels(self, ax, ay, warp_shape):
        self.allx = ax
        self.ally = ay
        # print("add_all_pixels-shape:{}".format(warp_shape))
        if ((len(self.ally) == 0) or (len(self.allx) == 0)):
            self.detected = False
        else:
            self._fit_line_polynomial(warp_shape)


    def use_starting_values(self):
        """ Starting values are used for the first frame and to realign the values
        when a detected position goes off track
        """
        self.detected = True
        self.recent_xfitted.append(self.line_fitx)
        self.bestx = self.line_fitx
        self.best_fit = self.current_fit
        self.best_fitQ.append(self.current_fit)



        # print("use_starting_values-best_fit:{}".format(self.best_fit))


    def use_staged_values(self):
        """ Staged values are typically used for most frames. It takes the 'temporary'
        values calculated by _fit_line_polynomial() and updates the line deque's and the
        averaged values.
        """
        # self.detected = True
        # self.detected = True
        self.recent_xfitted.append(self.line_fitx)
        if (len(self.recent_xfitted)> QLEN):
            self.recent_xfitted.popleft()

        # self.bestx = self.line_fitx
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.best_fitQ.append(self.current_fit)
        if (len(self.best_fitQ)> QLEN):
            self.best_fitQ.popleft()
        self.best_fit = np.mean(self.best_fitQ, axis=0)
        # print("\n{:.2f}:current_fit:{}".format(self.line_base_pos, self.current_fit))
        # print("{:.2f}:best_fit:{}".format(self.line_base_pos, self.best_fit))

        a = self.best_fitQ[0]
        b = self.best_fitQ[-1]
        self.diff = np.polysub(a, b)
        self.diffs_prev = np.polysub(self.best_fitQ[-2], self.current_fit)
        # print("{:.2f}:diff:{}".format(self.line_base_pos, self.diff))
        # print("len:{}".format(len(self.best_fitQ)))

    def discard_staged_values(self):
        self.detected = False

    def _fit_line_polynomial(self, frame_shape):
        """ from lesson 9.4
            Combined the polyfit(), lines, curves and other calculations into this
            single method as all the necessary data was right here.
        """

        # --- coefficients of line
        line_fit = np.polyfit(self.ally, self.allx, 2)
        _ploty = np.linspace(0, frame_shape[0]-1, frame_shape[0])

        try:
            # x points of line
            line_fitx = line_fit[0]*_ploty**2 + line_fit[1]*_ploty + line_fit[2]
            x_intercept = line_fit[0]*frame_shape[0]**2 + line_fit[1]*frame_shape[0] + line_fit[2]

        except TypeError:
            line_fitx = 1*_ploty**2 + 1*_ploty
            x_intercept = 0

        # --- curvature  recalculate to convert from pixels to meters--
        y_eval = np.max(_ploty)*self.ym_per_pix # convert from p to m
        line_fit_m = np.polyfit((self.ally*self.ym_per_pix), (self.allx*self.xm_per_pix), 2) # convert from p to m


        radius_curve = (np.sqrt (np.power((1+((2*line_fit_m[0]*y_eval)+(line_fit_m[1] ))**2), 3))) / abs(2*((line_fit_m[0])))


        self.line_base_pos = x_intercept * self.xm_per_pix # np.max(line_fitx) # abs(((frame_shape[1]/2)-(np.max(line_fitx)))) * self.xm_per_pix
        self.current_fit = line_fit
        self.radius_of_curvature = radius_curve # (radius_curve * self.ym_per_pix)
        self.line_fitx = line_fitx




