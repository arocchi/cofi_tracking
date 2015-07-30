#!/usr/bin/env python

__author__ = 'Alessio Rocchi'

import cv2
import numpy as np

def detect_hues(frame, hue_filters):
    """
    detects hues in an image based on a list of hue filter thresholds
    :param frame: the input frame
    :param hue_filters: a list of tuples (hue_min, hue_max)
    :return:
    """
    # smooth frame
    frame = cv2.blur(frame,(3,3))

    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    centers = list()

    for hue_filter in hue_filters:
        if len(hue_filter) == 3:
            h_min, h_max, h = hue_filter
        else:
            h_min, h_max = hue_filter
            h = (h_max - h_min)/2
            if h < 0:
                h += 180

        if h_min > h_max:
            thresh_l = cv2.inRange(hsv,np.array((h_min, 80, 80)), np.array((360.0/2, 255, 255)))
            thresh_u = cv2.inRange(hsv,np.array((0.0, 80, 80)),     np.array((h_max, 255, 255)))
            thresh = cv2.add(thresh_l, thresh_u)
        else:
            thresh = cv2.inRange(hsv,np.array((h_min, 80, 80)), np.array((h_max, 255, 255)))
        thresh2 = thresh.copy()
        #cv2.imshow('thresh'+str(h),thresh2)

        # find contours in the threshold image
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # finding contour with maximum area and store it as best_cnt
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt

                # finding centroids of best_cnt and draw a circle there
                M = cv2.moments(best_cnt)
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                centers.append((cx,cy))

    return centers
