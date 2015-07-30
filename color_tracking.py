#!/usr/bin/env python

__author__ = 'Alessio Rocchi'

import cv2
import numpy as np
import time
import cofi.generators.color_generator as cg

NUM_COLORS = 12
ok = True

if __name__ == "__main__":

    colors = cg.get_hsv_equispaced_hues(NUM_COLORS)

    hue_filters = list()

    for color in colors:
        h,_,_ = color
        print "h:", h
        threshold = 0.9*(360.0/NUM_COLORS)

        h_min = 2*h - threshold/2
        if h_min < 0:
            h_min += 360
        h_min /= 2

        h_max = 2*h + threshold/2
        if h_max > 360:
            h_max -= 360
        h_max /= 2

        #if h_min > h_max:
        #    temp = h_max
        #    h_max = h_min
        #    h_min = temp

        hue_filters.append((h_min , h_max, h))

    print hue_filters
    while(ok):

        # read the frames
        #_,frame = cap.read()
        frame = cv2.imread("in.jpg")

        start_time = time.clock()
        # smooth it
        frame = cv2.blur(frame,(3,3))

        # convert to hsv and find range of colors
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        for hue_filter in hue_filters:
            h_min, h_max, h = hue_filter

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
                    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    cv2.circle(frame,(cx,cy),7,(0,0,0),-1)
                    bgr = cv2.cvtColor(np.array([[[h,255,255]]],np.uint8),cv2.COLOR_HSV2BGR)
                    cv2.circle(frame,(cx,cy),5,tuple(bgr.tolist()[0][0]),-1)


        # Show it, if key pressed is 'Esc', exit the loop
        cv2.imshow('frame',frame)

        end_time = time.clock()
        print "elapsed time", end_time - start_time

        #if cv2.waitKey(33)== 27:
        #    break
        while (cv2.waitKey() & 0xff) != ord('q'): pass

        ok = False

    cv2.destroyAllWindows()