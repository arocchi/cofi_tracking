#!/usr/bin/env python

__author__ = 'Alessio Rocchi'

import cv2
import time
import numpy as np
import cofi.generators.color_generator as cg
import cofi.trackers.color_tracker as ct
import argparse


# can go from 0 (hard colors) to 1 (soft colors)
COLOR_MARGIN = 0.52

NUM_COLORS = 12
ok = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', nargs='?', default='', help="a local image file, or a number representing a camera id")
    parser.add_argument("--realsense", help="use the Realsense",action='store_true')
    args = parser.parse_args()

    img_mode = False
    opencv_mode = False
    realsense_mode = False
    camera_index = 0

    realsense_grabber = None    # RealSense grabber
    cap = None                  # OpenCV grabber

    if args.realsense:
        import RealSense.best_fast_grabber as gr
        realsense_mode = True
        realsense_grabber = gr.best_fast_grabber()

    if args.img_name != '':
        try:
            camera_index = int(args.img_name)
            opencv_mode = True
            cap = cv2.VideoCapture(camera_index)

        except:
            img_mode = True


    colors = cg.get_hsv_equispaced_hues(NUM_COLORS)

    hue_filters = list()

    for color in colors:
        h,_,_ = color
        threshold = COLOR_MARGIN*(360.0/NUM_COLORS)

        h_min = 2*h - threshold/2
        if h_min < 0:
            h_min += 360
        h_min /= 2

        h_max = 2*h + threshold/2
        if h_max > 360:
            h_max -= 360
        h_max /= 2

        hue_filters.append((h_min , h_max, h))

    while(ok):

        if(realsense_mode):
            (color_image, cloud, depth_uv, inverse_uv) = realsense_grabber.grab()
            frame = color_image.copy()
        elif(img_mode):
            frame = cv2.imread(args.img_name)
        else:
            # read the frames
            _,frame = cap.read()

        start_time = time.clock()

        centers = ct.detect_hues(frame, hue_filters)
        for center in centers:
            cx, cy, h = center
            bgr = cv2.cvtColor(np.array([[[h,255,255]]],np.uint8),cv2.COLOR_HSV2BGR)
            bgr = tuple(bgr.tolist()[0][0])
            cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 7,  bgr,     -1)

        # Show it, if key pressed is 'Esc', exit the loop
        cv2.imshow('frame',frame)

        end_time = time.clock()
        print "elapsed time", end_time - start_time

        if cv2.waitKey(33)== 27:
            break

        if img_mode and (cv2.waitKey() & 0xff) == ord('q'):
            ok = False

    cv2.destroyAllWindows()