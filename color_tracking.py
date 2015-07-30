#!/usr/bin/env python

__author__ = 'Alessio Rocchi'

import cv2
import time
import cofi.generators.color_generator as cg
import cofi.trackers.color_tracker as ct
import argparse

NUM_COLORS = 12
ok = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', nargs='?', default='')
    args = parser.parse_args()

    img_mode = False
    if args.img_name != '':
        img_mode = True


    colors = cg.get_hsv_equispaced_hues(NUM_COLORS)

    hue_filters = list()

    for color in colors:
        h,_,_ = color
        threshold = 0.9*(360.0/NUM_COLORS)

        h_min = 2*h - threshold/2
        if h_min < 0:
            h_min += 360
        h_min /= 2

        h_max = 2*h + threshold/2
        if h_max > 360:
            h_max -= 360
        h_max /= 2

        hue_filters.append((h_min , h_max, h))

    if not img_mode:
        cap = cv2.VideoCapture(0)

    while(ok):

        if(img_mode):
            frame = cv2.imread(args.img_name)
        else:
            # read the frames
            _,frame = cap.read()

        start_time = time.clock()

        centers = ct.detect_hues(frame, hue_filters)
        for center in centers:
            cx, cy = center
            cv2.circle(frame, (cx, cy), 7, (0, 0, 0), -1)

        # Show it, if key pressed is 'Esc', exit the loop
        cv2.imshow('frame',frame)

        end_time = time.clock()
        print "elapsed time", end_time - start_time

        if cv2.waitKey(33)== 27:
            break

        if img_mode and (cv2.waitKey() & 0xff) == ord('q'):
            ok = False

    cv2.destroyAllWindows()