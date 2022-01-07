'''
example to detect upright people in images using HOG features

Usage:
    peopledetect.py <image_names>

Press any key to continue, ESC to stop.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import sys
from glob import glob
import itertools as it

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__':

    # Initialize HOG detector.
    
    hog = cv2.HOGDescriptor()
    detectorCoefficients = cv2.HOGDescriptor_getDefaultPeopleDetector()
    hog.setSVMDetector( detectorCoefficients )

    # Open webcam --- if fails to open, try entering 0 or 1 as argument
    cam = cv2.VideoCapture()

    # Infinite loop of capturing and detecting.
    while True:
        
        img, status = cam.grab()
    
        if status:
            # Detect humans
            
            found, w = hog.detectMultiScale(img, 
                                            winStride=(8,8), 
                                            padding=(32,32), 
                                            scale=1.05,
                                            hitThreshold = -1)
    
            # Draw what we found

            draw_detections(img, found)
            draw_detections(img, found_filtered, 3)

            cv2.imshow('img', img)
            cv2.waitKey()
    
    cv2.destroyAllWindows()

