#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from argparse import ArgumentParser
from imutils import resize
from imutils.object_detection import non_max_suppression


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect(image_path):
    print('Image path:', image_path)
    image = cv2.imread(image_path)

    # resizing for faster detection
    image = cv2.resize(image, (640, 480))
    #image = resize(image, width=min(400, image.shape[1]))

    im = image.copy()
    orig = image.copy()

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        # display the detected boxes in the colour picture
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    filename = image_path[image_path.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
         filename, len(rects), len(pick)))
    
    # show the output images
    cv2.imshow("Original Image", im)
    cv2.waitKey(0)
    cv2.imshow("Original Bounding Boxes", orig)
    cv2.waitKey(0)
    cv2.imshow("Final Bounding Boxes", image)
    cv2.waitKey(0)


def parse_args():
    parser = ArgumentParser(description='Allowed options')
    parser.add_argument('-v', '--version', action='version',
                        version='version 1.2',
                        help='display application version')

    parser.add_argument('-i', '--image', required=True, help='Input image file')

    return parser.parse_args()


def main():
    args = parse_args()
    image_path =  args.image
    detect(image_path)


if __name__ == '__main__':
    main()
