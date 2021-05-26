#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import dlib
import numpy as np
import sys


FACE_RECOGNITION_MODEL_PATH = "data/models/dlib_face_recognition_resnet_model_v1.dat"


# Code work of: http://dlib.net/face_recognition.py.html (davisking on GitHub)
def get_detector():
    facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    return facerec


def main():
    f_path = sys.argv[1]
    img = cv2.imread(f_path)
    img = cv2.resize(img, (150, 150))
    d = get_detector()
    f = d.compute_face_descriptor(img)
    print(np.array(f).shape)


if __name__ == "__main__":
    main()
