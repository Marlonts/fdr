#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import pickle as pkl

from utils import get_features


PEOPLE_3DIM_PATH = "data/pkl/people_3dim.pkl"
RECOG_IMAGES_PATH = 'data/images/recog_images'


def register_recog_images():
    people = {}
    folders = os.listdir(RECOG_IMAGES_PATH)

    for name in folders:
        print(name)
        folder_path = os.path.join(RECOG_IMAGES_PATH, name)
        if not os.path.isdir(folder_path):
            continue

        image_representations = []
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            print(image_path)

            features = get_features(image_path)
            image_representations.append(features[0][0])

        people[name] = image_representations
    return people


def main():
    people = register_recog_images()
    with open(PEOPLE_3DIM_PATH, "wb") as f:
        pkl.dump(people, f)


if __name__ == "__main__":
    main()
