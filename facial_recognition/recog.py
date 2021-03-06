#!/usr/bin/python3
# -*- coding: utf-8 -*-


import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from utils import get_features, transform_image


PEOPLE_3DIM_PATH = "data/pkl/people_3dim.pkl"
MIN_DIFFERENCE = 0.6


def compare(features, target_features):
    return np.linalg.norm(features - target_features)


def show_image_and_identification(image_path, recognition_results):
    img = cv2.imread(image_path)
    img = transform_image(img, resize=False)

    for person_label, points in recognition_results:
        x,y,x2,y2 = points
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        img = cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 10)

        plt.imshow(img)
        plt.annotate(person_label, fontsize=15, xy=(.25, .75), xycoords='data', xytext=(x, y))
    
    plt.show()


@click.command()
@click.argument('image_path')
def main(image_path):
    image_features = get_features(image_path)

    with open(PEOPLE_3DIM_PATH, "rb") as f:
        people = pkl.load(f)

    print("Possible people:", list(people.keys()))

    recognition_results = []
    for info in image_features:
        features, points = info

        min_difference = float("inf")
        min_person = ""

        for ppl in people:
            people_features = people[ppl]
            for img_f in people_features:
                diff = compare(features, img_f)

                if diff < min_difference:
                    min_difference = diff
                    min_person = ppl

        if min_difference > MIN_DIFFERENCE:
            print("No faces recognized")
            #min_person = "Not recognized"
            continue

        print(f"Found: {min_person} {min_difference}")
        recognition_results.append([min_person, points])

    show_image_and_identification(image_path, recognition_results)


if __name__ == "__main__":
    main()
