#!/usr/bin/python3
# -*- coding: utf-8 -*-


import csv
import os
import pickle as pkl


BBOX_TRAIN_PATH = 'data/images/detect_images/train/bbox_train.csv'
IMAGE_DATA_PATH = 'data/images/detect_images/train/image_data/'
PREPROCESSED_DATA_PATH = 'data/pkl/preprocessed_data.pkl'

def preprocess_data(bbox_train, image_data):
    with open(bbox_train, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)  # skip the headers
        data = []

        for row in reader:
            img_path = os.path.join(image_data, row[0])
            width = int(row[1])
            height = int(row[2])
            points = []
            x1 = float(row[3])
            y1 = float(row[4])
            x2 = float(row[5])
            y2 = float(row[6])
            points.append([x1, y1, x2, y2])
            data.append({"img_path": img_path, "size": (width, height), "points": points})

    return data


def main():
    p_data = preprocess_data(BBOX_TRAIN_PATH, IMAGE_DATA_PATH)
    print('eg:', p_data[0])

    with open(PREPROCESSED_DATA_PATH, "wb") as f:
        pkl.dump(p_data, f)


if __name__ == "__main__":
    main()
