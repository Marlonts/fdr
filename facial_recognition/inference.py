#!/usr/bin/python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import cv2
# import os
import click

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import structures
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo


MODEL_FINAL_PATH = 'data/models/model_final.pth'
# MODEL_ZOO_NAME = 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'
MODEL_ZOO_NAME = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
THREASH = 0.7
DEVICE = "cuda"
# DEVICE = "cpu"

setup_logger()


def load_predictor(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_NAME))
    # cfg.MODEL.WEIGHTS = os.path.join("models", model_name)
    cfg.MODEL.WEIGHTS = model_name

    # cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.DEVICE = DEVICE

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THREASH
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    return DefaultPredictor(cfg)


# predictor = load_predictor("model_final_1.pth")
predictor = load_predictor(MODEL_FINAL_PATH)

def draw_image_with_bbox(img, points):
    for box in points:
        x,y,x2,y2 = box
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 10)

    plt.imshow(img)
    plt.show()


def inference(image):
    outputs = predictor(image)
    points = outputs["instances"].pred_boxes
    return points


@click.command()
@click.argument('image_path')
def cli(image_path):
    img = cv2.imread(image_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points = inference(img)
    print(f"Found {len(points)} faces...")

    draw_image_with_bbox(img, points)


if __name__ == "__main__":
    cli()
