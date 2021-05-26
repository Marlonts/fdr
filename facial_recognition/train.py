#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import matplotlib.pyplot as plt
# from matplotlib import patches
import numpy as np
import os
import pickle as pkl
from shutil import copyfile

import detectron2
from detectron2 import structures
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger



IMAGE_DATA_PATH = 'data/images/detect_images/train/image_data/'
MODEL_FINAL_PATH = 'data/models/model_final.pth'
PREPROCESSED_DATA_PATH = 'data/pkl/preprocessed_data.pkl'

DEVICE = "cuda"
# DEVICE = "cpu"
# MODEL_ZOO_NAME = 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'
MODEL_ZOO_NAME = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# ITERATIONS = 1500
ITERATIONS = 1501

setup_logger()


def draw_image_with_bbox(img, points):
    for box in points:
        x,y,x2,y2 = box

        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        #rect = patches.Rectangle((x,y),(x2,y2),linewidth=1,edgecolor='r',facecolor='none')
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 10)

    plt.imshow(img)
    plt.show()


def detectron2_dataset():
    with open(PREPROCESSED_DATA_PATH, "rb") as f:
        data = pkl.load(f)
        
    d2_data = []

    for idx, img in enumerate(data):
        annotations = []
        for point in img["points"]:
            
            width = img["size"][0]
            heigth = img["size"][1]
            
            # x1 = int(round(point[0]*width))
            # y1 = int(round(point[1]*heigth))
            # x2 = int(round(point[2]*width))
            # y2 = int(round(point[3]*heigth))

            x1 = int(round(point[0]))
            y1 = int(round(point[1]))
            x2 = int(round(point[2]))
            y2 = int(round(point[3]))

            annotations.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": structures.BoxMode.XYXY_ABS,
                "category_id": 0
            })
        
        img_d = {
            "file_name": img["img_path"],
            "width": width,
            "height": heigth, 
            "image_id": idx,
            "annotations": annotations
        }
        
        d2_data.append(img_d)
    
    return d2_data


def main():
    p_dataset = detectron2_dataset()
    i = np.random.randint(len(p_dataset))
    draw_image_with_bbox(cv2.imread(p_dataset[i]["file_name"]), [box["bbox"] for box in p_dataset[i]["annotations"]])

    plt.imshow(cv2.imread(p_dataset[i]["file_name"]))
    plt.show()

    DatasetCatalog.register("face_detection", detectron2_dataset)
    #d2_format_data = detectron2_dataset(d)

    cfg = get_cfg()
    cfg.merge_from_file(
    model_zoo.get_config_file(MODEL_ZOO_NAME))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_NAME)
    cfg.DATASETS.TRAIN = ("face_detection",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = ITERATIONS
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = DEVICE

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.WEIGHTS = MODEL_FINAL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    predictor = DefaultPredictor(cfg)

    images = os.listdir(IMAGE_DATA_PATH)
    img_name = np.random.choice(images)
    print(img_name)
    img_path = os.path.join(IMAGE_DATA_PATH, img_name)
    #im = cv2.imread(os.path.join("images", img_name))
    im = cv2.imread(img_path)
    outputs = predictor(im)
    print(outputs["instances"])
    draw_image_with_bbox(im, outputs["instances"].pred_boxes)

    copyfile(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"), MODEL_FINAL_PATH)


if __name__ == "__main__":
    main()
