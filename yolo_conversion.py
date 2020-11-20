import pandas as pd
import os
from sklearn import model_selection
from tqdm import tqdm
import numpy as np
import shutil
from PIL import Image
import random
from math import floor, ceil
import cv2


def image_data_to_yolo(image_data ,data_type="train"):
    """
    
    input:
    - image_data: receives a row of a pandas dataframe containing an image_id, width, height, bboxes.
    The bboxes column contains a list of lists [[xmin, ymin, xmax, ymax], [..., ...]]
    This contains all bounding boxes associated with the image.

    returns:
    - list of lists, [[class, xcenter, ycenter, bb_width, bb_height]]
    """
    image_name = image_data["image_id"]
    image_width = image_data["width"]
    image_height = image_data["height"]
    bounding_boxes = image_data["bboxes"]
    yolo_data = []
    for bbox in bounding_boxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        x_center = x_center / image_width
        y_center = y_center / image_height
        bb_width = (xmax - xmin) / image_width
        bb_height = (ymax - ymin) / image_height

        yolo_data.append([0, x_center, y_center, bb_width, bb_height])
    yolo_data = np.array(yolo_data)

    return yolo_data, image_name

def store_yolo_data(yolo_data, DATA_PATH, OUTPUT_PATH, data_type, image_name):
    np.savetxt(
        os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
        yolo_data,
        fmt=["%d", "%f", "%f", "%f", "%f"]
    )
    shutil.copyfile(
        os.path.join(DATA_PATH, f"images/shipcam/{image_name}.jpg"),
        os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"),
    )

def iterator(data, DATA_PATH, OUTPUT_PATH, data_type):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        yolo_data, image_name = image_data_to_yolo(row)
        store_yolo_data(yolo_data,DATA_PATH, OUTPUT_PATH, data_type, image_name)

if __name__ == "__main__":
    DATA_PATH = "C:/Users/clieshou/Documents/Sogeti/The Ocean Cleanup/Plasticdebris_data"
    OUTPUT_PATH = "C:/Users/clieshou/PycharmProjects/yolov5/shipcamdata_yolo"
    
    #read and filter shipcam
    df = pd.read_csv(os.path.join(DATA_PATH, "labels_split.csv"), delimiter=';')
    df = df[df['type'] == 'Shipcam'].reset_index(drop=True)

    #combines the coordinates into a list
    coordinates = ['xmin', 'ymin', 'xmax', 'ymax',]
    df['bbox'] = df[coordinates].values.tolist()
    df.drop(coordinates, axis=1,inplace=True)

    #get a list of lists for each image ID
    df = df.rename(columns={'filename':'image_id'})
    df['image_id'] = df['image_id'].str.lower()
    df['image_id'] = df['image_id'].str.split(".").str.get(0)
    df = df.groupby(["image_id", "width", "height"])["bbox"].apply(list).reset_index(name="bboxes")

    #create train test set
    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    iterator(df_train, DATA_PATH, OUTPUT_PATH, data_type="train")
    iterator(df_valid, DATA_PATH, OUTPUT_PATH, data_type="validation")
