import pandas as pd
import os
from sklearn import model_selection
from tqdm import tqdm
import numpy as np
import shutil

DATA_PATH = r"C:\Users\clieshou\Documents\Sogeti\The Ocean Cleanup\Plasticdebris_data"
OUTPUT_PATH = r"C:\Users\clieshou\PycharmProjects\yolov5\shipcamdata"

# https://www.youtube.com/watch?v=NU9Xr_NYslo

#check if values are in the list in the right order
def process_data(data, data_type="train"):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row["image_id"]
        image_width = row["width"]
        image_height = row["height"]
        bounding_boxes = row["bboxes"]
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
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels\\{data_type}\\{image_name}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH, f"images\\shipcam\\{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images\\{data_type}\\{image_name}.jpg"),
        )


if __name__ == "__main__":
    #read and filter shipcam
    df = pd.read_csv(os.path.join(DATA_PATH, "labels_split.csv"), delimiter=';')
    df = df[df['type'] == 'Shipcam'].reset_index(drop=True)
    # df = pd.read

    #combines the coordinates into a list
    coordinates = ['xmin', 'ymin', 'xmax', 'ymax',]
    df['bbox'] = df[coordinates].values.tolist()
    df.drop(coordinates, axis=1,inplace=True)

    #get a list of lists for each image ID
    df = df.rename(columns={'filename':'image_id'})
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

    process_data(df_train, data_type="train")
    process_data(df_valid, data_type="validation")

    # print(df)



