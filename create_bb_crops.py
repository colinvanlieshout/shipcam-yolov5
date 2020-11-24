import pandas as pd
from PIL import Image
import os

desired_crop_size = 1024

DATA_PATH = "C:/Users/clieshou/Documents/Sogeti/The Ocean Cleanup/Plasticdebris_data"
IMAGES_PATH = os.path.join(DATA_PATH, "images/shipcam/")
OUTPUT_PATH = "shipcam_bbcrops/images"
label_file_name = "labels_split.csv"

df = pd.read_csv(os.path.join(DATA_PATH, label_file_name), delimiter=';')

def generate_crop_per_bb(image_path, bb_df):
    """
    This functions aim is to obtain crops of a specified size from images of any size. 
    It does this by first checking whether the bounding box of interest is larger or smaller than desired.
    If larger, it downsamples the entire image first.
    Then, we determine from what region the center of the crop can be sampled such that the bb will always fully fall within the image.

    input:
    - i: index of the object of interest for the current image
    - image: the path to the image
    - df_image: subset of the total df, only contains rows of this image
    - downsample_marging: how much smaller should the bounding box be than the image, 1 is the same, 2 is half
    
    returns:
    - crop: cropped image
    - crop_coordinates: xmin, ymin, xmax and ymax of the crop within the original image
    """

    image_to_crop = Image.open(image_path)

    crop_coordinates = (bb_df['xmin'], bb_df['ymin'], bb_df['xmax'], bb_df['ymax'])
    crop = image_to_crop.crop(crop_coordinates)

    return crop
    #initialize variables for the required data
    # bb_data = df_image.iloc[i, :]
    # bb_xmin, bb_ymin, bb_xmax, bb_ymax = bb_data['xmin'], bb_data['ymin'], bb_data['xmax'], bb_data['ymax']




if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    
    df['filename'] = df['filename'].str.lower()

    classes_of_interest = ['net', 'hard', 'line']
    df = df[df['type'] == 'Shipcam'].reset_index(drop=True)
    df = df[df['class'].isin(classes_of_interest)].reset_index(drop=True)

    imagename_list = df.filename.unique().tolist()

    for image_name in imagename_list:
        image_df = df[df['filename'] == image_name].reset_index(drop=True)
        for bb_id in range(len(image_df)):
            bb_df = image_df.loc[bb_id, :]
            image_path = os.path.join(IMAGES_PATH, bb_df['filename'])

            crop = generate_crop_per_bb(image_path, bb_df)

            image_name = bb_df['filename'].replace('.jpg', '_' + str(bb_id)+ '_' + bb_df['class']+'.jpg')
            crop.save(os.path.join(OUTPUT_PATH, image_name))