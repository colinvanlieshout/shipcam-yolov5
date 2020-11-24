from PIL import Image
import random
import pandas as pd
from math import floor, ceil
import cv2
import glob
import os
from PIL import Image, ImageDraw

"""
#NOTE:
run the following in the folder with the images to make their titles all lowercase
- for /f "Tokens=*" %f in ('dir /l/b/a-d') do (rename "%f" "%f")

"""
desired_crop_size = 640
downsample_margin = 1.3

DATA_PATH = "C:/Users/clieshou/Documents/Sogeti/The Ocean Cleanup/Plasticdebris_data/"
image_paths = glob.glob(DATA_PATH + 'images/shipcam/*')
OUTPUT_PATH = "shipcamdata_cropped/images"
label_file_name = "labels_split.csv"

draw_bounding_box=False
# OUTPUT_PATH = "shipcamdata_cropped/images_annotated"
# draw_bounding_box = True

def image_resizer(image_path, downsample_factor, bb_data):
    """
    Resize images using opencv
    
    input:
    - image_path
    - downsample_factor: combination of how large the image is, and the margin you want it to have in terms of size in relationship to the size of the image
    - bb_data: needed for the width and the height of the bb

    """

    image_array = cv2.imread(image_path)
    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_to_crop = cv2.resize(image_array_rgb, (int(bb_data['width']/downsample_factor), int(bb_data['height']/downsample_factor)), interpolation= cv2.INTER_AREA)
    image_to_crop = Image.fromarray(image_to_crop)

    return image_to_crop

def get_width_height(bb_xmin, bb_ymin, bb_xmax, bb_ymax):
    """
    gets width and height of a bounding box
    """
    bb_width = bb_xmax - bb_xmin
    bb_height = bb_ymax - bb_ymin

    return bb_width, bb_height

def define_center_sampling_region(bb_xmin, bb_ymin, bb_xmax, bb_ymax, desired_crop_size, image_width, image_height):
    """
    Steps:
    Determines from what region the center of the crop can be randomly selected.
    It does this by compensating desired crop size for the size of the bounding box, to avoid it not being fully included.
    It compensates for the fact that bounding boxes may be close to the edge of the image, didn't get to it yet
    """

    bb_width, bb_height = get_width_height(bb_xmin, bb_ymin, bb_xmax, bb_ymax)

    bb_xcenter = bb_xmin + bb_width/2
    bb_ycenter = bb_ymin + bb_height/2

    #we want to select the center of the crop randomly, but have to make sure that the entire bb falls within the crop
    center_sampling_region_xmin = bb_xcenter - desired_crop_size/2 + bb_width/2
    center_sampling_region_ymin = bb_ycenter - desired_crop_size/2 + bb_height/2
    center_sampling_region_xmax = bb_xcenter + desired_crop_size/2 - bb_width/2
    center_sampling_region_ymax = bb_ycenter + desired_crop_size/2 - bb_height/2

    #randomly getting the center within the allowed frame, using seed to get the same each time for the same bounding box. Not sure if there are any reasons not to do this
    random.seed(bb_xmin)
    crop_xcenter = random.randint(floor(center_sampling_region_xmin), ceil(center_sampling_region_xmax))
    random.seed(bb_ymin)
    crop_ycenter = random.randint(floor(center_sampling_region_ymin), ceil(center_sampling_region_ymax))

    #make sure the center fall within the acceptable frame, where the entire crop will fall within the image
    if crop_xcenter < desired_crop_size/2:
        crop_xcenter = desired_crop_size/2
    if crop_xcenter > (image_width - desired_crop_size/2):
        crop_xcenter = (image_width - desired_crop_size/2)
    if crop_ycenter < desired_crop_size/2:
        crop_ycenter = desired_crop_size/2
    if crop_ycenter > (image_height - desired_crop_size/2):
        crop_ycenter = (image_height - desired_crop_size/2)

    return crop_xcenter, crop_ycenter

def downsample_large_bb(bb_data, desired_crop_size, downsample_margin, image_path):
    """
    checks if an object is too large for the desired crop size. If it is, it samples it down
    """
    
    bb_xmin, bb_ymin, bb_xmax, bb_ymax = bb_data['xmin'], bb_data['ymin'], bb_data['xmax'], bb_data['ymax']
    
    bb_width, bb_height = get_width_height(bb_xmin, bb_ymin, bb_xmax, bb_ymax)

    #if one of the dimensions is larger then the desired with some defined margin, we downsample the image with that margin
    downsample_factor = max(bb_width, bb_height)/desired_crop_size*downsample_margin
    if (bb_width > desired_crop_size/downsample_factor) or (bb_height > desired_crop_size/downsample_factor):
                
        #read the image as array, convert to rgb, resize and turn into an acutal image bb
        image_to_crop = image_resizer(image_path, downsample_factor, bb_data)

        #scale the coordinates based on the downsample_factor
        bb_xmin, bb_ymin, bb_xmax, bb_ymax = [x/downsample_factor for x in [bb_xmin, bb_ymin, bb_xmax, bb_ymax]]
        downsampled = True
    else:
        image_to_crop = Image.open(image_path)
        downsampled = False

    return bb_xmin, bb_ymin, bb_xmax, bb_ymax, downsampled, downsample_factor, image_to_crop

def generate_crop_per_bb(i, image_path, df_image, downsample_margin, desired_crop_size):
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

    #initialize variables for the required data
    bb_data = df_image.iloc[i, :]
    # bb_xmin, bb_ymin, bb_xmax, bb_ymax = bb_data['xmin'], bb_data['ymin'], bb_data['xmax'], bb_data['ymax']
    
    #check if a bounding box is too large for the desired crop size, and if so, sample it down
    bb_xmin, bb_ymin, bb_xmax, bb_ymax, downsampled, downsample_factor, image_to_crop = downsample_large_bb(bb_data, desired_crop_size, downsample_margin, image_path)
    
    #determine what the center of the crop should be
    image_width, image_height = image_to_crop.size
    center_sampling_region_xcenter, center_sampling_region_ycenter = define_center_sampling_region(bb_xmin, bb_ymin, bb_xmax, bb_ymax, desired_crop_size, image_width, image_height)

    #determine crop coordinates, then crop
    crop_xmin = center_sampling_region_xcenter - desired_crop_size/2
    crop_ymin = center_sampling_region_ycenter - desired_crop_size/2
    crop_xmax = center_sampling_region_xcenter + desired_crop_size/2
    crop_ymax = center_sampling_region_ycenter + desired_crop_size/2

    #store the bounding box coordinates relative to the sample/downsample
    bb_coordinates = [bb_xmin, bb_ymin, bb_xmax, bb_ymax]

    crop_coordinates = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    crop = image_to_crop.crop(crop_coordinates)

    if downsampled == True:
        crop_coordinates = tuple([x*downsample_factor for x in list(crop_coordinates)])

    return crop, crop_coordinates, downsample_factor, downsampled, bb_coordinates

def draw_bounding_boxes(image, data):
    draw = ImageDraw.Draw(image)
        # df_crop_draw = df_crop.iloc[i, :]
    draw = ImageDraw.Draw(image)
    for i in range(len(data)):
        data_draw = data.iloc[i, :]
        draw.rectangle(((data_draw['xmin'], data_draw['ymin']), (data_draw['xmax'], data_draw['ymax'])), outline = 'red', width = 3)

    return image

if __name__ == "__main__":
    #load label file and create empty dataframe with the same columns
    df = pd.read_csv(os.path.join(DATA_PATH, label_file_name), delimiter=';')
    df['filename'] = df['filename'].str.lower()
    df_crops = pd.DataFrame(columns = list(df))

    #check if output folder exists, if not, create it
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for image_path in image_paths:
        #replace the double back slash to make sure it also works on not-windows machines
        image_name = image_path.replace('\\', '/').split('/')[-1]
        df_image = df[df["filename"] == image_name.lower()].reset_index(drop=True)

        for i in range(len(df_image)):
            #crop the images and alter the coordinates accordingly
            crop, crop_coordinates, downsample_factor, downsampled, bb_coordinates = generate_crop_per_bb(i, image_path, df_image, downsample_margin, desired_crop_size)
            
            #check if there are other bounding boxes in the crop
            df_crop = df_image[(df_image['xmin'] < crop_coordinates[2]) & (df_image['ymin'] < crop_coordinates[3]) & (df_image['xmax'] > crop_coordinates[0]) & (df_image['ymax'] > crop_coordinates[1])].reset_index(drop=True)
            df_crop['filename'] = df_crop['filename'].str.replace('.jpg', '_c' + str(i) + '.jpg')
            
            #correct the data for the cropping, and if applied, the downsampling
            df_crop['width'] = desired_crop_size
            df_crop['height'] = desired_crop_size

            if downsampled:     
                df_crop['xmin'] = bb_coordinates[0] - crop_coordinates[0] / downsample_factor
                df_crop['ymin'] = bb_coordinates[1] - crop_coordinates[1] / downsample_factor
                df_crop['xmax'] = bb_coordinates[2] - crop_coordinates[0] / downsample_factor
                df_crop['ymax'] = bb_coordinates[3] - crop_coordinates[1] / downsample_factor
            else:
                df_crop['xmin'] -= crop_coordinates[0]
                df_crop['ymin'] -= crop_coordinates[1]
                df_crop['xmax'] -= crop_coordinates[0]
                df_crop['ymax'] -= crop_coordinates[1]

            df_crops = df_crops.append(df_crop)

            #store the images
            image_path_crop = os.path.join(OUTPUT_PATH, df_crop.loc[0, 'filename'])
            if draw_bounding_box == True:
                crop = draw_bounding_boxes(crop, df_crop)
            crop.save(image_path_crop)

    #store the new csv based on the crops
    df_crops.to_csv("shipcamdata_cropped/labels_cropped_images.csv", index = False)