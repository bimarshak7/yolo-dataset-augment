'''
Implement pipeline for yolo dataset augmentation.
The pipeline is composed od following tasks:
    1. Read image and corresponding label text from dataset directory
    2. Pass both image and bounding boxes through transformation pipeline
    3. Save the transformed image-bbox in the output directory
'''

import argparse
import os

from tqdm import tqdm
import cv2

from utils import read_yolo_annotation
from utils import convert_annotation
from utils import augment_and_save
from transform import get_transformer

# define commandline arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='data',
                    help='Path of the yolo dataset directory')
parser.add_argument('--out', type=str, default='aug_data',
                    help='Number of training epochs.')
parser.add_argument('--prefix', type=str, default='aug',
                    help='Prefix to add to augmented output files')

# parse commandline arguments
args = parser.parse_args()

input_dir = args.data
output_dir = args.out
out_prefix = args.prefix


def main():
    '''
    Main pipeline function
    '''
    try:
        if os.path.exists(input_dir):
            # define image and label path for input dataset
            train_images_dir = os.path.join(input_dir, 'images')
            train_labels_dir = os.path.join(input_dir, 'labels')
        else:
            raise FileNotFoundError('Invalid dataset path.')

        # define path to save augmented images and labels
        output_images_dir = os.path.join(output_dir, 'images')
        output_labels_dir = os.path.join(output_dir, 'labels')

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # get transformation pipeline
        transform = get_transformer(data_format="yolo",
                                    min_visibility=0.4)

        # create progress bar with list of input images
        pbar = tqdm(os.listdir(train_images_dir))

        for filename in pbar:
            try:
                pbar.set_description(f'Working on {filename}')
                if filename.endswith('.jpg'):
                    image_path = os.path.join(train_images_dir, filename)
                    annotation_path = os.path.join(
                        train_labels_dir, filename.replace('.jpg', '.txt'))
                    image = cv2.imread(image_path)
                    annotations = read_yolo_annotation(annotation_path)
                    image_height, image_width = image.shape[:2]
                    annotations = convert_annotation(
                        annotations, image_width, image_height)
                    output_image_path = os.path.join(
                        output_images_dir, 'aug_'+filename)
                    output_annotation_path = os.path.join(
                        output_labels_dir,
                        'aug_' + filename.replace('.jpg', '.txt'))

                    augment_and_save(
                        image, annotations, output_image_path,
                        output_annotation_path, transform)
            except Exception:
                print(f'Failed to augment image: {
                      filename} due to bounding box conversion error')

    except FileNotFoundError as e:
        print(e)
    except Exception as exp:
        print(exp)


if __name__ == '__main__':
    main()
