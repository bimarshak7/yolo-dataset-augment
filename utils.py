'''
Collection of helper functions to implement yolo
dataset augmentation.
'''

import cv2
import numpy as np
import albumentations as A


def read_yolo_annotation(file_path: str) -> list:
    """
    Reads YOLO format annotations from a text file.

    This function reads a text file containing annotations in YOLO format
    and returns a list of annotations.
    Each annotation is expected to be a single line string in YOLO format.

    Args:
        file_path (str): The path to the text file containing YOLO annotations.

    Returns:
        list: A list of YOLO annotations, where each element is a string
        representing a single annotation.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        annotations = [line.strip() for line in file.readlines()]
    return annotations


def write_yolo_annotation(file_path: str, annotations: list) -> None:
    """
    Writes YOLO format annotations to a text file.

    This function writes a list of annotations in YOLO format to a
    specified text file.
    Each annotation is expected to be provided as a string in the YOLO format.

    Args:
        file_path (str): The path to the text file where annotations
        will be saved.
        annotations (list): A list of YOLO annotations, where each
        element is a string representing a single annotation.

    Returns:
        None
    """

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(annotations)


def augment_and_save(image: np.ndarray, annotations: list,
                     output_image_path: str,
                     output_annotation_path: str,
                     transform: A.Compose) -> None:
    """
    Applies image augmentation and saves the results.

    This function takes an image, its corresponding annotations in YOLO format,
    a transformation pipeline, and output paths for the augmented image and
    annotations.

    Args:
        image (np.ndarray): The original image to be augmented.
        annotations (list): A list of YOLO format annotations for
        the original image.
        output_image_path (str): The path to save the augmented image.
        output_annotation_path (str): The path to save the augmented
        annotations in YOLO format.
        transform (A.Compose): An Albumentations `Compose` object
        containing the image transformation pipeline.

    Returns:
        None
    """

    # Perform augmentation
    transformed = transform(image=image, bboxes=annotations, class_labels=[
                            ann[-1] for ann in annotations])
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # Convert annotations back to YOLO format and save
    transformed_annotations = []
    for bbox, class_label in zip(transformed_bboxes, transformed['class_labels']):
        x_center, y_center, width, height, _ = bbox
        transformed_annotations.append(f"{class_label} {x_center} {
                                       y_center} {width} {height}\n")

    # Save augmented image
    cv2.imwrite(output_image_path, transformed_image)

    # Save augmented annotations
    write_yolo_annotation(output_annotation_path, transformed_annotations)


def convert_annotation(annotation, image_width, image_height):
    converted = []
    for ann in annotation:
        class_id, x_center, y_center, width, height = map(
            float, ann.split(' '))
        converted.append((x_center, y_center, width, height, int(class_id)))
    return converted
