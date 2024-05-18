'''
Module to define albumentations Compose of different image-bbox transformation functions.
See albumentation documentation for list of all available transforms
https://albumentations.ai/docs/api_reference/full_reference/
'''

import albumentations as A


def get_transformer(data_format: str = "yolo",
                    min_visibility: float = 0.4) -> A.Compose:
    """
    This function creates a composed image transformation pipeline
    using albumentations library.
    Args:
        format (str, optional): The format of bounding boxes expected
        by the pipeline. Defaults to "yolo".
        min_visibility (float, optional): Minimum visibility threshold for
        bounding boxes to be considered during transformation. Defaults to 0.4.

    Returns:
        A.Compose: An Albumentations `Compose` object containing
        the image-bbox transformation pipeline.
    """

    transform = A.Compose([
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomCrop(height=640, width=640, p=0.2),
        A.RandomShadow(p=0.3)
    ], bbox_params=A.BboxParams(format=data_format,
                                label_fields=['class_labels'],
                                min_visibility=min_visibility))

    return transform
