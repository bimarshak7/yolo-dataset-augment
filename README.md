# YOLO Dataset Augmentation Tool

This project provides a versatile tool to artificially expand your YOLO dataset using image augmentation techniques. By applying various transformations to existing images and their corresponding bounding boxes, you can significantly increase the size and diversity of your dataset, leading to potentially improved performance in object detection models trained on it. It crates a new copy of augmented dataset, hence increasing the dataset size too.

## Yolo dataset format
```
dataset
    |- images
        |- 1.jpg
        |- 2.jpg
        |- ...
    |- labels
        |- 1.txt
        |- 2.txt
        |- ...
```
Labels file format
```
<class index> <x_center> <y_center> <height> <width>
```
Each line represent a bounding box  
Example:
```
1 0.457773 0.540167 0.412009 0.133819
1 0.788676 0.784694 0.162204 0.291972
```
## Project Setup
* Install requirements
```
pip install -r requirements.txt
```
* Get help
```
python main.py --help
```
* Usage
```
python main.py --data <path/to/your/yolo/dataset> [--out <output_directory>] [--prefix <prefix_for_output_files>] [--min_visibility <minimum_bbox_visibility>]
```

Arguments:

* --data (str, required): The path to your existing YOLO dataset directory containing images and labels.
* --out (str, default='aug_data'): The directory path where you want to save the augmented images and labels. Defaults to 'aug_data'.
* --prefix (str, default='aug'): An optional prefix to add to the filenames of the augmented outputs. Defaults to 'aug'.
* --min_visibility (float, default=0.4): The minimum visibility threshold (between 0.0 and 1.0) required for a bounding box to be preserved during image augmentation. Defaults to 0.4.

Example:
```
python main.py --data my_dataset --out augmented_data --prefix my_aug --min_visibility 0.3
```

This command will augment the images and labels in the my_dataset directory and save the augmented outputs (images and corresponding labels) in the augmented_data directory, with filenames prefixed with my_aug_. Bounding boxes with visibility below 0.3 will be discarded during augmentation.

Contribution:

We welcome contributions to this project! Feel free to submit pull requests for bug fixes, improvements, or additional augmentation techniques.
