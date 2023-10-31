import json
import numpy as np
import cv2
import os
import re

def get_image_id_from_filename(filename, annotations):
    for img in annotations['images']:
        if img['file_name'] == filename:
            return img['id']
    return None

def load_masks_and_number(image_id, annotations):
    masks = []
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            segmentation = ann['segmentation'][0]
            img_info = next(filter(lambda x: x['id'] == image_id, annotations['images']))
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(np.int32)], 255)
            masks.append(mask)
    if len(masks) > 0:
        combined_mask = np.max(np.stack(masks, axis=0), axis=0)
        return combined_mask, len(masks) - 1  # subtract one to get number of people
    else:
        return None, None

def process_images_and_save(input_folder, annotation_file, prefix):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    def sort_files(file):
        number = int(re.search(r'_(\d+)_png', file).group(1))
        return number

    if prefix == 'two':
        image_files = sorted([f for f in os.listdir(input_folder) if f.startswith('two-mp4')], key=sort_files)
    else:
        image_files = sorted([f for f in os.listdir(input_folder) if f.startswith('2023-10-23-07-17-29-mp4')], key=sort_files)

    images = []
    filters = []
    numbers = []

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask, number = load_masks_and_number(get_image_id_from_filename(image_file, annotations), annotations)

        if mask is None or number is None:
            continue

        filtered_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        images.append(original_image)
        filters.append(filtered_image)
        numbers.append(number)

    np.savez(f'{prefix}_dataset.npz', 
             images=np.array(images), 
             filters=np.array(filters), 
             numbers=np.array(numbers))

# 실행 부분
input_folder = '/Users/mac/Dev/Project/ThermalCamera/Model/Custom/Datasets/roboflow/train/'
annotation_file = '/Users/mac/Dev/Project/ThermalCamera/Model/Custom/Datasets/roboflow/train/_annotations.coco.json'

process_images_and_save(input_folder, annotation_file, 'two')
process_images_and_save(input_folder, annotation_file, 'one')
