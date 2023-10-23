import json
import numpy as np
import cv2
import os

def get_image_id_from_filename(filename, annotations):
    for img in annotations['images']:
        if img['file_name'] == filename:
            return img['id']
    return None

def load_mask(image_id, annotations):
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            segmentation = ann['segmentation'][0]
            img_info = next(filter(lambda x: x['id'] == image_id, annotations['images']))
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(segmentation).reshape(-1, 2).astype(np.int32)], 255)
            return mask
    return None

def extract_object_from_coco(image_path, annotations):
    image_filename = os.path.basename(image_path)
    image_id = get_image_id_from_filename(image_filename, annotations)
    if image_id is None:
        return np.zeros_like(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

    mask = load_mask(image_id, annotations)
    if mask is None:
        return np.zeros_like(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    
    return mask

def process_images_and_save(input_folder, annotation_file, npz_filename):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    images = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        result = extract_object_from_coco(image_path, annotations)
        
        images.append(original_image)
        labels.append(result)

    
    images = np.array(images)
    labels = np.array(labels)
    
    
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)

    test_split = int(0.1 * images.shape[0])
    
    test_indices = indices[:test_split]
    train_indices = indices[test_split:]
    
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    
    # Save to npz file
    np.savez(npz_filename, train_images=train_images, train_labels=train_labels, 
             test_images=test_images, test_labels=test_labels)

# 실행 부분
input_folder = '/Users/mac/Dev/Project/ThermalCamera/Model/Custom/Datasets/roboflow/train/'
annotation_file = '/Users/mac/Dev/Project/ThermalCamera/Model/Custom/Datasets/roboflow/train/_annotations.coco.json'
npz_filename = '/Users/mac/Dev/Project/ThermalCamera/Model/Custom/Datasets/dataset.npz'

process_images_and_save(input_folder, annotation_file, npz_filename)
