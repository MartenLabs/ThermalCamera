import os
import cv2
import numpy as np

# 경로 설정
image_dir = 'labels/Classifiy Number of Human.v1i.yolov5-obb/train/images/'
label_dir = 'labels/Classifiy Number of Human.v1i.yolov5-obb/train/labelTxt/'

# 파일 리스트 생성
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# 파일 이름을 기준으로 이미지와 레이블 매핑
file_map = {}
for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    lbl_file = base_name + '.txt'
    if lbl_file in label_files:
        file_map[img_file] = lbl_file

# 결과 저장을 위한 리스트 생성
datasets = {
    'origin': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'horizon': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'vertical': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'hv': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []}
}

MAX_BOXES = 4
EMPTY_BOX = [0, 0, 0, 0, 0, 0, 0, 0]

def flip_coordinates(coords, img_width, img_height, mode):
    """
    Flip coordinates according to the mode.
    'horizontal' for left-right, 'vertical' for up-down, 'both' for both directions.
    """
    flipped_coords = coords.copy()
    for i in range(0, 8, 2):
        if mode == 'horizontal' or mode == 'both':
            flipped_coords[i] = img_width - coords[i]
        if mode == 'vertical' or mode == 'both':
            flipped_coords[i + 1] = img_height - coords[i + 1]
    return flipped_coords

for img_file, lbl_file in file_map.items():
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    
    # 원본 이미지를 읽기
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = img.shape
    
    if img_height != 24 or img_width != 32:
        continue

    # 이미지의 좌우, 상하, 상하좌우 반전을 수행
    img_flip_hor = cv2.flip(img, 1)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_both = cv2.flip(img, -1)
    
    for dataset_name, transformations in zip(
            ['origin', 'horizon', 'vertical', 'hv'], 
            [(img, None), (img_flip_hor, 'horizontal'), 
             (img_flip_ver, 'vertical'), (img_flip_both, 'both')]):
        
        processed_img, flip_mode = transformations
        filtered_img = np.zeros_like(processed_img)
        boxes = []

        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            person_count = 0

            for line in lines:
                items = line.strip().split(' ')
                label = items[-2]
                if label == 'person':
                    person_count += 1
                    coords = [int(round(float(coord))) for coord in items[:8]]
                    
                    # 좌우, 상하, 상하좌우 반전에 따른 좌표 변환
                    if flip_mode:
                        coords = flip_coordinates(coords, img_width, img_height, flip_mode)
                    
                    boxes.append(coords) 

                # 바운딩 박스를 이용해서 filtered_img 업데이트
                pts = np.array([[coords[j], coords[j+1]] for j in range(0, 8, 2)], dtype=np.int32)
                cv2.polylines(filtered_img, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

        # 최대 바운딩 박스 개수까지 고려해 남은 바운딩 박스를 EMPTY_BOX로 채우기
        while len(boxes) < MAX_BOXES:
            boxes.append(EMPTY_BOX)

        datasets[dataset_name]['images'].append(processed_img)
        datasets[dataset_name]['filters'].append(filtered_img)
        datasets[dataset_name]['numbers'].append(person_count)
        datasets[dataset_name]['coordinates'].append(boxes)

# 각각의 데이터셋을 npz 파일로 저장
for name, data in datasets.items():
    np.savez(f'{name}.npz', 
             images=np.array(data['images']), 
             filters=np.array(data['filters']), 
             numbers=np.array(data['numbers']), 
             coordinates=np.array(data['coordinates']))