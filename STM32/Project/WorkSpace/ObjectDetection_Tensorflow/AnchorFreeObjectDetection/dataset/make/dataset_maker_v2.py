import os
import cv2
import numpy as np

# 경로 설정
image_dir = 'labels/ABC.v1i.yolov5-obb/train/images/'
label_dir = 'labels/ABC.v1i.yolov5-obb/train/labelTxt'

def extract_number(filename):
    # 파일 이름에서 'count_png' 형식의 부분만 추출하여 정수로 변환
    return int(filename.split('_')[0])

# 파일을 'count_png' 부분을 기준으로 정렬
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')], key=extract_number)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')], key=extract_number)

# 결과 저장을 위한 리스트 생성
datasets = {
    'origin': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'horizon': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'vertical': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []},
    'test': {'images': [], 'filters': [], 'numbers': [], 'coordinates': []}
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

for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    
    # 원본 이미지를 읽기
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = img.shape
    
    # 이미지의 좌우, 상하, 상하좌우 반전을 수행
    img_flip_hor = cv2.flip(img, 1)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_both = cv2.flip(img, -1)
    
    for dataset_name, transformations in zip(
            ['origin', 'horizon', 'vertical', 'horvert'], 
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
                    coords = [int(coord) for coord in items[:8]]
                    
                    # 좌우, 상하, 상하좌우 반전에 따른 좌표 변환
                    if flip_mode:
                        coords = flip_coordinates(coords, img_width, img_height, flip_mode)
                    
                    boxes.append(coords)

                    # 바운딩 박스를 이용해서 filtered_img 업데이트
                    pts = np.array([[coords[j], coords[j+1]] for j in range(0, 8, 2)])
                    rect = cv2.boundingRect(pts)
                    x, y, w, h = rect
                    roi = processed_img[y:y+h, x:x+w]
                    filtered_img[y:y+h, x:x+w] = roi

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
