import os
import cv2
import numpy as np

# 경로 설정
image_dir = './images'
label_dir = './labelTxt'

def extract_number(filename):
    # 파일 이름에서 'count_png' 형식의 부분만 추출하여 정수로 변환
    return int(filename.split('_')[0])

# 파일을 'count_png' 부분을 기준으로 정렬
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')], key=extract_number)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')], key=extract_number)

# 결과 저장을 위한 리스트
images = []
filters = []
numbers = []
coordinates = []

MAX_BOXES = 4
EMPTY_BOX = [0, 0, 0, 0, 0, 0, 0, 0]

for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    filtered_img = np.zeros_like(img)

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
                boxes.append(coords)

                # 바운딩 박스를 이용해서 filtered_img 업데이트
                pts = np.array([[coords[j], coords[j+1]] for j in range(0, 8, 2)])
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                roi = img[y:y+h, x:x+w]
                filtered_img[y:y+h, x:x+w] = roi

        # 최대 바운딩 박스 개수까지 고려해 남은 바운딩 박스를 EMPTY_BOX로 채우기
        while len(boxes) < MAX_BOXES:
            boxes.append(EMPTY_BOX)

        # 사람의 수에 따라 numbers 배열에 값 추가
        numbers.append(person_count)

        images.append(img)
        filters.append(filtered_img)
        coordinates.append(boxes)

np.savez('detection_dataset.npz', 
         images=np.array(images), 
         filters=np.array(filters), 
         numbers=np.array(numbers), 
         coordinates=np.array(coordinates))
