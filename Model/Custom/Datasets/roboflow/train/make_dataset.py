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

for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    
    # 이미지 읽기 (흑백 이미지로 로드)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    filtered_img = np.zeros_like(img)  # 사람이 없는 부분은 검은색으로 초기화

    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        person_count = 0

        for line in lines:
            items = line.strip().split(' ')
            label = items[-2]
            if label == 'person':
                person_count += 1
                # 바운딩 박스 좌표 가져오기
                pts = np.array([[int(items[j]), int(items[j+1])] for j in range(0, 8, 2)])
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                roi = img[y:y+h, x:x+w]
                filtered_img[y:y+h, x:x+w] = roi  # 사람 있는 부분은 원본 이미지 데이터로 업데이트

        # 사람의 수에 따라 numbers 배열에 값 추가
        if person_count == 0:
            numbers.append(0)
        elif person_count == 1:
            numbers.append(1)
        else:
            numbers.append(2)

        images.append(img)
        filters.append(filtered_img)

# npz 파일로 저장
np.savez('dataset.npz', images=np.array(images), filters=np.array(filters), numbers=np.array(numbers))
