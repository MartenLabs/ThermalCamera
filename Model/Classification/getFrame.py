import os
import cv2
import datetime

get_location = '/Users/mac/Dev/Project/ThermalCamera/Model/Classification/dataset/tmp/'

img_count:int = 0
# 해당 경로에 있는 모든 mp4 파일 경로를 리스트로 저장
file_list = [file for file in os.listdir(get_location) if not file.startswith('.')]

# for문을 통해 모든 mp4 파일을 돌면서 이미지 추출
for file in file_list:
    # 비디오 이름으로 폴더 생성 후 경로 저장 
    try:
        os.mkdir(f"/Users/mac/Dev/Project/ThermalCamera/Model/Classification/all_img/transfer/{file}")
    except FileExistsError:
        pass 
    vidcap = cv2.VideoCapture(get_location + file)
    count = 0
    while True:
        ret, image = vidcap.read()
        # 더 이상 프레임을 읽을 수 없을 때 루프 종료
        if not ret:
            break
        # 5프레임당 하나씩 이미지 추출
        if int(vidcap.get(1)) % 1 == 0:
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            # 추출된 이미지가 저장되는 경로
            cv2.imwrite(f"/Users/mac/Dev/Project/ThermalCamera/Model/Classification/all_img/transfer/{file}/{file}_{count}.png", image)
        count += 1
        img_count += 1

    vidcap.release()
    
print(f'{file} is done! 총: {img_count}장 추출')