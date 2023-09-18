import os
import shutil

# 이미지 파일이 있는 폴더 경로 설정
image_folder = "DataSet"

# 폴더 내부의 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 이미지 파일 이름을 0부터 시작해서 +1 씩 증가시키며 변경
counter = 0
for image_file in image_files:
    old_path = os.path.join(image_folder, image_file)
    new_name = f"{counter}.jpg"  # 변경될 파일 이름 예시
    new_path = os.path.join(image_folder, new_name)
    
    # 이미지 파일 이름 변경
    os.rename(old_path, new_path)
    
    counter += 1

print("이미지 파일 이름 변경 완료.")
