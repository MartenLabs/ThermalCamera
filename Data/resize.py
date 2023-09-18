from PIL import Image
import os

# 이미지 파일이 있는 폴더 경로 설정
image_folder = "Dataset"

# 폴더 내부의 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 이미지 크기를 10배로 확대
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # 이미지 열기
    img = Image.open(image_path)
    
    # 현재 이미지 크기
    width, height = img.size
    
    # 크기를 10배로 확대
    new_width = width * 10
    new_height = height * 10
    
    # 크기 변경
    img = img.resize((new_width, new_height))
    
    # 확대된 이미지 저장
    img.save(image_path)

print("이미지 크기 확대 완료.")
