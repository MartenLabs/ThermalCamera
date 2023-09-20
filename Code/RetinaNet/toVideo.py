import cv2
import numpy as np
import os
import time
from tensorflow import keras

import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time


# Load RetinaNet model
model_path = '/Users/mac/Dev/Project/ThermalCamera/Model/keras-retinanet/snapshots/main110.h5'
model = models.load_model(model_path, backbone_name='resnet50')

# Load label to names mapping
labels_to_names = {0: 'person'}  # 본인의 라벨 딕셔너리로 업데이트

# 경로 설정
video_path = '2person.mp4'
output_folder = 'folder'
output_video_path = 'video.mp4'

# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 폴더 생성 (이미 존재할 경우 무시)
os.makedirs(output_folder, exist_ok=True)

frame_count = 0

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break


    # 프레임 저장
    frame_count += 1
    frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')

    # 프레임을 10배로 업스케일링
    frame_scaled = cv2.resize(frame, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(frame_filename, frame_scaled)


    # 프레임에서 바운딩 박스 그리기
    image = read_image_bgr(frame_filename)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    # 이미지 저장
    output_frame_filename = os.path.join(output_folder, f'frame_{frame_count}_annotated.jpg')
    cv2.imwrite(output_frame_filename, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

    # 프레임을 비디오에 추가
    out.write(frame)

# VideoCapture, VideoWriter 객체 해제
cap.release()
out.release()

# 마지막으로, 모든 이미지를 하나의 동영상으로 결합 (ffmpeg를 사용하여)
os.system(f"ffmpeg -framerate {fps} -i {output_folder}/frame_%d_annotated.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")
