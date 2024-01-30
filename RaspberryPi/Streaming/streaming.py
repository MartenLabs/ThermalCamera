import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
from flask import Flask, Response, render_template, make_response
import cv2
import threading
import adafruit_mlx90640
import cv2
import datetime
import logging
from tensorflow.keras.models import load_model
from loss import ssim_loss  
from keras.utils import custom_object_scope

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


model_path = 'filter_model.h5'

# 모델 로드
with custom_object_scope({'ssim_loss': ssim_loss}):
    model = load_model(model_path)

class ThermalCamera:
    def __init__(self, frame_shape=(24, 32), fps=7):
        # 현재 시간을 가져와서 출력 파일 이름 생성
        self.current_datetime = datetime.datetime.now()

        # I2C 통신 설정
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        
        # MLX90640 센서 초기화
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        self.mlx_shape = frame_shape

        # 센서에서 읽어온 데이터를 저장하기 위한 배열 초기화
        self.frame = np.zeros(self.mlx_shape[0] * self.mlx_shape[1])  # 총 픽셀 수: 768개

    def capture_frame(self):
        # MLX90640 센서에서 데이터 프레임 get
        self.mlx.getFrame(self.frame)
        
        # 데이터를 올바른 형태로 변환
        data_array_raw = np.fliplr(np.reshape(self.frame, self.mlx_shape))
        
        # 데이터 정규화: 데이터 값을 0-255 범위로 변환
        data_array_raw_normalized_uint8 = ((data_array_raw - np.min(data_array_raw)) / (np.max(data_array_raw) - np.min(data_array_raw)) * 255).astype(np.uint8)
        # scaled_img_array = cv2.resize(data_array_raw_normalized_uint8, (self.mlx_shape[1]*4, self.mlx_shape[0]*4), interpolation=cv2.INTER_LINEAR)
        # print(scaled_img_array.shape)
        # return expanded_img_array
        # expanded_img_array = np.expand_dims(data_array_raw_normalized_uint8, axis=-1)
        expanded_img_array = np.expand_dims(data_array_raw_normalized_uint8, axis= 0)
        # expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))
        return expanded_img_array.astype(np.uint8)


def draw_bounding_boxes(image, number):
    try:
        print(image.shape)
        # 임계값을 적용하여 열원을 탐지
        _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # 좌표 추출
        points = np.column_stack(np.where(thresholded.transpose() > 0))
        if len(points) < 1:
            return image  # 점이 없으면 원본 이미지 반환
        
        # DBSCAN 클러스터링을 실행
        clustering = DBSCAN(eps=2, min_samples=2).fit(points)
        labels = clustering.labels_

        # 각 레이블에 대한 포인트를 그룹화하고 각 그룹에 대해 바운딩 박스를 계산
        bboxes = []
        for label in np.unique(labels):
            if label == -1:
                continue  # 노이즈는 무시한다.
            # 현재 레이블의 모든 포인트를 선택한다.
            label_points = points[labels == label]
            if len(label_points) == 0:
                return image
            x, y, w, h = cv2.boundingRect(label_points.astype(np.int32))
            bboxes.append((x, y, w, h))

        # 가장 큰 바운딩 박스를 선택
        bboxes = sorted(bboxes, key=lambda b: b[2]*b[3], reverse=True)[:number]

        # 바운딩 박스를 이미지에 그린다.
        for (x, y, w, h) in bboxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return image
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        return image  # 오류 발생 시 원본 이미지를 반환



def predict(frame):
    # 모델에 전처리된 프레임을 적용
    noise_removed_image, people_count = model.predict(frame)
    noise_removed_image = ((noise_removed_image - np.min(noise_removed_image)) / (np.max(noise_removed_image) - np.min(noise_removed_image)) * 255).astype(np.float32)
    
    image_with_boxes = draw_bounding_boxes(np.array(noise_removed_image), people_count)
    image_with_boxes = np.tile(noise_removed_image[0], (1, 1, 3))

    # 이미지 크기를 4배 확대
    scaled_img_array = cv2.resize(image_with_boxes, (image_with_boxes.shape[1]*4, image_with_boxes.shape[0]*4), )
    # interpolation=cv2.INTER_LINEAR
    return scaled_img_array, np.array(people_count[0]).argmax()

def generate_frames():
    camera = ThermalCamera()  # ThermalCamera 인스턴스 생성
    while True:
        frame = camera.capture_frame()  # 센서에서 프레임 캡처
        noise_removed_image, people_count = predict(frame)
        ret, buffer = cv2.imencode('.jpg', noise_removed_image)  # JPEG 형식으로 인코딩
        streaming_frame = buffer.tobytes()
        
        # 멀티파트 스트리밍 형식으로 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + streaming_frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/no-cache')
def no_cache():
    response = make_response("This response is not cached")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)