import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import cv2
import torch
import datetime

current_datetime = datetime.datetime.now()

# YOLOv5 모델 로드 (PyTorch 필요)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x

# I2C 통신 설정.
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# MLX90640 센서 초기화.
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# 센서 배열의 형태를 정의합니다.
mlx_shape = (24, 32)

# 영상 파일 설정
output_filename = f"{current_datetime.strftime('%Y-%m-%d %H-%M-%S')}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 7  # 프레임 속도 (프레임/초)
out = cv2.VideoWriter(output_filename, fourcc, fps, mlx_shape[::-1], isColor=True)  # 가로와 세로를 반전

frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 총 픽셀 수: 768개


def capture_frame():
    mlx.getFrame(frame)
    data_array_raw = np.fliplr(np.reshape(frame, mlx_shape))
    data_array_smoothed = ndimage.gaussian_filter(data_array_raw, sigma=1.3)
    x_idx_valid_values, y_idx_valid_values = np.where(data_array_smoothed > 0)
    valid_values = data_array_smoothed[x_idx_valid_values, y_idx_valid_values]
    x_idx_all, y_idx_all = np.indices(data_array_smoothed.shape)
    interpolated_data_array = interpolate.griddata(
        (x_idx_valid_values, y_idx_valid_values), valid_values, (x_idx_all, y_idx_all), 'nearest'
    )

    # 인체 온도 범위 외의 픽셀 마스킹 (°C 가정).
    interpolated_data_array[(interpolated_data_array <= 25) | (interpolated_data_array > 40)] = 0

    img_array = (interpolated_data_array - np.min(interpolated_data_array)) / (
            np.max(interpolated_data_array) - np.min(interpolated_data_array)) * 255
    img_array = img_array.astype(np.uint8)
    # img_array = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_CUBIC)
    # img_array를 3차원으로 확장 (1 채널에서 3 채널로)
    expanded_img_array = np.expand_dims(img_array, axis=2)
    expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))

    # 필요한 경우 데이터 타입을 uint8로 변환
    expanded_img_array = expanded_img_array.astype(np.uint8)
    
    return expanded_img_array


def detect_objects(img):
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Color 이미지로 변환
    
    # img_array를 3차원으로 확장 (1 채널에서 3 채널로)
    expanded_img_array = np.expand_dims(img, axis=2)
    expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))

    # 필요한 경우 데이터 타입을 uint8로 변환
    expanded_img_array = expanded_img_array.astype(np.uint8)

    results = model(expanded_img_array)
    # 'person' 클래스에 해당하는 결과만 저장
    pred = results.pred[0]
    pred = pred[pred[:, -1] == 0]  # 'person' 클래스에 해당하는 결과만 선택

    # 이미지를 추출하고 반환
    img_with_objects = results.render()[0]
    return img_with_objects


try:
    t_start = time.monotonic()
    frame_count = 0

    while True:
        try:
            t_frame_start = time.monotonic()
            img_array = capture_frame()
            frame_count += 1
            # img_with_objects = detect_objects(img_array)
            # out.write(img_with_objects)  # imgs 속성에 그림이 그려진 이미지 존재
            out.write(img_array)
            t_frame = time.monotonic() - t_frame_start
            print('Frame Rate: {0:2.1f}fps'.format(1 / t_frame))

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

except KeyboardInterrupt:
    out.release()
    exit(1)

finally:
    t_total = time.monotonic() - t_start
    print('Average Frame Rate: {0:2.1f}fps'.format(frame_count / t_total))
    out.release()
    exit(1)
print("Recording finished.")