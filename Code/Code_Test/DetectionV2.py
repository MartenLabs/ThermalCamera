import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import cv2
import datetime

current_datetime = datetime.datetime.now()

# Darknet YOLO 모델 및 설정 파일 경로
yolo_cfg = '/home/pi/ThermalCamera/darknet/cfg/yolov3-tiny.cfg'
yolo_weights = '/home/pi/ThermalCamera/yolo.weights'

# Darknet 클래스 이름 파일 경로
yolo_classes = '/home/pi/ThermalCamera/darknet/cfg/coco.data'

# Darknet 모델 로드
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# 클래스 이름 로드
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

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
    img_array = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_CUBIC)
    return img_array


def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * img.shape[1])
                center_y = int(obj[1] * img.shape[0])
                width = int(obj[2] * img.shape[1])
                height = int(obj[3] * img.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


try:
    t_start = time.monotonic()
    frame_count = 0

    while True:
        try:
            t_frame_start = time.monotonic()
            img_array = capture_frame()
            frame_count += 1
            img_with_objects = detect_objects(img_array)
            out.write(img_with_objects)  # imgs 속성에 그림이 그려진 이미지 존재
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
