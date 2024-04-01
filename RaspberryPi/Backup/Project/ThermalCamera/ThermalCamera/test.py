import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Setup I2C communication
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_shape = (24, 32)

frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts

# Initialize a list to store frame data
frame_data = []

def plot_update():
    mlx.getFrame(frame)  # Read MLX temperatures into the frame variable
    data_array = np.fliplr(np.reshape(frame, mlx_shape))  # Reshape, flip data

    # Apply Gaussian filter for additional noise reduction and smoothing
    data_array = ndimage.gaussian_filter(data_array, sigma=1.3)

    # Mask pixels outside human body temperature range (assuming °C)
    data_array[data_array < 25] = 0
    data_array[data_array > 38] = 0

    frame_data.append(data_array)  # Append the frame data

# 녹화 시간을 조절하려면 프레임 데이터를 얼마나 모을 것인지를 결정합니다.
# 현재 코드에서는 while 루프가 무한 루프로 설정되어 있으므로, 원하는 녹화 시간만큼 루프를 실행합니다.
# 아래의 예제는 10초 동안 녹화하는 코드입니다. 10초 동안 20프레임/초로 녹화하므로 200프레임을 모읍니다.
desired_recording_time_seconds = 10
desired_frame_count = int(desired_recording_time_seconds * 20)  # 20프레임/초로 설정

frame_data = []  # 프레임 데이터를 저장할 리스트

while len(frame_data) < desired_frame_count:
    t1 = time.monotonic()  # for determining frame rate
    try:
        plot_update()  # Update plot
    except:
        continue

    # Approximating frame rate
    t_array.append(time.monotonic() - t1)
    if len(t_array) > 10:
        t_array = t_array[1:]  # Recent times for frame rate approximation
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array) / np.sum(t_array)))

# 프레임 데이터가 모이면 비디오를 저장하고 종료합니다.
if len(frame_data) > 0:
    out = cv2.VideoWriter('recorded_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, mlx_shape)
    for frame_array in frame_data:
        out.write(np.uint8(frame_array))
    out.release()
